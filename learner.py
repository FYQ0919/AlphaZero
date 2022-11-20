import ray
from network import ActorCritic
import torch
from torch.utils.tensorboard import SummaryWriter
import time

@ray.remote
class Learner(object):

    def __init__(self, input_dim, out_dim, buffer, storage):
        self.buffer = buffer
        self.storage = storage
        self.device = 'cpu'
        self.network = ActorCritic(input_dim, out_dim).to(self.device)
        self.network.to(self.device)
        self.network.train()
        self.writer = SummaryWriter("./log/ray/train/")
        self.training_step = 0
        self.batch_size = 256
        self.max_training_step = 1e7


    def train(self):
        self.send_weights()
        while ray.get(self.buffer.size.remote()) < 1000:
            time.sleep(1)

        print("Start Training")

        current_turn = ray.get(self.buffer.size.remote()) // 100
        while self.training_step < self.max_training_step:

            not_ready_batches = [self.buffer.sample.remote(self.batch_size) for _ in range(10)]
            while len(not_ready_batches) > 0:
                self.current_turn = ray.get(self.buffer.size.remote()) // 100
                if current_turn == self.current_turn:
                    ready_batches, not_ready_batches = ray.wait(not_ready_batches, num_returns=1)

                    batch = ray.get(ready_batches[0])

                    if self.training_step % 500 == 0:
                        self.send_weights()

                    states, actions, values = batch
                    pi, v = self.network(states)

                    policy_loss = -(actions * torch.log(pi)).sum(dim=1)  # *(1-dones)
                    policy_loss = policy_loss.mean()
                    value_loss = torch.sum((values - v.view(-1)) ** 2) / values.size()[0]
                    # value_loss = torch.sum(F.smooth_l1_loss(values, v.view(-1)))

                    loss = policy_loss + value_loss
                    self.writer.add_scalar("value_loss", value_loss, self.training_step)
                    self.writer.add_scalar("policy_loss", policy_loss, self.training_step)

                    self.network.optimizer.zero_grad()
                    loss.backward()
                    self.network.optimizer.step()

                    self.training_step += 1
                    if (self.training_step) % 5000 == 0:
                        torch.save(self.network.state_dict(), f"./log/ray/{self.training_step}.pkl")
                        current_turn += 1

    def send_weights(self):
        self.storage.store_weights.remote(self.network.get_weights(), self.training_step)

    def launch(self):
        print("Learner is online on {}.".format(self.device))
        self.train()
