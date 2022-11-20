import math
import random
import numpy as np
import gym
from network import ActorCritic
from curling import Curling

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

device='cpu'

class MinMaxStats(object):

  def __init__(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound

  def update(self, value: float):
    self.minimum = min(self.minimum, value)
    self.maximum = max(self.maximum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    elif self.maximum == self.minimum:
      return 1.0
    return value

  def reset(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound

class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000, device=device):
    self.states  = np.zeros((length, obs_dim))
    self.actions = np.zeros((length, act_dim))
    self.values = np.zeros(length)
    self.size = length
    self.idx  = 0

  def __len__(self): return self.idx

  def store(self, obs, action, values):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = action
    self.values[idx] = values

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(device)
    values  = torch.tensor( self.values[indices], dtype=torch.float).to(device)
    return states, actions, values

class Node:
  def __init__(self, prior: float):
    self.visit_count = 0
    self.value_sum = 0
    self.prior = prior   # prior policy probabilities
    self.children = {} 
    self.hidden_state = None
    self.reward = 0
    self.to_play = -1
    self.done = False
    self.env_state = None  # we store a copy of the OpenAi Gym library


  def value(self) -> float:
    if self.visit_count == 0:
        return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> bool:
    return len(self.children) > 0
 
class AlphaZero:
  def __init__(self, in_dims, out_dims, actor_key = 0):
    self.model = ActorCritic(in_dims, out_dims).to(device)
    self.memory = ReplayBuffer(in_dims, out_dims, device=device)
    self.train_steps = 0
    self.writer = SummaryWriter(f"./log/actor_{actor_key}/")
    self._env = Curling()
    self.min_max_stats = MinMaxStats()

  def load_model(self, dict):
    self.model.load_state_dict(dict)

  def softmax(self, x):#
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def store(self, obs,action,value):
    self.memory.store(obs,action,value)
    
  def ucb_score(self, parent: Node, child: Node, MinMaxStats=None) -> float:
    # a Node's score is based on its value, plus an exploration bonus based on the prior.
    value = -child.value() if child.visit_count > 0 else 0
    value_score = self.min_max_stats.normalize(child.reward + value)
    prior_score = child.prior * math.sqrt(parent.visit_count) /(child.visit_count+1)
    return prior_score + value_score

  def select_child(self, node: Node):
    # We select child using UCT
    out = [(self.ucb_score(node,child),action,child) for action, child in node.children.items()]
    smax = max([x[0] for x in out])     # this max is why it favors 1's over 0's
    _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
    return action, child

  def predict(self, obs):

      policy, value = self.model(torch.tensor(obs))

      return policy, value

  def MCTS(self, env_state, observation, num_simulations=600):
    self.min_max_stats.reset()
    # init root node
    root = Node(0)
    ## EXPAND root node
    policy, value = self.model(torch.tensor(observation))
    root.to_play = env_state.to_play()
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])
      root.children[i].state = env_state.get_state()
      root.children[i].done = False

    # run mcts
    for j in range(num_simulations):

      print(j)

      action_history = [] 
      node = root 
      search_path = [node] # nodes in the tree that we select
      to_play = root.to_play
      # for _ in range(tree_depth): 
      # move code below under a loop to increase tree depth

      ## SELECT: traverse down the tree according to the ucb_score 
      while node.expanded() and not node.done:
        action, node = self.select_child(node)
        action_history.append(action)
        search_path.append(node)
        to_play = 1 - to_play

      # now we are at a leaf which is not "expanded", run the dynamics model

      self._env.set_state(node.state,render=True)
      next_state, node.reward, node.done, _ = self._env.step(action_history[-1])
      node.to_play = to_play

      if not node.done:
        ## EXPANED create all the children of the newly expanded node
        policy, value = self.model(torch.tensor(next_state))
        for i in range(policy.shape[0]):
          node.children[i] = Node(prior=policy[i])
          node.children[i].state = self._env.get_state()

        # BACKPROPAGATE: update the state with "backpropagate"
        idx = 0
        for bnode in reversed(search_path):

          bnode.visit_count += 1

          if bnode.to_play != root.to_play:
            bnode.value_sum -= value
            reward = -bnode.reward
          else:
            bnode.value_sum += value
            reward = bnode.reward

          if idx < len(search_path) - 1:
              new_q = bnode.reward - bnode.value()
              self.min_max_stats.update(new_q)

          value = reward +  value
          idx += 1


      else:
        idx = 0
        policy, value = self.model(torch.tensor(next_state))
        for bnode in reversed(search_path):

          bnode.visit_count += 1

          if bnode.to_play != root.to_play:
            bnode.value_sum -= value
            reward = -bnode.reward
          else:
            bnode.value_sum += value
            reward = bnode.reward

          if idx < len(search_path) - 1:
            new_q = bnode.reward - bnode.value()
            self.min_max_stats.update(new_q)

          value = reward + value
          idx += 1

    # Each node represents a potential action, number of visits to each node - normalized
    # (by a softmax) represent the probabilty of taking that action. This is our policy
    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    visit_counts = [x[1] for x in sorted(visit_counts)]
    av = np.array(visit_counts).astype(np.float64)
    policy = self.softmax(av)

    return policy, value, root

  def train(self, batch_size=64):
    if(len(self.memory) >= 100):
      for i in range(10):
        states, actions, values = self.memory.sample(batch_size)
        pi, v = self.model(states)

        policy_loss = -(actions * torch.log(pi)).sum(dim=1) #*(1-dones)
        policy_loss = policy_loss.mean()
        value_loss = torch.sum((values-v.view(-1))**2)/values.size()[0]
        #value_loss = torch.sum(F.smooth_l1_loss(values, v.view(-1)))

        loss = policy_loss + value_loss
        self.writer.add_scalar("value_loss", value_loss, self.train_steps)
        self.writer.add_scalar("policy_loss", policy_loss, self.train_steps)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        self.train_steps += 1
        if (self.train_steps) % 100 == 0:
            torch.save(self.model.state_dict(), f"./save_model/{self.train_steps}.pkl")

    pass


if __name__ == '__main__':

  env = Curling()
  agent = AlphaZero(env.observation_space.shape[0], env.action_space.n)
  game_completed = 0
  start = datetime.datetime.now()

  for epi in range(10000):
    obs = env.reset()
    done = False
    while not done:
      policy, value, _ = agent.MCTS(env, obs, 20)
      action = np.argmax(policy)
      n_obs, reward, done, info = env.step(action)

      agent.store(obs,policy,value)

      obs = n_obs
      agent.train()
      game_completed += 1

    time_now = datetime.datetime.now()

    print(f"complete game {game_completed}, used time = {(time_now - start).seconds}s")

