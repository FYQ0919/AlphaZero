import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
  def __init__(self, in_dims, out_dims, lr=0.001):
    super(ActorCritic, self).__init__()
    
    self.fc_actor = nn.Sequential(
      nn.Linear(in_dims, 32), nn.ReLU(),
      nn.Linear(32, 32), nn.ReLU(),
      nn.Linear(32, out_dims),
      nn.Softmax(dim=-1)
    )

    self.fc_critic = nn.Sequential(
      nn.Linear(in_dims, 32), nn.ReLU(),
      nn.Linear(32, 32), nn.ReLU(),
      nn.Linear(32, 1),
      nn.Tanh(),
    )
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, state):
    action = self.fc_actor(state)
    value  = self.fc_critic(state)
    return action, value

  def predict(self, obs):
    obs = torch.tensor(obs).float().to('cpu') 
    obs = obs.view(1, obs.shape[0])
    policy, value = self.forward(obs)
    return policy.detach().numpy()[0], value.detach().numpy()[0]

  def load_weights(self, weights):
    self.load_state_dict(weights)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}



if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')
  model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

