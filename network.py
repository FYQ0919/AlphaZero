import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ActorCritic(nn.Module):
  def __init__(self, in_space, out_space, lr=0.0001):
    super(ActorCritic, self).__init__()
    
    self.fc_actor = nn.Sequential(
      nn.Linear(in_space.shape[0], 16), nn.ReLU(),
      nn.Linear(16, 16), nn.ReLU(),
      nn.Linear(16, out_space.n),
      nn.Softmax(dim=-1)
    )

    self.fc_critic = nn.Sequential(
      nn.Linear(in_space.shape[0], 16), nn.ReLU(),
      nn.Linear(16, 16), nn.ReLU(),
      nn.Linear(16, 1),
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



if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')
  model = ActorCritic(env.observation_space, env.action_space)

