import torch
import numpy as np
class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000, device='cpu'):
    self.states  = np.zeros((length, obs_dim))
    self.actions = np.zeros((length, act_dim))
    self.rewards = np.zeros(length)
    self.values = np.zeros(length)
    self.dones   = np.zeros(length, dtype=bool)
    self.size = length
    self.idx  = 0
    self.device = device


  def __len__(self): return self.idx

  def store(self, obs, action, reward, values, done):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.values[idx] = values
    self.dones[idx] = done

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(self.device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(self.device)
    rewards = torch.tensor( self.rewards[indices], dtype=torch.float).to(self.device)
    values  = torch.tensor( self.values[indices], dtype=torch.float).to(self.device)
    dones   = torch.tensor( self.dones[indices] ).float().to(self.device)
    return states, actions, rewards, values, dones
 
if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')
  memory = ReplayBuffer(env.observation_space.shape[0], env.action_space.n)

