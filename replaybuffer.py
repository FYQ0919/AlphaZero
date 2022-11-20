import ray
import numpy as np
import torch

@ray.remote
class ReplayBuffer(object):
  def __init__(self, input_dim, out_dim, buffer_size):
    self.states = np.zeros((buffer_size, input_dim))
    self.actions = np.zeros((buffer_size, out_dim))
    self.values = np.zeros(buffer_size)
    self.size = buffer_size
    self.idx = 0
    self.device = 'cpu'
    self.load_buffer(idx = 5000)

  def __len__(self): return self.idx

  def size(self):
      return self.idx

  def store(self, obs, action, values):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = action
    self.values[idx] = values

    if self.idx % 100 == 0:
      print(f'Collect {self.idx} data')
      self.save_buffer()

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(self.device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(self.device)
    values  = torch.tensor( self.values[indices], dtype=torch.float).to(self.device)
    return states, actions, values

  def save_buffer(self):
    np.save(f"state_{self.idx}.npy", self.states)
    np.save(f"actions_{self.idx}.npy", self.actions)
    np.save(f"values_{self.idx}.npy", self.values)

  def load_buffer(self,idx):
    self.states = np.load("state_" + f"{idx}.npy", allow_pickle=True)
    self.actions = np.load("actions_" + f"{idx}.npy", allow_pickle=True)
    self.values = np.load("values_" + f"{idx}.npy", allow_pickle=True)
    self.idx = idx
