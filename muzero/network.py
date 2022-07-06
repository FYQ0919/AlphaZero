import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Model(f,g,h)
# representation:  s_0 = h(o_1, ..., o_t)
# dynamics:        r_k, s_k = g(s_km1, a_k)
# prediction:      p_k, v_k = f(s_k)

class Representation(nn.Module):
  def __init__(self, in_dims, out_dims, lr=0.0005):
    super(Representation, self).__init__()
    self.fc_1 = nn.Linear(in_dims, 32)
    self.fc_2 = nn.Linear(32, 32) 
    self.fc_3 = nn.Linear(32, in_dims)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc_1(x))
    x = F.relu(self.fc_2(x)) 
    state = self.fc_3(x)
    return state

class Dynamics(nn.Module):
  def __init__(self, in_dims, out_dims, lr=0.0005):
    super(Dynamics, self).__init__()
    self.fc_1 = nn.Linear(in_dims+out_dims, 32)
    self.fc_2 = nn.Linear(32, 32) 
    self.fc_s = nn.Linear(32, in_dims) # change this
    self.fc_r = nn.Linear(32, 1)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, s, a):
    x = torch.cat([s,a],dim=0)
    x = F.relu(self.fc_1(x))
    x = F.relu(self.fc_2(x)) 
    nstate = self.fc_s(x)
    reward = self.fc_r(x)
    return nstate, reward

class Prediction(nn.Module):
  def __init__(self, in_dims, out_dims, lr=0.0001):
    super(Prediction, self).__init__()
    self.fc_1 = nn.Linear(in_dims, 32)
    self.fc_2 = nn.Linear(32, 32) 
    self.fc_v = nn.Linear(32, 1)
    self.fc_pi = nn.Linear(32, out_dims)
    self.optimizer = optim.Adam(self.parameters(), lr=lr)

  def forward(self, x):
    x = F.relu(self.fc_1(x))
    x = F.relu(self.fc_2(x)) 
    value  = self.fc_v(x)
    policy = self.fc_pi(x)
    return policy, value
class Model:
  def __init__(self, in_dims, out_dims):
    self._g = Dynamics(in_dims, out_dims)
    self._f = Prediction(in_dims, out_dims)
    self._h = Representation(in_dims, out_dims)
  def h(self, obs):
    obs = torch.tensor(obs)
    return self._h(obs)

  def g(self, s, a):
    nstate, reward = self._g(s, a)
    return reward.detach().numpy()[0], nstate

  def f(self, s):
    policy, value = self._f(s)
    action = policy.detach().numpy().argmax()
    return action, policy, value

if __name__ == '__main__':
  import gym
  env = gym.make('CartPole-v1')

  # representation:  s_0 = h(o_1, ..., o_t)
  # dynamics:        r_k, s_k = g(s_km1, a_k)
  # prediction:      p_k, v_k = f(s_k)


  mm = Model(env.observation_space.shape[0], env.action_space.n)

  for epi in range(50):
    obs = env.reset()
    done = False
    score, _score = 0, 0
    while not done:
      _state = mm.h(obs)
      action, policy, _value = mm.f(_state)
      _reward, _nstate = mm.g(_state, policy)

      n_obs, reward, done, _ = env.step(action)
      score += reward
      _score += _reward
      obs = n_obs
    print(f'Episode:{epi}  Score:{score} Predicted score:{_score}')
  env.close()



