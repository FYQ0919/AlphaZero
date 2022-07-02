
from copy import deepcopy
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box

## https://stackoverflow.com/questions/57839665/how-to-set-a-openai-gym-environment-start-with-a-specific-state-not-the-env-res
class CartPole:
  def __init__(self, config=None):
      self.env = gym.make("CartPole-v1")
      self.env = gym.wrappers.RecordEpisodeStatistics(self.env)
      self.action_space = Discrete(2)
      self.observation_space = self.env.observation_space

  def reset(self):
      return self.env.reset()

  def step(self, action):
      obs, rew, done, info = self.env.step(action)
      return obs, rew, done, info

  def set_state(self, state):
      self.env = deepcopy(state)
      obs = np.array(list(self.env.unwrapped.state))
      return obs

  def get_state(self):
      return deepcopy(self.env)

  def render(self):
      self.env.render()

  def close(self):
      self.env.close()

if __name__ == '__main__':
  env = CartPole()
  obs = env.reset()
  scores = []
  while True:

    tmp = env.get_state()
    env = CartPole()
    env.set_state(tmp)

    act = env.env.action_space.sample()
    n_obs, rew, done, info = env.step(act)
    obs = n_obs
    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Return: {scores[-1]}, Avg return: {avg_score}")
      break
