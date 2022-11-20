from curling_simulator.gym_env import *
from copy import deepcopy
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box

class Curling:
  def __init__(self, config=None, render=False):
      self.env = CurlingTwoAgentGymEnv_v0(render=render)
      self.action_space = self.env.action_space
      self.observation_space = self.env.observation_space

  def reset(self):
      return self.env.reset()

  def step(self, action):
      obs, rew, done, info = self.env.step(action)
      return obs, rew, done, info

  def get_state(self):
      return self.env.get_state()

  def set_state(self,state,render):
      self.env.set_state(state,render)

  def to_play(self):
      return self.env.to_play()

  def render(self):
      self.env.render()

  def close(self):
      self.env.close()
# if __name__ == '__main__':
#   env = Curling()
#   obs = env.reset()
#   scores = []
#   while True:
#     act = env.action_space.sample()
#     obs, rew, done, info = env.step(act)

