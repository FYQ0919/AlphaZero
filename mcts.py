import math
import torch
import numpy as np


class Node:
  def __init__(self, prior: float):
    self.visits = 0
    self.value  = 0
    self.prior  = prior   # prior policy probability
    self.player = player
    self.children = {} 
    self.state = None
    self.to_play = None

  def value(self) -> bool:
    if self.visits == 0:
        return 0
    return self.value_sum / self.visits_count

  def expanded(self) ->float:
    return len(self.children) > 0


def MCTS(model, observation, num_simulations=10, minimax=True):
  # init root node
  root = Node(0) 
  root.to_play = observation[-2]  # determine who to play

  ## EXPAND root node
  policy, value = model.predict(observation)
  for i in range(policy.shape[0]):
    root.children[i] = Node(policy[i])
    root.children[i].to_play = -root.to_play

  return root



if __name__ == '__main__':

  from game import TicTacToe
    
  class MockModel():
    def __init__(self, in_dims, out_dims): pass
    # asssume current board state is: [1, 0, 0, -1, 1, 0, 0, 0, -1] : player 1
    # [ 1, 0, 0,  
    #  -1, 1, 0, 
    #   0, 0,-1 ]
    def predict(self, state):
      return np.array([0.10, 0.66, 0.75, 0.10, 0.10, 0.55, 0.75, 0.75, 0.10]), 0.7

  env = TicTacToe()
  obs = env.reset()
  net = MockModel(env.observation_space, env.action_space)
  player = obs[-2]

  root = MCTS(net, obs, num_simulations=10, minimax=True)

  # computer can play against itself...and tie!
  #net = MockModel(env.observation_space, env.action_space)
  #env = TicTacToe()
  #done = False
  #while not done:
  #  policy, node = mcts(net, env.state, 2000)
  #  print(policy)
  #  act = np.random.choice(list(range(len(policy))), p=policy)
  #  print(act)
  #  _, _, done, _ = gg.step(act)
  #  gg.render()	
