import math
import torch
import numpy as np


class Node():
  def __init__(self, prior, player):
    self.visits = 0
    self.value  = 0
    self.prior  = prior   # prior policy probability
    self.player = player
    self.children = {} 
    self.state = None

  def Value(self):
    NotImplemented

  def expanded(self):
    NotImplemented

  def expand(self, state, player, action_probs):
    NotImplemented

  def select_child(self):
    NotImplemented

  def select_action(self, temperature=0):
    NotImplemented

def MCTS(model, state, player, env, simulations=100):
  root = Node(0, player)
  
  ## EXPAND root
  action_probs, _ = model.predict(state)
  #valid_moves = env.get_valid_moves(state)

  ##action_probs[valid_moves] = 1  # mask invalid moves
  #action_probs = action_probs * valid_moves  # mask invalid moves
  #print( action_probs)

  #action_probs /= np.sum(action_probs)
  #root.expand(state, player, action_probs)


  pass


if __name__ == '__main__':

  from game import TicTacToe
    
  class MockModel():
    def __init__(self, in_dims, out_dims): pass
    # asssume current board state is: [1, 0, 0, -1, 1, 0, 0, 0, -1] : player 1
    # [ 1, 0, 0,  
    #  -1, 1, 0, 
    #   0, 0,-1 ]
    def predict(self, state):
      return np.array([0.10, 0.66, 0.75, 0.10, 0.10, 0.55, 0.75, 0.75, 0.10]), 0.0001

  env = TicTacToe()
  obs = env.reset()
  net = MockModel(env.observation_space, env.action_space)
  player = obs[-2]

  root = MCTS( net, obs, player, env, simulations=100)

