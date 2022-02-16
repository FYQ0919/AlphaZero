import math
import torch
import numpy as np

class Node:
  def __init__(self, prior, player):
    self.visits = 0
    self.value = 0
    self.prior  = prior   # prior policy probability
    self.action = player
    self.children = {} 
    self.state = None

  def Value(self):
    if self.visits == 0:
        return 0
    return self.value / self.visits

  def expanded(self):
    return len(self.children) > 0

  def expand(self, state, player, action_probs):
    # We expand a node and keep track of the prior policy probability given by neural network
    self.player = player
    self.state = state
    for a, prob in enumerate(action_probs):
      if prob != 0:
        self.children[a] = Node(prior=prob, player=player * -1)

  def select_child(self):
    best_child = None
    best_score = -np.inf
    best_action = -1
    parent = self
    for act, child in self.children.items():
      # child's value from opponenet's view
      value_score = -child.Value() if child.visits > 0 else 0
      prior_score = child.prior * math.sqrt(parent.visits) /(1+child.visits)
      UCBscore = prior_score + value_score
      if UCBscore > best_score:
        best_score = UCBscore
        best_child = child
        best_action = act
      
    return best_action, best_child


class MCTS:
  def __init__(self, game):
    self.env = game
    self.simulations = 3

  def run(self, model, state, player):
    root = Node(0, player)

    ## EXPAND root
    action_probs, _ = model.predict(state)
    valid_moves = self.env.get_valid_moves(state)
    action_probs[valid_moves] = 1  # mask invalid moves
    action_probs /= np.sum(action_probs)
    root.expand(state, player, action_probs)

    for _ in range(self.simulations):
      node = root 
      search_path = [node]  

      ## SELECT
      while node.expanded():
        action, node = node.select_child()
        search_path.append(node)
      
      parent = search_path[-2] # ??????? lists will allways have 2 elements parent and best child??
      state = parent.state     

      # Now we're at a leaf node and we would like to expand
      # Players always play from their own perspective
      next_state, _ = self.env.get_next_state(state, player=1, action=action)
      # Get the board from the perspective of the other player
      next_state = self.env.get_canonical_board(next_state, player=-1)

      ## EXPAND AND ROLLOUT
      # The value of the new state from the perspective of the other player, None if game is not over
      value = self.env.get_reward_for_player(next_state, player=1)
      if value is None:
        action_probs, value = model.predict(state)
        valid_moves = self.env.get_valid_moves(state)
        action_probs[valid_moves] = 1  # mask invalid moves
        action_probs /= np.sum(action_probs)
        node.expand(next_state, -parent.player, action_probs)
      self.backpropagate(search_path, value, -parent.player)

  def backpropagate(self, search_path, value, player):
    """
    At the end of a simulation, we propagate the evaluation all the way up the tree
    to the root.
    """
    for node in reversed(search_path):
      node.value += value if node.player == player else -value
      node.visits+= 1

from Game import Connect2Game
from network import ActorCritic

env = Connect2Game()
net = ActorCritic(env.observation_space, env.action_space)

state = env.reset() 

mcts = MCTS(game=env)
mcts.run( net, state, 1)


