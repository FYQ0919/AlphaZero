import math
import torch
import numpy as np

class Node:
  def __init__(self, prior, player):
    self.visits = 0
    self.value  = 0
    self.prior  = prior   # prior policy probability
    self.player = player
    self.children = {} 
    self.state = None

  def Value(self) -> bool:
    if self.visits == 0:
        return 0
    return self.value / self.visits

  def expanded(self) ->float:
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

  def select_action(self, temperature=0):
    visit_counts = np.array([c.visits for c in self.children.values()])
    actions = [action for action in self.children.keys()]
    if temperature == 0:
      action = actions[np.argmax(visit_counts)]
    elif temperature == float("inf"):
      action = np.random.choice(actions)
    else:
      dist = visit_counts ** (1 / temperature)
      dist = dist / sum(dist)
      action = np.random.choice(actions, p=dist)
    return action

def backpropagate(search_path, value, player):
  """
  At the end of a simulation, we propagate the evaluation all the way up the tree
  to the root.
  """
  for node in reversed(search_path):
    node.value += value if node.player == player else -value
    node.visits+= 1


def MCTS(model, state, player, env, simulations=100):
  root = Node(0, player)

  ## EXPAND root
  action_probs, _ = model.predict(state)
  valid_moves = env.get_valid_moves(state)

  #action_probs[valid_moves] = 1  # mask invalid moves
  action_probs = action_probs * valid_moves  # mask invalid moves
  print( action_probs)

  action_probs /= np.sum(action_probs)
  root.expand(state, player, action_probs)

  for _ in range(simulations):
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
    next_state, _ = env.get_next_state(state, player=1, action=action)
    # Get the board from the perspective of the other player
    next_state = env.get_canonical_board(next_state, player=-1)

    ## EXPAND AND ROLLOUT
    # The value of the new state from the perspective of the other player, None if game is not over
    value = env.get_reward_for_player(next_state, player=1)
    if value is None:
      action_probs, value = model.predict(next_state)
      valid_moves = env.get_valid_moves(state)
      #action_probs[valid_moves] = 1  # mask invalid moves
      action_probs = action_probs * valid_moves  # mask invalid moves
      action_probs /= np.sum(action_probs)
      node.expand(next_state, -parent.player, action_probs)
    backpropagate(search_path, value, -parent.player)
  return root

if __name__ == '__main__':

  from Game import Connect2Game
  from network import ActorCritic
    
  class MockModel():
    # starting board is: [0, 0, 1, -1]
    def predict(self, state):
      return np.array([0.26, 0.24, 0.24, 0.26]), 0.0001

  #net = ActorCritic(env.observation_space, env.action_space)
  net = MockModel()
  env = Connect2Game()

  state = env.reset() 
  root = MCTS( net, state, 1, env,simulations=100)


