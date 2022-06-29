import math
import random
#import torch
import numpy as np


class Node:
  def __init__(self, prior: float):
    self.visits = 0
    self.value  = 0
    self.prior  = prior   # prior policy probability
    self.player = player
    self.children = {} 
    self.hidden_state = None
    self.to_play = None

  def value(self) -> bool:
    if self.visits == 0:
        return 0
    return self.value_sum / self.visits_count

  def expanded(self) -> float:
    return len(self.children) > 0

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node, MinMaxStats=None) -> float:
  #pb_c =  math.log((parent.visit_count + pb_c_base + 1) /pb_c_base) + pb_c_init
  #pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
  #prior_score = pb_c * child.prior
  #if child.visit_count > 0:
  #  if min_max_stats is not None:
  #    value_score = child.reward + discount * min_max_stats.normalize(child.value())
  #  else:
  #    value_score = child.reward + discount * child.value()
  #else:
  #  value_score = 0

  value_score = -child.Value() if child.visits > 0 else 0
  prior_score = child.prior * math.sqrt(parent.visits) /(child.visits+1)
  return prior_score + value_score

def select_child(node: Node, min_max_stats=None):
  out = [(ucb_score(node,child,min_max_stats),action,child)for action, child in node.children.items()]
  smax = max([x[0] for x in out]) # this max is why it favors 1's over 0's
  _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
  return action, child



def MCTS(model, observation, num_simulations=10, minimax=True):
  # init root node
  root = Node(0) 
  root.to_play = observation[-2]  # determine who to play

  ## EXPAND root node
  policy, value = model.predict(observation)
  for i in range(policy.shape[0]):
    root.children[i] = Node(policy[i])
    root.children[i].to_play = -root.to_play

  # run mcts
  for _ in range(num_simulations):
    action_history = []
    node = root 
    search_path = [node]  

    ## SELECT
    while node.expanded():
      #action, node = select_child(node, min_max_stats)
      action, node = select_child(node)
      action_history.append(action)
      search_path.append(node)

    # now we are at a leaf which is not "expanded" 
    parent = search_path[-2]

    ## EXPAND AND ROLLOUT
    env = TicTacToe(parent.hidden_state)
    next_state,value,done = env.step(action_history[-1])

    if not done:
      policy, value = model.predict(parent.hidden_state)
      #valid_moves = env.get_valid_moves(state)

      #action_probs = action_probs * valid_moves  # mask invalid moves
      #action_probs /= np.sum(action_probs)
      #node.expand(next_state, -parent.player, action_probs)

      for i in range(policy.shape[0]):
        node.children[i] = Node(prior=policy[i])
        node.children[i].to_play = -node.to_play
    #backpropagate(search_path, value, -parent.player)

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
  #root = MCTS(env, net, obs, num_simulations=10, minimax=True)

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
