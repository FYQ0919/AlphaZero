import math
import random
import numpy as np

from game import TicTacToe
from network import ActorCritic

class Node:
  def __init__(self, prior: float):
    self.visit_count = 0
    self.value_sum = 0
    self.prior = prior   # prior policy probabilities
    self.children = {} 
    self.hidden_state = None
    self.reward = 0
    self.to_play = -1

  def value(self) -> bool:
    if self.visit_count == 0:
        return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> float:
    return len(self.children) > 0

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum()

# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(parent: Node, child: Node, MinMaxStats=None) -> float:
  value_score = -child.value() if child.visit_count > 0 else 0
  prior_score = child.prior * math.sqrt(parent.visit_count) /(child.visit_count+1)
  return prior_score + value_score

def select_child(node: Node, min_max_stats=None):
  out = [(ucb_score(node,child,min_max_stats),action,child)for action, child in node.children.items()]
  smax = max([x[0] for x in out])     # this max is why it favors 1's over 0's
  _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
  return action, child


def MCTS(model, observation, num_simulations=10, minimax=True):
  # init root node
  root = Node(0) 
  root.to_play = observation[-2]  # determine who to play

  ## EXPAND the children of the root node
  policy, value = model.predict(observation)
  for i in range(policy.shape[0]):
    root.children[i] = Node(prior=policy[i])
    root.children[i].to_play = -root.to_play

  # run mcts
  for _ in range(num_simulations):
    action_history = []
    node = root 
    search_path = [node]  

    ## SELECTION
    while node.expanded():
      action, node = select_child(node)
      action_history.append(action)
      search_path.append(node)

    # Now we're at a leaf node and we would like to expand
    parent = search_path[-2]
    env = TicTacToe(parent.hidden_state)

    #expansion
    next_state, node.reward, _ = env.step(action_history[-1])

    # EXPANSION create all the children of the newly expanded node
    policy, value = model.predict(next_state)
    for i in range(policy.shape[0]):
      node.children[i] = Node(prior=policy[i])
      node.children[i].to_play = -node.to_play

    # BACKPROPAGATION update the state with 
    for bnode in reversed(search_path):
      bnode.value_sum += value
      bnode.visit_count += 1
      discount = 0.95
      value = bnode.reward + discount * value

  # output the final policy
  visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
  visit_counts = [x[1] for x in sorted(visit_counts)]
  av = np.array(visit_counts).astype(np.float64)
  policy = softmax(av)
  return policy, root

if __name__ == '__main__':
    
  env = TicTacToe()
  net = ActorCritic(env.observation_space.shape[0], env.action_space.n)

  # computer can play against itself...and tie!
  done = False
  state = env.reset()
  while not done:
    policy, node = MCTS(net, state, 10)
    #print(policy)
    act = np.random.choice(np.arange(len(policy)), p=policy)
    #act = policy.argmax()
    print(act)
    nstate, reward, done = env.step(act)
    state = nstate
    env.render()	
