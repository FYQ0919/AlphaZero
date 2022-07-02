import math
import random
import numpy as np
import torch

from game import TicTacToe
from network import ActorCritic

device='cpu'
class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000):
    self.states  = np.zeros((length, obs_dim))
    self.actions = np.zeros((length, act_dim))
    self.rewards = np.zeros(length)
    self.values = np.zeros(length)
    self.dones   = np.zeros(length, dtype=bool)
    self.size = length
    self.idx  = 0

  def __len__(self): return self.idx

  def store(self, obs, action, reward, values, done):
    idx = self.idx % self.size
    self.idx += 1

    self.states[idx]  = obs
    self.actions[idx] = action
    self.rewards[idx] = reward
    self.values[idx] = values
    self.dones[idx] = done

  def sample(self, batch_size):
    indices = np.random.choice(self.size, size=batch_size, replace=False)
    states  = torch.tensor( self.states[indices] , dtype=torch.float).to(device)
    actions = torch.tensor( self.actions[indices], dtype=torch.float).to(device)
    rewards = torch.tensor( self.rewards[indices], dtype=torch.float).to(device)
    values  = torch.tensor( self.values[indices], dtype=torch.float).to(device)
    dones   = torch.tensor( self.dones[indices] ).float()
    return states, actions, rewards, values, dones

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

    #SIMULATION
    _env = TicTacToe(parent.hidden_state)
    next_state, node.reward, _ = _env.step(action_history[-1])

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

env = TicTacToe()
net = ActorCritic(env.observation_space.shape[0], env.action_space.n)
memory = ReplayBuffer(env.observation_space.shape[0], env.action_space.n)

def train():
  if(len(memory) >= 20):
    for i in range(10):
      states, actions, rewards, values, dones = memory.sample(32)
      pi, v = net(states)

      policy_loss = -(actions * torch.log(pi)).sum(dim=1) *(1-dones)
      policy_loss = policy_loss.mean()
      value_loss = torch.sum((values-v.view(-1))**2)/values.size()[0]

      loss = policy_loss + value_loss
      #print(loss)

      net.optimizer.zero_grad()
      loss.backward()
      net.optimizer.step()

if __name__ == '__main__':
    

  # computer can play against itself...and tie!
  scores = []
  for epi in range(100):
    done = False
    state = env.reset()
    score = 0
    while not done:
      policy, node = MCTS(net, state, 10)
      #print(policy)
      action = np.random.choice(np.arange(len(policy)), p=policy)
      #action = policy.argmax()
      train()

      nstate, reward, done = env.step(action)
      memory.store(state,action,reward,node.value(),done)
      #env.render()	

      score += reward
      state = nstate
    scores.append(score)
    print(f'Episode {epi}, return{score}')

