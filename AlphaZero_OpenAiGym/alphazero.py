import math
import torch
import random
import numpy as np
import gym
from network import ActorCritic 

device='cpu'

class ReplayBuffer():
  def __init__(self, obs_dim, act_dim, length=50000, device=device):
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
    self.env_state = None  # we store a copy of the OpenAi Gym library

  def value(self) -> bool:
    if self.visit_count == 0:
        return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> float:
    return len(self.children) > 0
 
class AlphaZero:
  def __init__(self, in_dims, out_dims):
    self.model = ActorCritic(in_dims, out_dims).to(device)
    self.memory = ReplayBuffer(in_dims, out_dims, device=device)

  def store(self, obs,action,reward,_obs,done):
    self.memory.store(obs,action,reward,_obs,done)
    
  def ucb_score(self, parent: Node, child: Node, MinMaxStats=None) -> float:
    # a Node's score is based on its value, plus an exploration bonus based on the prior.
    value_score = -child.value() if child.visit_count > 0 else 0
    prior_score = child.prior * math.sqrt(parent.visit_count) /(child.visit_count+1)
    return prior_score + value_score

  def select_child(self, node: Node):
    # We select child using UCT
    out = [(self.ucb_score(node,child),action,child)for action, child in node.children.items()]
    smax = max([x[0] for x in out])     # this max is why it favors 1's over 0's
    _, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))
    return action, child

  def MCTS(self, env_state, observation, num_simulations=10):
    # init root node
    root = Node(0) 
    root.state = env_state # envirnoment state

    ## EXPAND root node
    policy, value = self.model(torch.tensor(observation))
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])
      root.children[i].state = env_state
    # run mcts
    for _ in range(num_simulations):
      action_history = [] 
      node = root 
      search_path = [node] # nodes in the tree that we select

      # for _ in range(tree_depth): 
      # move code below under a loop to increase tree depth

      ## SELECT: traverse down the tree according to the ucb_score 
      while node.expanded():
        action, node = self.select_child(node)
        action_history.append(action)
        search_path.append(node)

      # now we are at a leaf which is not "expanded", run the dynamics model
      parent = search_path[-2]
      _env = CartPole()
      _env.set_state(env_state.get_state())
      next_state, node.reward, _, _ = _env.step(action_history[-1])

      ## EXPANED create all the children of the newly expanded node
      policy, value = self.model(torch.tensor(next_state))
      for i in range(policy.shape[0]):
        node.children[i] = Node(prior=policy[i])
        node.children[i].state = _env

      # BACKPROPAGATE update the state with 
      for bnode in reversed(search_path):
        bnode.value_sum += value
        bnode.visit_count += 1
        discount = 0.95
        value = bnode.reward + discount * value
    return policy.argmax().numpy(), root

  def train(self, batch_size=64):
    pass

from CartPole import CartPole
env = CartPole()
agent = AlphaZero(env.observation_space.shape[0], env.action_space.n)

scores, time_step = [], 0
for epi in range(100):
  obs = env.reset()
  while True:

    action, _ = agent.MCTS(env, obs, 1)
    n_obs, reward, done, info = env.step(action)

    agent.store(n_obs,action,reward,1,done)
    obs = n_obs

    agent.train()

    if "episode" in info.keys():
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
