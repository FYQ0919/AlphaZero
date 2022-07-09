import gym
import math
import random
import numpy as np

import torch
import torch.nn.functional as F
from network import Model 

device='cpu'

class Node:
  def __init__(self, prior: float):
    self.prior = prior       # prior policy probabilities
    self.hidden_state = None # from dynamics function
    self.reward = 0          # from dynamics function
    self.policy = None       # from prediction function
    self.value_sum = 0       # from prediction function
    self.visit_count = 0
    self.children = {}
    #self.to_play = -1

  def value(self) -> bool:
    if self.visit_count == 0:
        return 0
    return self.value_sum / self.visit_count

  def expanded(self) -> float:
    return len(self.children) > 0


class MuZero:
  def __init__(self, in_dims, out_dims):
    self.model  = Model(in_dims, out_dims)
    #self.memory = ReplayBuffer(in_dims, out_dims, device=device)

    self.discount  = 0.95
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

  def softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  #def store(self, obs,action,value):
  #  self.memory.store(obs,action,value)

  def ucb_score(self, parent: Node, child: Node, min_max_stats=None) -> float:
     """
     Calculate the modified UCB score of this Node. This value will be used when selecting Nodes during MCTS simulations.
     The UCB score balances between exploiting Nodes with known promising values, and exploring Nodes that haven't been 
     searched much throughout the MCTS simulations.
     """
     self.pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
     self.pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

     prior_score = self.pb_c * child.prior
     value_score = child.reward + self.discount * child.value()
     return prior_score + value_score

  def select_child(self, node: Node):
    ## We select child using UCT
    #out = [(self.ucb_score(node,child),action,child)for action, child in node.children.items()]
    #smax = max([x[0] for x in out])     # this max is why it favors 1's over 0's
    #_, action, child = random.choice(list(filter(lambda x: x[0] == smax, out)))

    total_visits_count = max(1 , sum([child.visit_count  for action, child in node.children.items()]) )
    action_index = np.argmax([self.ucb_score(node,child).detach().numpy() for action, child in node.children.items()])
    child  = node.children[action_index]
    action = np.array([1 if i==action_index else 0 for i in range(len(node.children))]) #.reshape(1,-1) 
    return action, child
    
  def mcts(self, obs, num_simulations=10):
    # init root node
    root = Node(0) 
    root.hidden_state = self.model.h(obs)

    ## EXPAND root node
    action, policy, value = self.model.f(root.hidden_state)
    for i in range(policy.shape[0]):
      root.children[i] = Node(prior=policy[i])
      #root.children[i].to_play = -root.to_play

    # run mcts

    # keep track of min and max mean-Q values to normalize them during selection phase
    # this is for environments that have unbounded Q-values, otherwise the prior could 
    # potentially have very little influence over selection, if Q-values are large
    # min_q_value, max_q_value = root.value, root.value 
    for _ in range(num_simulations):
      node = root 
      search_path = [node] # nodes in the tree that we select
      action_history = []  # the actions we took that got

      ## SELECT: traverse down the tree according to the ucb_score 
      while node.expanded():
        action, node = self.select_child(node)
        action_history.append(action)
        search_path.append(node)

      # EXPAND : now we are at a leaf which is not "expanded", run the dynamics model
      parent = search_path[-2]
      action = torch.tensor(action_history[-1])
 
      # use the model to estimate the policy and value, use policy as prior
      # run the dynamics model then use predict a policy and a value
      node.reward, node.hidden_state = self.model.g(parent.hidden_state, action)
      action, policy, value = self.model.f( node.hidden_state )

      # create all the children of the newly expanded node
      for i in range(policy.shape[0]):
        node.children[i] = Node(prior=policy[i])
        #node.children[i].to_play = -node.to_play

      # BACKPROPAGATE: update the state with "backpropagate"
      for bnode in reversed(search_path):
        bnode.visit_count += 1
        bnode.value_sum += value
        discount = 0.95
        value = bnode.reward + discount * value

    # Each node represents a potential action, number of visits to each node - normalized
    # (by a softmax) represent the probabilty of taking that action. This is our policy
    visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
    visit_counts = [x[1] for x in sorted(visit_counts)]
    av = np.array(visit_counts).astype(np.float64)
    policy = self.softmax(av)
    return policy, value, root



env = gym.make('CartPole-v1')
env = gym.wrappers.RecordEpisodeStatistics(env)
agent = MuZero(env.observation_space.shape[0], env.action_space.n)

scores, time_step = [], 0
for epi in range(1000):
  obs = env.reset()
  while True:

    #env.render()
    policy, value, _ = agent.mcts(obs, 1)
    action = np.argmax(policy)
    #action = env.action_space.sample()

    n_obs, reward, done, info = env.step(action)
    #agent.store(obs,policy,value)

    obs = n_obs
    #agent.train()

    if "episode" in info.keys(): 
      scores.append(int(info['episode']['r']))
      avg_score = np.mean(scores[-100:]) # moving average of last 100 episodes
      print(f"Episode {epi}, Return: {scores[-1]}, Avg return: {avg_score}")
      break
