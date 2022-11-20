
import numpy as np
import datetime
import pytz
import time
import torch
import ray
import random
import os
from network import ActorCritic
from curling import Curling
from alphazero import Node, MinMaxStats, SummaryWriter
import math


@ray.remote
class Actor(object):

  def __init__(self, actor_key, storage, replay_buffer):

    self.storage = storage
    self.replay_buffer = replay_buffer
    self.device = "cpu"
    self.env = Curling()
    self.max_training_steps = 100000
    self.writer = SummaryWriter(f'./log/ray/{actor_key}/')

    self.network = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n)
    self.network.to(self.device)
    self.network.eval()
    self.min_max_stats = MinMaxStats()
    self.actor_key = actor_key

    self.experiences_collected = 0
    self.training_step = 0
    self.games_played = 0
    self.return_to_log = 0
    self.length_to_log = 0
    self.value_to_log = {'avg': 0, 'max': 0}

    self._env = Curling()

  def softmax(self, x):#
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  def sync_weights(self, force=False):
    weights, training_step = ray.get(self.storage.get_weights.remote(self.games_played, self.actor_key))
    if training_step != self.training_step or force:
      self.network.load_weights(weights)
      self.training_step = training_step

  def ucb_score(self, parent: Node, child: Node, MinMaxStats=None) -> float:
    # a Node's score is based on its value, plus an exploration bonus based on the prior.
    value = -child.value() if child.visit_count > 0 else 0
    value_score = self.min_max_stats.normalize(child.reward + value)
    prior_score = child.prior * math.sqrt(parent.visit_count) /(child.visit_count+1)
    return prior_score + value_score

  def select_child(self, node: Node):
    # We select child using UCT
    out = [(self.ucb_score(node,child),action,child)for action, child in node.children.items()]
    smax = max([x[0] for x in out])     # this max is why it favors 1's over 0's
    _, action, child = random.choice(list(filter(lambda x: abs(x[0] - smax) < 1e-3, out)))
    return action, child

  def mcts(self, env_state, observation, num_simulations=300):
      self.min_max_stats.reset()
      # init root node
      root = Node(0)
      root.to_play = env_state.to_play()

      ## EXPAND root node
      policy, value = self.network(torch.tensor(observation))
      for i in range(policy.shape[0]):
          root.children[i] = Node(prior=policy[i])
          root.children[i].state = env_state.get_state()
          root.children[i].done = False

      # run mcts
      for j in range(num_simulations):

          action_history = []
          node = root
          search_path = [node]  # nodes in the tree that we select
          to_play = node.to_play

          # for _ in range(tree_depth):
          # move code below under a loop to increase tree depth

          ## SELECT: traverse down the tree according to the ucb_score
          while node.expanded() and not node.done:
              action, node = self.select_child(node)
              action_history.append(action)
              search_path.append(node)
              to_play = 1 - to_play

          # now we are at a leaf which is not "expanded", run the dynamics model

          self._env.set_state(node.state, render=False)
          next_state, node.reward, node.done, _ = self._env.step(action_history[-1])
          node.to_play = to_play

          if not node.done:
              ## EXPANED create all the children of the newly expanded node
              policy, value = self.network(torch.tensor(next_state))
              for i in range(policy.shape[0]):
                  node.children[i] = Node(prior=policy[i])
                  node.children[i].state = self._env.get_state()

              # BACKPROPAGATE: update the state with "backpropagate"
              idx = 0
              for bnode in reversed(search_path):

                  bnode.visit_count += 1

                  if bnode.to_play != to_play:
                      bnode.value_sum -= value
                      reward = -bnode.reward
                  else:
                      bnode.value_sum += value
                      reward = bnode.reward

                  if idx < len(search_path) - 1:
                      new_q = bnode.reward - bnode.value()
                      self.min_max_stats.update(new_q)

                  value = reward + value
                  idx += 1

          else:
              policy, value = self.network(torch.tensor(next_state))
              idx = 0
              for bnode in reversed(search_path):

                  bnode.visit_count += 1

                  if bnode.to_play != to_play:
                      bnode.value_sum -= value
                      reward = -bnode.reward
                  else:
                      bnode.value_sum += value
                      reward = bnode.reward

                  if idx < len(search_path) - 1:
                      new_q = bnode.reward - bnode.value()
                      self.min_max_stats.update(new_q)

                  value = reward + value
                  idx += 1

      # Each node represents a potential action, number of visits to each node - normalized
      # (by a softmax) represent the probabilty of taking that action. This is our policy
      visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
      visit_counts = [x[1] for x in sorted(visit_counts)]
      av = np.array(visit_counts).astype(np.float64)
      policy = self.softmax(av)

      return policy, value, root

  def run_selfplay(self):

    while not ray.get(self.storage.is_ready.remote()):
      time.sleep(1)

    self.sync_weights(force=True)

    start = datetime.datetime.now()

    while self.training_step < self.max_training_steps:
      env = Curling()
      obs = env.reset()
      done = False
      while not done:
          policy, value, _ = self.mcts(env, obs, 300)
          action = np.argmax(policy)
          n_obs, reward, done, info = env.step(action)
          self.replay_buffer.store.remote(obs,action,value)
          obs = n_obs
          self.experiences_collected += 1
          now = datetime.datetime.now()
          # print(f'actor {self.actor_key} collect {self.experiences_collected} data using time {(now - start).seconds}s')

      self.games_played += 1

      self.writer.add_scalar("return", reward, self.games_played)

      self.sync_weights(force=True)

  def launch(self):

    with torch.inference_mode():
      self.run_selfplay()

