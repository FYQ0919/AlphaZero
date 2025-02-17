
import numpy as np
device = "cpu"

import torch

import ray


@ray.remote
class SharedStorage(object):

    def __init__(self, num_actors):
        actor_games = {key: 0 for key in range(num_actors)}
        self.stats = {'training_step': 0, 'actor_games': actor_games}
        self.weights = None

    def get_weights(self, games, actor_key):
        self.stats['actor_games'][actor_key] = games
        return self.weights, self.stats['training_step']

    def get_elo_weights(self):
        return self.weights, self.stats['training_step']

    def store_weights(self, weights, step):
        self.stats['training_step'] = step
        self.weights = weights

    def get_stats(self, key=None):
        return self.stats if key is None else self.stats[key]

    def is_ready(self):
        return self.weights is not None


