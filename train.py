import time

from curling import Curling
import ray
from learner import Learner
import os
import numpy as np
import torch
from network import ActorCritic
from curling import Curling
from alphazero import Node, MinMaxStats, SummaryWriter
import math
import random
from Strorage import SharedStorage
from replaybuffer import ReplayBuffer
from learner import Learner
from actors import Actor


if __name__ == '__main__':

    os.environ["OMP_NUM_THREADS"] = "1"
    ray.init()

    env = Curling()
    buffer_size = 50000
    num_actors = 15
    input_dim = env.observation_space.shape[0]
    out_dim = env.action_space.n

    buffer = ReplayBuffer.remote(input_dim, out_dim, buffer_size)

    storage = SharedStorage.remote(num_actors)
    #
    actors = [Actor.remote(actor_key, storage, buffer) for actor_key in range(num_actors)]
    learner = Learner.remote(input_dim, out_dim, buffer, storage)
    workers = [learner] + actors

    ray.get([worker.launch.remote() for worker in workers])
    ray.shutdown()