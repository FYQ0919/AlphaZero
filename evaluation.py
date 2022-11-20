from curling import Curling
import torch
from alphazero import AlphaZero
import numpy as np


if __name__ == '__main__':

    env = Curling(render=True)
    agent = AlphaZero(env.observation_space.shape[0], env.action_space.n)

    model_file = "./log/ray/1000.pkl"

    model_dict = torch.load(model_file)

    agent.load_model(dict=model_dict)

    while True:
        obs = env.reset()
        done = False
        while not done:
          policy, value = agent.predict(obs)
          action = np.argmax(policy.detach().numpy())
          obs, reward, done, info = env.step(action)




