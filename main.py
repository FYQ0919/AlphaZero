import os
import numpy as np
import torch
import torch.optim as optim
from random import shuffle

from mcts import MCTS
from Game import Connect2Game
from network import ActorCritic

device = 'cpu'
batch_size = 64
numEps = 100           # Number of full games (episodes) to run during each iteration
numIters = 500         # Total number of training iterations
numEpochs = 2          # Number of epochs of training per iteration
num_simulations = 100  # Total number of MCTS simulations to run when deciding on a move to play
numItersForTrainExamplesHistory = 20

env = Connect2Game()
net = ActorCritic(env.observation_space, env.action_space)

def Train(memory):
  shuffle(memory)
  p_losses = []
  v_losses = []
  for e in range( numEpochs ):
    idx = 0
    while( idx < int(len(memory) / batch_size) ):
      samples = np.random.randint( len(memory),  size=batch_size)
      states, actions, values = list(zip(*[memory[i] for i in samples]))

      states = torch.tensor(states).float().to(device)
      target_values  = torch.tensor(values).float().to(device)
      target_actions = torch.tensor(actions).float().to(device)

      out_pi, out_v = net(states)

      #print(states.shape, target_actions.shape, target_values.shape)
      #print(out_pi.shape, out_v.shape)


      policy_loss = -(target_actions * torch.log(out_pi)).sum(dim=1)
      policy_loss = policy_loss.mean()

      value_loss = torch.sum((target_values-out_v.view(-1))**2)/target_values.size()[0]

      total_loss = policy_loss + value_loss
      net.optimizer.zero_grad()
      total_loss.backward()
      net.optimizer.step()

      #p_losses.append(policy_loss)
      #v_losses.append(value_loss)

      print("Total Loss ", total_loss.item())

      idx +=1
      pass
  pass


rewards = []
for _ in range( numIters ):
  buffer = []
  for e in range(numEps):
    #train_examples = []
    player = 1
    state = env.reset()
    DONE = False

    while not DONE:
      canonical_board = env.get_canonical_board(state, player)
      root = MCTS( net, canonical_board, 1, env,simulations=100)

      action_probs = [0 for _ in range(env.action_space.n)]
      for k, v in root.children.items():
          action_probs[k] = v.visits
      action_probs = action_probs / np.sum(action_probs)   ## normalize
      buffer.append((canonical_board, action_probs, player))

      action = root.select_action(temperature=0)
      state, current_player = env.get_next_state(state, player, action)
      reward = env.get_reward_for_player(state, player)

      rewards.append(reward)

      if reward is not None:
        ret = []
        for hist_state, hist_current_player, hist_action_probs in buffer:
          # [Board, currentPlayer, actionProbabilities, Reward]
          ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != player))))
          DONE = True
          break

    #print( np.mean(np.array(rewards)) )
    #print(rewards[-1])
    Train(buffer)

