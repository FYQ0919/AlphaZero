import numpy as np
import unittest

from mcts import MCTS
from Game import Connect2Game
from network import ActorCritic


class TreeTests(unittest.TestCase):

  # Test the MCTS from root with equal priors
  def test1(self):
    class MockModel():
      # starting board is: [0, 0, 0, 0]
      def predict(self, state):
        return np.array([0.26, 0.24, 0.24, 0.26]), 0.0001

    env = Connect2Game()
    net = MockModel()
    state = env.reset() 
    canonical_board = [0, 0, 0, 0]
    root = MCTS( net, canonical_board, 1, env,simulations=50)

    #print("starting")
    # the best move is to play at index 1 or 2
    best_outer_move  = max(root.children[0].visits, root.children[0].visits)
    best_center_move = max(root.children[1].visits, root.children[2].visits)
    self.assertGreater(best_center_move, best_outer_move)
    print("test1",best_center_move, best_outer_move)

  # Test the MCTS best move with bad priors
  def test2(self):
    class MockModel():
      # starting board is: [0, 0, 1, -1]
      def predict(self, state):
        return np.array([0.3, 0.7, 0, 0]), 0.0001

    env = Connect2Game()
    net = MockModel()
    state = env.reset() 
    canonical_board = [0, 0, 1, -1]
    root = MCTS( net, canonical_board, 1, env,simulations=25)

    #print("starting")
    # the best move is to play at index 1 
    self.assertGreater(root.children[1].visits, root.children[0].visits)
    print("test2", root.children[1].visits, root.children[0].visits)

  # Test the MCTS best move with qual  priors
  def test3(self):
    class MockModel():
      # starting board is: [0, 0, 1, -1]
      def predict(self, state):
        return np.array([0.51, 0.49, 0, 0]), 0.0001

    env = Connect2Game()
    net = MockModel()
    state = env.reset() 
    canonical_board = [0, 0, -1, 1]
    root = MCTS( net, canonical_board, 1, env,simulations=25)

    #print("starting")
    # the best move is to play at index 1 
    self.assertGreater(root.children[0].visits, root.children[1].visits)
    print("test3", root.children[0].visits, root.children[1].visits)

  # Test the MCTS best move with losing conditon
  def test4(self):
    class MockModel():
      # starting board is: [0, 0, 1, -1]
      def predict(self, state):
        #return np.array([0.51, 0.49, 0, 0]), 0.0001
        return np.array([0, 0.3, 0.3, 0.3]), 0.0001

    env = Connect2Game()
    net = MockModel()
    state = env.reset() 
    canonical_board = [-1, 0, 0, 0]
    root = MCTS( net, canonical_board, 1, env,simulations=25)

    #print("starting")
    # the best move is to play at index 1 
    self.assertGreater(root.children[1].visits, root.children[2].visits)
    self.assertGreater(root.children[1].visits, root.children[3].visits)
    #print("test3", root.children[0].visits, root.children[1].visits)

if __name__ == '__main__':
    unittest.main()
