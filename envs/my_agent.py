import gym
import envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


class MyAgent(object):
    def __init__(self):
        self.policy = np.zeros((1, 9))
        print(self.policy)
        self.value = []

    # def policy(self):
    # def value(self):
    def act(self, state):
        board = np.array(state[1])
        board_empty = 9 - np.nonzero(board)
        self.policy = np.full(1 / board_empty, (self.policy.shape))

    def learn(self):
        ...


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.policy_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=1), state_values


if __name__ == "__main__":

    env = gym.make('TicTacToe-v0')
    env.seed(2017)
    torch.manual_seed(2017)

    episode_count = 1000
    result = {1: 0, 0: 0, -1: 0}

    for i in range(episode_count):
        state = env.reset()
        print('-' * 14, '\nepisode: %d' % (i + 1))
        k = 0
        while True:
            action = [k % 2, env.action_space.sample()[1]]
            print(action)
            state, reward, done, info = env.step(action)
            k += 1
            if done:
                result[reward] += 1
                print(info)
                break
    env.close()
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100))
