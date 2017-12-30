# -*- coding: utf-8 -*-
import envs
import neural_network
import gym
import numpy as np
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
from collections import deque


PLAYER = 0
OPPONENT = 1
MARK_O = 2


# 에이전트 클래스
class ZeroAgent(object):
    def __init__(self, state, action_space):
        self.state = state
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.action_space = action_space

    def act(self):
        while True:
            target = self.action_space.sample()
            if self.board[target[1]][target[2]] == 0:
                return [target[1], target[2]]


class MCTS(object):
    def __init__(self, pr=None):
        self.pr = pr
        self.node_memory = deque(maxlen=9 * 1000 * 2)
        self.edge_memory = deque(maxlen=9 * 1000 * 2)
        self.c_puct = 5
        self.tau = 0.67
        self.Pi = None

    def run(self):
        self.make_node()
        self.make_edge()
        self.select()
        self.backup()

    def make_node(self):
        self.node_memory.append(self.state)
        return self.node_memory[0]

    def select_action(self, state):
        self.state = state.copy()
        self.make_edge()
        return 0

    def _make_edge(self):
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        edge = np.zeros((3, 3, 4), 'float')
        empty_loc = np.where(self.board == 0)
        legal_move_n = len(empty_loc[0])
        puct_memory = []
        if self.pr is None:
            self.pr = 1 / legal_move_n
        for i in range(legal_move_n):
            N = edge[empty_loc[0][i]][empty_loc[1][i]][0]
            W = edge[empty_loc[0][i]][empty_loc[1][i]][1]
            if N != 0:
                Q = W / N
                edge[empty_loc[0][i]][empty_loc[1][i]][2] = Q
            else:
                Q = edge[empty_loc[0][i]][empty_loc[1][i]][2]
        P = edge[empty_loc[0][i]][empty_loc[1][i]][3] = self.pr
        self.edge_memory.append(edge)

    def backup(self, reward):
        print(self.node_memory, self.edge_memory)


if __name__ == "__main__":
    env = gym.make('TicTacToe-v0')
    env.seed(2018)
    action_space = env.action_space
    selfplay = MCTS()
    episode_count = 1000
    reward_memory = []
    for e in range(episode_count):
        state = env.reset()
        done = False
        while not done:
            action = selfplay.select_action(state)
            state, reward, done, info = env.step(action)
            reward_memory.append(reward)
        if done:
            selfplay.backup(reward)
    env.close()

'''
    episode_count = 8000
    result = {1: 0, 0: 0, -1: 0}

    for i in range(episode_count):
        state = env.reset()
        print('-' * 14, '\nepisode: %d' % (i + 1))
        k = 0
        while True:
            state = torch.from_numpy(state).float().unsqueeze(0)
            state = Variable(state, requires_grad=True)
            p, v = agent.brain(state)
            loss = p[0]
            print('p-vector:{}'.format(p))
            print('value:{}'.format(v))
            action = [k % 2,
                      env.action_space.sample()[1],
                      env.action_space.sample()[2]]
            state, reward, done, info = env.step(action)
            k += 1
            if done:
                result[reward] += 1
                break
    env.close()
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100))
'''
