# -*- coding: utf-8 -*-
import envs
import neural_network

import gym
from gym.utils import seeding

import numpy as np
import hashlib
import h5py

import math
from collections import deque

import torch
import torch.optim as optim
from torch.autograd import Variable


PLAYER = 0
OPPONENT = 1
MARK_O = 2
episode_count = 20


# 에이전트 클래스 (실제 플레이 용, 예시)
class ZeroAgent(object):
    def __init__(self, state, action_space):
        self.state = state
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.action_space = action_space

    def select_action(self):
        while True:
            target = self.action_space.sample()
            if self.board[target[1]][target[2]] == 0:
                return [target[1], target[2]]


# 몬테카를로 트리 탐색 클래스 (train 데이타 생성 및 실제 플레이에 사용될 예정)
class MCTS(object):
    def __init__(self, pr=True):
        self.pr = pr
        self.edge = np.zeros((3, 3, 4), 'float')
        self.state_memory = deque(maxlen=9 * episode_count)
        self.node_memory = deque(maxlen=9 * episode_count)
        self.edge_memory = deque(maxlen=9 * episode_count)
        self.pi_memory = deque(maxlen=9 * episode_count)
        self.action_memory = deque(maxlen=9)
        self.puct_memory = np.zeros((3, 3), 'float')
        self.pi = np.zeros((3, 3), 'float')
        self.legal_move_n = 0
        self.total_visit = 0
        self.first_turn = PLAYER or OPPONENT
        self.action_count = 1

        # 하이퍼파라미터
        self.c_puct = 5
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.pi = np.zeros((3, 3), 'float')
        self.action_memory = deque(maxlen=9)

    def select_action(self, state):
        memory = dict(zip(self.node_memory, self.edge_memory))
        self.action_count += 1
        user_type = (self.first_turn + self.action_count) % 2
        self.state = state.copy()
        self._make_edge()
        self._cal_puct()
        tmp = np.argwhere(self.puct_memory == self.puct_memory.max())
        while True:
            move_target = tmp[self.np_random.choice(
                tmp.shape[0])]
            is_empty = []
            for i in range(self.legal_move_n):
                is_empty.append(np.array_equiv(self.empty_loc[i], move_target))
            if sum(list(map(int, is_empty))) == 1:
                action = np.r_[user_type, move_target]
                self.edge[move_target[0]][move_target[1]][0] += 1
                self.edge_memory.appendleft(self.edge)
                self.action_memory.appendleft(action)
                return action

    def _cal_puct(self):
        for i in range(3):
            for k in range(3):
                self.total_visit += self.edge[i][k][0]
                self.puct_memory[i][k] = self.edge[i][k][2] + \
                    self.c_puct * self.edge[i][k][3] / (1 + self.edge[i][k][0])
        for i in range(3):
            for k in range(3):
                self.puct_memory[i][k] = self.puct_memory[i][k] * \
                    math.sqrt(self.total_visit - self.edge[i][k][0])

    def _make_edge(self):
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.empty_loc = np.where(self.board == 0)
        self.empty_loc = np.asarray(self.empty_loc)
        self.empty_loc = np.transpose(self.empty_loc)
        self.legal_move_n = self.empty_loc.shape[0]
        if self.pr:
            self.pr = 1 / self.legal_move_n
        for i in range(self.legal_move_n):
            self.edge[self.empty_loc[i][0]][self.empty_loc[i][1]][3] = self.pr

    def backup(self, reward, info):
        steps = info['steps']
        for i in range(steps):
            if self.action_memory[i][0] == 0:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][1] += reward
            else:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][1] += -reward
        print(self.edge_memory)

    def cal_pi(self, tau=0):
        for i in range(len(self.edge_memory)):
            for r in range(3):
                for c in range(3):
                    self.pi[r][c] = self.edge_memory[i][r][c][3]
            self.pi_memory.appendleft(self.softmax(self.pi.flatten()))
        return self.pi_memory

    def softmax(self, N):
        e_x = np.exp(N - np.max(N))
        pi = e_x / e_x.sum(axis=0)
        return np.reshape(pi, (3, 3))


if __name__ == "__main__":
    # 환경 생성 및 시드 설정
    env = gym.make('TicTacToe-v0')
    env.seed(2018)
    action_space = env.action_space
    # 셀프 플레이 인스턴스 생성
    selfplay = MCTS()
    selfplay.seed(2018)
    result = {1: 0, 0: 0, -1: 0}
    # train data 생성
    for e in range(episode_count):
        state = env.reset()
        selfplay.reset()
        print('-' * 15, '\nepisode: %d' % (e + 1))
        selfplay.first_turn = selfplay.np_random.choice(2, replace=False)
        done = False
        while not done:
            state_copy = state.copy()
            state_tuple = tuple(state_copy.flatten())
            selfplay.node_memory.appendleft(state_tuple)
            selfplay.state_memory.appendleft(state)
            action = selfplay.select_action(state)
            print('action: {}'.format(action))
            state, reward, done, info = env.step(action)
        if done:
            selfplay.node_memory.appendleft(state)
            selfplay.backup(reward, info)
            result[reward] += 1
        env.close()
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100))

 #           with h5py.File('edge_memory.h5', 'w') as hf:
 #               hf.create_dataset("edge_memory", data=edge_memory)
 #           print('node: {}'.format(node_memory), '\n')
 #           print('edge: {}'.format(edge_memory), '\n')

    # 신경망 학습
    # 신경망 평가
