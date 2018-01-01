# -*- coding: utf-8 -*-
import envs

import gym
from gym.utils import seeding

import numpy as np
import math
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
episode_count = 2000


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


# 몬테카를로 트리 탐색 클래스 (train 데이타 생성 용)
class MCTS(object):
    def __init__(self, pr=0):
        self.pr = pr
        self.edge = np.zeros((3, 3, 4), 'float')
        self.pi = np.zeros((3, 3), 'float')
        self.puct_memory = np.zeros((3, 3), 'float')
        self.state_memory = deque(maxlen=9 * episode_count)
        self.node_memory = deque(maxlen=9 * episode_count)
        self.edge_memory = deque(maxlen=9 * episode_count)
        self.pi_memory = deque(maxlen=9 * episode_count)
        self.action_memory = deque(maxlen=9)
        self.tree_memory = defaultdict(lambda: 0)
        self.legal_move_n = 0
        self.total_visit = 0
        self.first_turn = PLAYER or OPPONENT
        self.action_count = -1

        # 하이퍼파라미터
        self.c_puct = 5
        self.epsilon = 0.25
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset_step(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.pi = np.zeros((3, 3), 'float')
        self.puct_memory = np.zeros((3, 3), 'float')
        self.total_visit = 0

    def _reset_episode(self):
        self.action_memory = deque(maxlen=9)
        self.action_count = -1

    def select_action(self, state):
        self.action_count += 1
        self.state = state
        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 나중에 왼쪽에 붙여줌
        user_type = (self.first_turn + self.action_count) % 2
        self._make_edge()
        self._remember_tree()
        self._cal_puct()
        tmp = np.argwhere(self.puct_memory == self.puct_memory.max())
        # 착수금지, 동점 처리
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
                self._reset_step()
                return action

    def _remember_tree(self):
        memory = list(zip(self.node_memory, self.edge_memory))
        for v in memory:
            key = v[0]
            val = v[1]
            self.tree_memory[key] += val  # 더하면 P는 신뢰안됨 (보완필요)

    def _cal_puct(self):
        if self.node_memory[0] in self.tree_memory:
            edge = self.tree_memory[self.node_memory[0]]
            for i in range(3):
                for k in range(3):
                    self.total_visit += edge[i][k][0]
                    # P는 여기서
                    self.puct_memory[i][k] = self.c_puct * \
                        self.edge[i][k][3] / (1 + edge[i][k][0])
            for c in range(3):
                for r in range(3):
                    if edge[c][r][0] != 0:
                        edge[c][r][2] = edge[c][r][1] / edge[c][r][0]
                    self.puct_memory[c][r] = self.puct_memory[c][r] * \
                        math.sqrt(self.total_visit - edge[c][r][0]) + \
                        edge[c][r][2]

    def _make_edge(self):
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.empty_loc = np.asarray(np.where(self.board == 0)).transpose()
        self.legal_move_n = self.empty_loc.shape[0]
        if self.pr == 0:
            self.pr = 1 / self.legal_move_n
        for i in range(self.legal_move_n):
            self.edge[self.empty_loc[i][0]][self.empty_loc[i][1]][3] = self.pr

    def backup(self, reward, info):
        steps = info['steps']
        for i in range(steps):
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][1] += reward
            else:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][1] -= reward
        self._reset_episode()

    def cal_pi(self, tau=0):
        for i in range(len(self.edge_memory)):
            for r in range(3):
                for c in range(3):
                    self.pi[r][c] = self.edge_memory[i][r][c][3]
            self.pi_memory.appendleft(self._softmax(self.pi.flatten()))
        return self.pi_memory

    def _softmax(self, N):
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
    # 기록용
    result = {1: 0, 0: 0, -1: 0}
    # train data 생성
    for e in range(episode_count):
        state = env.reset()
        print('-' * 15, '\nepisode: %d' % (e + 1))
        # 첫턴을 나와 상대 중 누가 할지 정하기
        selfplay.first_turn = selfplay.np_random.choice(2, replace=False)
        done = False
        while not done:
            # state 타입 tuple로 변환하기 (dict의 key로 쓰려고)
            state_copy = state.copy()
            state_tuple = tuple(state_copy.flatten())
            selfplay.node_memory.appendleft(state_tuple)
            selfplay.state_memory.appendleft(state)
            # 액션 고르기
            action = selfplay.select_action(state)
            state, reward, done, info = env.step(action)
        if done:
            # 승부난 상황 보기
            print(state[PLAYER] + state[OPPONENT] * 2)
            # 보상 백업
            selfplay.backup(reward, info)
            result[reward] += 1
    # 에피소드 정리
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100))

    # with h5py.File('edge_memory.h5', 'w') as hf:
    # hf.create_dataset("edge_memory", data=edge_memory)
    # print('node: {}'.format(node_memory), '\n')
    # print('edge: {}'.format(edge_memory), '\n')

    # 신경망 학습
    # 신경망 평가
