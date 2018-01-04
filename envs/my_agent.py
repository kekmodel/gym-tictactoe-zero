# -*- coding: utf-8 -*-
import envs

import gym
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt
import h5py
import math
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
episode_count = 1600


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
# edge는 현재 state에서 착수 가능한 모든 action
# edge 구성: (3*3*4)array: 9개 좌표에 4개의 요소 매칭
# 4개의 요소: (N, W, Q, P) N: edge 방문횟수 W: 보상누적값 Q: 보상평균(W/N) P: edge 선택확률
# edge[좌표행][좌표열][요소번호]로 요소 접근
class MCTS(object):
    def __init__(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.pi = np.zeros((3, 3), 'float')
        self.puct_memory = np.zeros((3, 3), 'float')
        self.state_memory = deque(maxlen=9 * episode_count)
        self.node_memory = deque(maxlen=9 * episode_count)
        self.edge_memory = deque(maxlen=9 * episode_count)
        self.pi_memory = deque(maxlen=9 * episode_count)
        self.action_memory = deque(maxlen=9)
        self.tree_memory = defaultdict(lambda: 0)
        self.pr = None
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

    def _reset_ram(self):
        self.edge = np.zeros((3, 3, 4), 'float')
        self.pi = np.zeros((3, 3), 'float')
        self.puct_memory = np.zeros((3, 3), 'float')
        self.total_visit = 0
        self.pr = 0

    def _reset_rom(self):
        self.action_memory = deque(maxlen=9)
        self.action_count = -1

    def select_action(self, state):
        self.action_count += 1
        self.state = state
        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 최종 action에 붙여줌
        user_type = (self.first_turn + self.action_count) % 2
        self.make_edge()
        self._cal_puct()
        tmp = np.argwhere(self.puct_memory == self.puct_memory.max())
        # 착수금지, 동점 처리
        while True:
            move_target = tmp[self.np_random.choice(
                tmp.shape[0], replace=False)]
            is_empty = []
            for i in range(self.legal_move_n):
                is_empty.append(np.array_equiv(self.empty_loc[i], move_target))
            if sum(list(map(int, is_empty))) == 1:
                # array 두개 붙여서 action 구성
                action = np.r_[user_type, move_target]
                # 방문횟수 +1
                self.edge[move_target[0]][move_target[1]][N] += 1
                self.edge_memory.appendleft(self.edge)
                self.action_memory.appendleft(action)
                self._reset_ram()
                return action

    def _cal_puct(self):
        # 지금까지의 액션을 반영한 트리 구성 하기
        memory = list(zip(self.node_memory, self.edge_memory))
        for v in memory:
            state = v[0]
            edge = v[1]
            self.tree_memory[state] += edge  # N,W 계산
        # 트리에서 현재 state의 edge를 찾아 NWQP를 사용하여 하나의 PUCT값 계산
        # 9개의 좌표에 맞는 PUCT 매칭 
        # 계산된 최종 NWQP를 트리에 저장
        if self.node_memory[0] in self.tree_memory:
            edge = self.tree_memory[self.node_memory[0]]
            for i in range(3):
                for k in range(3):
                    self.total_visit += edge[i][k][N]
                    self.puct_memory[i][k] = self.c_puct * \
                        edge[i][k][P] / (1 + edge[i][k][N])
            for c in range(3):
                for r in range(3):
                    if edge[c][r][0] != 0:
                        edge[c][r][2] = edge[c][r][W] / edge[c][r][N]
                    self.puct_memory[c][r] = self.puct_memory[c][r] * \
                        math.sqrt(self.total_visit - edge[c][r][N]) + \
                        edge[c][r][Q]

    def make_edge(self, pr=0):
        self.pr = pr
        self.board = self.state[PLAYER] + self.state[OPPONENT]
        self.empty_loc = np.asarray(np.where(self.board == 0)).transpose()
        self.legal_move_n = self.empty_loc.shape[0]
        if self.pr == 0:
            self.pr = 1 / self.legal_move_n
        for i in range(self.legal_move_n):
            self.edge[self.empty_loc[i][0]][self.empty_loc[i][1]][P] = self.pr

    def backup(self, reward, info):
        steps = info['steps']
        for i in range(steps):
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][1] += reward
            else:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][1] -= reward
        self._reset_rom()

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
    # 셀프 플레이 인스턴스 생성
    selfplay = MCTS()
    selfplay.seed(2018)
    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    play_mark_O = 0
    win_mark_O = 0
    # train data 생성
    for e in range(episode_count):
        state = env.reset()
        print('-' * 15, '\nepisode: %d' % (e + 1))
        # 첫턴을 나와 상대 중 누가 할지 정하기
        selfplay.first_turn = selfplay.np_random.choice(2, replace=False)
        done = False
        while not done:
            # state를 hash로 변환 (dict의 key로 쓰려고)
            state_copy = state.copy()
            state_hash = hash(state_copy.tostring())
            # hash 저장
            selfplay.node_memory.appendleft(state_hash)
            # raw state 저장
            selfplay.state_memory.appendleft(state)
            # action 선택하기
            action = selfplay.select_action(state)
            # action 진행
            state, reward, done, info = env.step(action)
        if done:
            # 승부난 후 보드 보기: 플레이어 착수:1 상대착수:2
            print(state[PLAYER] + state[OPPONENT] * 2)
            # 보상을 edge에 백업
            selfplay.backup(reward, info)
            # 결과 dict에 기록
            result[reward] += 1
            if env.mark_O == PLAYER:
                play_mark_O += 1
                if reward == 1:
                    win_mark_O += 1
    # 에피소드 통계 내기
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%% PlayMarkO: %d WinMarkO: %d' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100, play_mark_O, win_mark_O))
    '''# 데이터 저장
    with h5py.File('state_memory.hdf5', 'w') as hf:
        hf.create_dataset("zero_data_set", data=selfplay.state_memory)
    with h5py.File('edge_memory.hdf5', 'w') as hf:
        hf.create_dataset("zero_data_set", data=selfplay.edge_memory)
    # 신경망 학습
    # 신경망 평가
    '''
