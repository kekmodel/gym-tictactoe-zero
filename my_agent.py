# -*- coding: utf-8 -*-
from tictactoe_env import TicTacToeEnv
from gym.utils import seeding
import numpy as np
import h5py
import math
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
episode_count = 400


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
        self.state_memory = deque(maxlen=9 * episode_count)
        self.node_memory = deque(maxlen=9 * episode_count)
        self.edge_memory = deque(maxlen=9 * episode_count)
        self.pi_memory = deque(maxlen=9 * episode_count)
        self.tree_memory = None
        self.action_memory = None
        self.puct = None
        self.edge = None
        self.pi = None
        self.legal_move_n = None
        self.empty_loc = None
        self.total_visit = None
        self.first_turn = None
        self.action_count = None
        self.board = None
        self.state = None
        self._reset_step()
        self._reset_episode()

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
        self.puct = np.zeros((3, 3), 'float')
        self.total_visit = 0
        self.legal_move_n = 0
        self.empty_loc = None
        self.tree_memory = defaultdict(lambda: 0)

    def _reset_episode(self):
        self.action_memory = deque(maxlen=9)
        self.action_count = -1
        self.board = np.zeros((3, 3), 'float')
        self.state = np.zeros((3, 3, 3), 'float')

    def select_action(self, state):
        self.action_count += 1
        self.state = state
        # 호출될 때마다 첫턴 기준 교대로 행동주체 바꿈, 최종 action에 붙여줌
        user_type = (self.first_turn + self.action_count) % 2
        self.init_edge()
        self._cal_puct()
        puct = self.puct.tolist()
        for i, v in enumerate(puct):
            for k, s in enumerate(v):
                if [i, k] not in self.empty_loc.tolist():
                    puct[i][k] = -99999
        self.puct = np.asarray(puct)
        tmp = np.argwhere(self.puct == self.puct.max()).tolist()
        for i, v in enumerate(tmp):
            if v not in self.empty_loc.tolist():
                del tmp[i]
        tmp = np.asarray(tmp)
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
                self.action_memory.appendleft(action)
                self.edge_memory[0] = self.edge
                self.tree_memory[self.node_memory[0]] = self.edge
                self._reset_step()
                return action

    def init_edge(self, pr=0):
        if pr == 0:
            self.board = self.state[PLAYER] + self.state[OPPONENT]
            self.empty_loc = np.asarray(np.where(self.board == 0)).transpose()
            self.legal_move_n = self.empty_loc.shape[0]
            pr = round(1 / self.legal_move_n, 5)
            for i in range(self.legal_move_n):
                self.edge[self.empty_loc[i][0]
                          ][self.empty_loc[i][1]][P] = pr
        else:
            for i in range(3):
                for k in range(3):
                    self.edge[i][k][P] = pr[i][k]
        self.edge_memory.appendleft(self.edge)

    def _cal_puct(self):
        # 지금까지의 액션을 반영한 트리 구성 하기
        memory = list(zip(self.node_memory, self.edge_memory))
        # N,W 계산
        for v in memory:
            key = v[0]
            value = v[1]
            self.tree_memory[key] += value
        # 트리에서 현재 state의 edge를 찾아 NWQP를 사용하여 PUCT값 계산
        # 9개의 좌표에 맞는 PUCT값 매칭한 PUCT
        # 계산된 NWQP를 최종 트리에 업데이트
        if self.node_memory[0] in self.tree_memory:
            edge = self.tree_memory[self.node_memory[0]]
            for i in range(3):
                for k in range(3):
                    self.total_visit += edge[i][k][N]
            for c in range(3):
                for r in range(3):
                    if edge[c][r][N] != 0:
                        # Q 업데이트
                        edge[c][r][Q] = edge[c][r][W] / edge[c][r][N]
                    # P 업데이트
                    edge[c][r][P] = self.edge[c][r][P]
                    self.puct[c][r] = round(edge[c][r][Q] + self.c_puct *
                                            edge[c][r][P] * math.sqrt(
                        self.total_visit - edge[c][r][N]) /
                        (1 + edge[c][r][N]), 5)
            self.tree_memory[self.node_memory[0]] = edge

    def backup(self, reward, info):
        steps = info['steps']
        for i in range(steps):
            if self.action_memory[i][0] == PLAYER:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][W] += reward
            else:
                self.edge_memory[i][self.action_memory[i][1]
                                    ][self.action_memory[i][2]][W] -= reward
            self.edge_memory[i][self.action_memory[i][1]
                                ][self.action_memory[i][2]][N] += 1
        self._reset_episode()

    def _cal_pi(self, tau=0):
        if tau == 0:
            for i in range(len(self.edge_memory)):
                for r in range(3):
                    for c in range(3):
                        self.pi[r][c] = self.edge_memory[i][r][c][N]
                self.pi_memory.appendleft(self._softmax(self.pi.flatten()))
            return self.pi_memory

    def _softmax(self, visit_count):
        e_x = np.exp(visit_count - np.max(visit_count))
        pi = e_x / e_x.sum(axis=0)
        return np.reshape(pi, (3, 3))


if __name__ == "__main__":
    # 환경 생성 및 시드 설정
    env = TicTacToeEnv()
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
            # print(state[PLAYER] + state[OPPONENT] * 2)
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
            # print(len(selfplay.node_memory))
            if env.mark_O == PLAYER:
                play_mark_O += 1
                if reward == 1:
                    win_mark_O += 1
    # 에피소드 통계 내기
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%% PlayMarkO: %d WinMarkO: %d' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100, play_mark_O, win_mark_O))
    # 데이터 저장
    with h5py.File('state_memory.hdf5', 'w') as hf:
        hf.create_dataset("zero_data_set", data=selfplay.state_memory)
    with h5py.File('edge_memory.hdf5', 'w') as hf:
        hf.create_dataset("zero_data_set", data=selfplay.edge_memory)
    # 신경망 학습
    # 신경망 평가
