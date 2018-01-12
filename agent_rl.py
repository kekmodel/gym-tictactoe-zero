# -*- coding: utf-8 -*-
from tictactoe_env import TicTacToeEnv
from gym.utils import seeding
import numpy as np
import h5py
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
episode_count = 800


class ZeroTree(object):
    def __init__(self):
        self._load_data()
        self.node_memory = deque(maxlen=len(self.state_memory))
        self.tree_memory = defaultdict(lambda: 0)
        self._make_tree()

        # hyperparameter
        self.epsilon = 0.25
        self.alpha = 1.5

        self.visit_count = deque(maxlen=9)
        self.pi_val = deque(maxlen=9)
        self.e_x = deque(maxlen=9)
        self.state_data = deque(maxlen=len(self.tree_memory))
        self.pi_data = deque(maxlen=len(self.tree_memory))
        self._cal_pi()

    def _load_data(self):
        hfs = h5py.File('data/state_memory.hdf5', 'r')
        state_memory = hfs.get('state')
        self.state_memory = deque(state_memory)
        hfs.close()
        hfe = h5py.File('data/edge_memory.hdf5', 'r')
        edge_memory = hfe.get('edge')
        self.edge_memory = deque(edge_memory)
        hfe.close()

    def _make_tree(self):
        for v in self.state_memory:
            v_tuple = tuple(v)
            self.node_memory.append(v_tuple)
        tree_tmp = list(zip(self.node_memory, self.edge_memory))
        for v in tree_tmp:
            self.tree_memory[v[0]] += v[1]

    def _cal_pi(self):
        for k, v in self.tree_memory.items():
            self.state_data.append(k)
            for r in range(3):
                for c in range(3):
                    self.visit_count.append(v[r][c][0])
            for i in range(9):
                self.pi_val.append(self.visit_count[i] / sum(self.visit_count))
            self.pi_data.append(np.asarray(self.pi_val).reshape((3, 3)))

    def get_pi(self, state):
        self.state = state.copy()
        board = self.state[PLAYER] + self.state[OPPONENT] * 2
        if tuple(state.flatten()) in self.state_data:
            i = tuple(self.state.flatten())
            j = self.state_data.index(i)
            pi = self.pi_data[j]
            # print("----- board -----")
            # print(board)
            print('"* zero policy *"')
            print(pi.round(decimals=4))
            return pi
        else:
            empty_loc = np.asarray(np.where(board == 0)).transpose()
            legal_move_n = empty_loc.shape[0]
            pi = np.zeros((3, 3, 3), 'float')
            prob = 1 / legal_move_n
            pr = (1 - self.epsilon) * prob + self.epsilon * \
                self.np_random.dirichlet(self.alpha * np.ones(legal_move_n))
            for i in range(legal_move_n):
                pi[empty_loc[i][0]][empty_loc[i][1]][P] = pr[i]
            # print("----- board -----")
            # print(board)
            print('* random policy *')
            print(pi.round(decimals=4))
            return pi


# 에이전트 클래스 (실제 플레이 용)
class ZeroAgent(object):
    def __init__(self):
        # 학습한 모델 불러오기
        self.model = ZeroTree()

        # action space 좌표 공간 구성
        self.action_space = self._action_space()

        # reset_step member
        self.legal_move_n = None
        self.empty_loc = None
        self.first_turn = None

        # reset_episode member
        self.action_count = None
        self.board = None
        self.state = None

        # member 초기화 및 시드 생성
        self._reset_step()
        self.reset_episode()
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _action_space(self):
        action_space = []
        for i in range(3):
            for j in range(3):
                action_space.append([i, j])
        return np.asarray(action_space)

    def _reset_step(self):
        self.legal_move_n = 0
        self.empty_loc = None

    def reset_episode(self):
        self.action_count = -1
        self.board = np.zeros((3, 3), 'float')
        self.state = np.zeros((3, 3, 3), 'float')

    def select_action(self, state, mode='self'):
        if mode == 'self':
            self.action_count += 1
            user_type = (self.first_turn + self.action_count) % 2
            pi = self.model.get_pi(state)
            choice = self.np_random.choice(
                9, 1, p=pi.flatten(), replace=False)
            move_target = self.action_space[choice[0]]
            action = np.r_[user_type, move_target]
            self._reset_step()
            return action
        elif mode == 'human':
            pi = self.model.get_pi(state)
            choice = self.np_random.choice(
                9, 1, p=pi.flatten(), replace=False)
            move_target = self.action_space[choice[0]]
            action = np.r_[OPPONENT, move_target]
            self._reset_step()
            return action


class HumanAgent(object):
    def __init__(self):
        self.first_turn = None
        self.action_space = self._action_space()
        self.action_count = -1
        self.ai_agent = ZeroAgent()

    def reset_episode(self):
        self.first_turn = None
        self.action_count = -1

    def _action_space(self):
        action_space = []
        for i in range(3):
            for j in range(3):
                action_space.append([i, j])
        return np.asarray(action_space)

    def select_action(self, state):
        self.action_count += 1
        if self.first_turn == PLAYER:
            if self.action_count % 2 == 0:
                print("It's your turn!")
                move_target = input("1 ~ 9: ")
                i = int(move_target) - 1
                action = np.r_[PLAYER, self.action_space[i]]
                return action
            else:
                print("AI's turn!")
                action = self.ai_agent.select_action(state, mode='human')
                return action
        else:
            if self.action_count % 2 == 0:
                print("AI's turn!")
                action = self.ai_agent.select_action(state, mode='human')
                return action
            else:
                print("It's your turn!")
                move_target = input("1 ~ 9: ")
                i = int(move_target) - 1
                action = np.r_[PLAYER, self.action_space[i]]
                return action


if __name__ == "__main__":
    # 환경 생성 및 시드 설정
    env = TicTacToeEnv()
    env.seed(0)
    # 에이전트 생성 및 시드 생성
    my_agent = ZeroAgent()
    my_agent.seed(0)
    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    # play game
    for e in range(episode_count):
        state = env.reset()
        print('-' * 15, '\nepisode: %d' % (e + 1))
        # 첫턴을 나와 상대 중 누가 할지 정하기
        my_agent.first_turn = np.random.choice(2, replace=False)
        env.mark_O = my_agent.first_turn
        # user_type = {PLAYER: 'You', OPPONENT: 'AI'}
        # print('First Turn: {}'.format(user_type[my_agent.first_turn]))
        done = False
        while not done:
            print("---- BOARD ----")
            print(state[PLAYER] + state[OPPONENT] * 2)
            # action 선택하기 (셀프 모드)
            action = my_agent.select_action(state, mode='self')
            # action 진행
            state, reward, done, info = env.step(action)
        if done:
            # 승부난 보드 보기: 내 착수:1, 상대 착수:2
            print("- FINAL BOARD -")
            print(state[PLAYER] + state[OPPONENT] * 2)
            my_agent.reset_episode()
            # 결과 dict에 기록
            result[reward] += 1
    # 에피소드 통계
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], result[1] / episode_count * 100))


'''
    def _cal_pi(self):
        for k, v in self.tree_memory.items():
            self.state_data.append(k)
            for r in range(3):
                for c in range(3):
                    self.visit_count.append(v[r][c][0])
            self.pi_data.append(self.softmax(self.visit_count))

    def softmax(self, visit_count):
        for i in range(9):
            self.e_x.append(
                np.exp(visit_count[i] - np.max(visit_count)))
        for j in range(9):
            self.pi_val.append(self.e_x[j] / np.sum(self.e_x, axis=0))
        return np.asarray(self.pi_val).reshape((3, 3))
'''
