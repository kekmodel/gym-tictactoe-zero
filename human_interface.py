# -*- coding: utf-8 -*-
from tictactoe_env import TicTacToeEnv
import numpy as np
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
EPISODE = 5


class ZeroTree(object):
    def __init__(self):
        self._load_data()
        self.node_memory = deque(maxlen=len(self.state_memory))
        self.tree_memory = defaultdict(lambda: 0)
        self._make_tree()

        # hyperparameter
        self.epsilon = 0.25
        self.alpha = 1

        self.state_data = deque(maxlen=len(self.tree_memory))
        self.pi_data = deque(maxlen=len(self.tree_memory))
        self._cal_pi()

    # 로드할 데이터
    def _load_data(self):
        self.state_memory = np.load('data/state_memory_25k.npy')
        self.edge_memory = np.load('data/edge_memory_25k.npy')

    def _make_tree(self):
        for v in self.state_memory:
            v_tuple = tuple(v)
            self.node_memory.append(v_tuple)
        tree_tmp = list(zip(self.node_memory, self.edge_memory))
        for v in tree_tmp:
            self.tree_memory[v[0]] += v[1]

    def _cal_pi(self):
        for k, v in self.tree_memory.items():
            tmp = []
            visit_count = []
            self.state_data.append(k)
            for r in range(3):
                for c in range(3):
                    visit_count.append(v[r][c][0])
            for i in range(9):
                tmp.append(visit_count[i] / sum(visit_count))
            self.pi_data.append(np.asarray(tmp, 'float').reshape((3, 3)))

    def get_pi(self, state):
        new_state = state.copy()
        temp_state = state.reshape(9, 3, 3)
        origin_state = np.r_[temp_state[0].flatten(),
                             temp_state[4].flatten(),
                             temp_state[8].flatten()]
        self.state = origin_state.reshape(3, 3, 3)
        board = self.state[PLAYER] + self.state[OPPONENT]
        if tuple(new_state.flatten()) in self.state_data:
            i = tuple(new_state.flatten())
            j = self.state_data.index(i)
            pi = self.pi_data[j]
            print('"zero policy"')
            return pi
        else:
            empty_loc = np.argwhere(board == 0)
            legal_move_n = empty_loc.shape[0]
            pi = np.zeros((3, 3))
            prob = 1 / legal_move_n
            pr = (1 - self.epsilon) * prob + self.epsilon * \
                np.random.dirichlet(self.alpha * np.ones(legal_move_n))
            for i in range(legal_move_n):
                pi[empty_loc[i][0]][empty_loc[i][1]] = pr[i]
            print('"random policy"')
            return pi


# 에이전트 클래스
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

        # member 초기화
        self._reset_step()
        self.reset_episode()

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
        self.board = None
        self.state = None

    def select_action(self, state, mode='self'):
        if mode == 'self':
            self.action_count += 1
            user_type = (self.first_turn + self.action_count) % 2
            _pi = self.model.get_pi(state)
            choice = np.random.choice(9, 1, p=_pi.flatten())
            move_target = self.action_space[choice[0]]
            action = np.r_[user_type, move_target]
            self._reset_step()
            return action
        elif mode == 'human':
            self.action_count += 1
            _pi = self.model.get_pi(state)
            if self.action_count < 0:
                pi_max = np.argwhere(_pi == _pi.max()).tolist()
                target = pi_max[np.random.choice(len(pi_max))]
                one_hot_pi = np.zeros((3, 3), 'int')
                one_hot_pi[target[0]][target[1]] = 1
                choice = np.random.choice(
                    9, 1, p=one_hot_pi.flatten())
            else:
                choice = np.random.choice(9, 1, p=_pi.flatten())
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
    my_agent = HumanAgent()
    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    # play game
    mode = input("Play mode >> 1.Text 2.Graphic: ")
    if mode == '1':
        for e in range(EPISODE):
            plane = np.zeros((3, 3)).flatten()
            my_history = deque([plane, plane, plane, plane], maxlen=4)
            your_history = deque([plane, plane, plane, plane], maxlen=4)
            state = env.reset()
            print('-' * 15, '\nepisode: %d' % (e + 1))
            # 선공 정하고 교대로 하기
            my_agent.first_turn = (PLAYER + e) % 2
            # 환경에 알려주기
            env.mark_O = my_agent.first_turn
            turn = {PLAYER: 'You', OPPONENT: 'AI'}
            print('First Turn: {}'.format(turn[my_agent.first_turn]))
            action_count = 0
            done = False
            while not done:
                action_count += 1
                user_type = (my_agent.first_turn + action_count) % 2
                print("---- BOARD ----")
                print(state[PLAYER] + state[OPPONENT] * 2)
                if user_type == PLAYER:
                    my_history.appendleft(state[PLAYER].flatten())
                else:
                    your_history.appendleft(state[OPPONENT].flatten())
                new_state = np.r_[np.array(my_history).flatten(),
                                  np.array(your_history).flatten(),
                                  state[MARK_O].flatten()]
                # action 선택하기
                action = my_agent.select_action(new_state)
                # action 진행
                state, reward, done, info = env.step(action)
            if done:
                import time
                # 승부난 보드 보기: 내 착수:1, 상대 착수:2
                print("- FINAL BOARD -")
                print(state[PLAYER] + state[OPPONENT] * 2)
                time.sleep(1)
                # 결과 dict에 기록
                result[reward] += 1
                my_agent.reset_episode()
                my_agent.ai_agent.reset_episode()
    if mode == '2':
        for e in range(EPISODE):
            plane = np.zeros((3, 3)).flatten()
            my_history = deque([plane, plane, plane, plane], maxlen=4)
            your_history = deque([plane, plane, plane, plane], maxlen=4)
            state = env.reset()
            print('-' * 15, '\nepisode: %d' % (e + 1))
            # 선공 정하고 교대로 하기
            my_agent.first_turn = (PLAYER + e) % 2
            # 환경에 알려주기
            env.mark_O = my_agent.first_turn
            turn = {PLAYER: 'You', OPPONENT: 'AI'}
            print('First Turn: {}'.format(turn[my_agent.first_turn]))
            action_count = 0
            done = False
            while not done:
                env.render()
                action_count += 1
                user_type = (my_agent.first_turn + action_count) % 2
                print("---- BOARD ----")
                print(state[PLAYER] + state[OPPONENT] * 2)
                if user_type == PLAYER:
                    my_history.appendleft(state[PLAYER].flatten())
                else:
                    your_history.appendleft(state[OPPONENT].flatten())
                new_state = np.r_[np.array(my_history).flatten(),
                                  np.array(your_history).flatten(),
                                  state[MARK_O].flatten()]
                # action 선택하기
                action = my_agent.select_action(new_state)
                # action 진행
                state, reward, done, info = env.step(action)
            if done:
                import time
                env.render()
                # 승부난 보드 보기: 내 착수:1, 상대 착수:2
                print("- FINAL BOARD -")
                print(state[PLAYER] + state[OPPONENT] * 2)
                time.sleep(1)
                # 결과 dict에 기록
                result[reward] += 1
                my_agent.reset_episode()
                my_agent.ai_agent.reset_episode()
            env.close()
    # 에피소드 통계
    print('-' * 15, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%%' %
          (result[1], result[-1], result[0], result[1] / EPISODE * 100))
