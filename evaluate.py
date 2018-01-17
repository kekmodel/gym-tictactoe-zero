# -*- coding: utf-8 -*-
from tictactoe_env import TicTacToeEnv
import numpy as np
from collections import deque, defaultdict


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
EPISODE = 2400


# MCTS에서 생성한 데이터로 Tree 구성 {state: sum(edge)}인 dict.
class ZeroTree(object):
    def __init__(self, state_path='path', edge_path='path'):
        self.state_path = state_path
        self.edge_path = edge_path
        self._load_data()
        self.node_memory = deque(maxlen=len(self.state_memory))
        self.tree_memory = defaultdict(lambda: 0)
        self._make_tree()

        # hyperparameter
        self.epsilon = 0.25
        self.alpha = 3

        self.state_data = deque(maxlen=len(self.tree_memory))
        self.pi_data = deque(maxlen=len(self.tree_memory))
        self._cal_pi()

    # 로드할 데이터
    def _load_data(self):
        self.state_memory = np.load(self.state_path)
        self.edge_memory = np.load(self.edge_path)

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
        self.state = state.copy()
        board = self.state[PLAYER] + self.state[OPPONENT]
        if tuple(state.flatten()) in self.state_data:
            i = tuple(self.state.flatten())
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


# 플레이어 에이전트
class AgentPlayer(object):
    def __init__(self):
        # 모델 불러오기
        self.model = ZeroTree(state_path='data/state_memory_20000_f1.npy',
                              edge_path='data/edge_memory_20000_f1.npy')

        # action space 좌표 공간 구성
        self.action_space = self._action_space()

        # reset_step member
        self.legal_move_n = None
        self.empty_loc = None

        # reset_episode member
        self.action_count = None
        self.board = None
        self.state = None
        self.first_turn = None

        # member 초기화 및 시드 생성
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
        self.first_turn = None

    def select_action(self, state, mode='self'):
        if mode == 'self':
            self.first_turn = PLAYER
            self.action_count += 1
            user_type = self.first_turn
            _pi = self.model.get_pi(state)
            if self.action_count < 2:
                pi_max = np.argwhere(_pi == _pi.max()).tolist()
                target = pi_max[np.random.choice(len(pi_max))]
                one_hot_pi = np.zeros((3, 3), 'int')
                one_hot_pi[target[0]][target[1]] = 1
                choice = np.random.choice(
                    9, 1, p=one_hot_pi.flatten())
            else:
                choice = np.random.choice(9, 1, p=_pi.flatten())
            move_target = self.action_space[choice[0]]
            action = np.r_[user_type, move_target]
            self._reset_step()
            return action
        elif mode == 'human':
            print("You are not human")


# 상대 에이전트
class AgentOppnent(object):
    def __init__(self):
        # 모델 불러오기
        self.model = ZeroTree(state_path='data/state_memory_20000_f.npy',
                              edge_path='data/edge_memory_20000_f.npy')

        # action space 좌표 공간 구성
        self.action_space = self._action_space()

        # reset_step member
        self.legal_move_n = None
        self.empty_loc = None

        # reset_episode member
        self.action_count = None
        self.board = None
        self.state = None
        self.first_turn = None

        # member 초기화 및 시드 생성
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
        self.first_turn = None
        self.action_count = -1
        self.board = None
        self.state = None

    def select_action(self, state, mode='self'):
        if mode == 'self':
            self.first_turn = OPPONENT
            self.action_count += 1
            user_type = self.first_turn
            _pi = self.model.get_pi(state)
            choice = np.random.choice(9, 1, p=_pi.flatten(), replace=False)
            move_target = self.action_space[choice[0]]
            action = np.r_[user_type, move_target]
            self._reset_step()
            return action
        elif mode == 'human':
            _pi = self.model.get_pi(state)
            print(_pi.round())
            choice = np.random.choice(9, 1, p=_pi.flatten(), replace=False)
            move_target = self.action_space[choice[0]]
            action = np.r_[OPPONENT, move_target]
            self._reset_step()
            return action


# 싸움 붙이는 클래스
class AgentVsAgent(object):
    def __init__(self):
        self.first_turn = None
        self.action_space = self._action_space()
        self.action_count = -1
        self.agent_player = AgentPlayer()
        self.agent_oppnent = AgentOppnent()

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
                print("Agent Player's turn!")
                action = self.agent_player.select_action(state, mode='self')
                return action
            else:
                print("Agent Opponent's turn!")
                action = self.agent_oppnent.select_action(state, mode='self')
                return action
        else:
            if self.action_count % 2 == 0:
                print("Agent Opponent's turn!")
                action = self.agent_oppnent.select_action(state, mode='self')
                return action
            else:
                print("Agent Player's turn!")
                action = self.agent_player.select_action(state, mode='self')
                return action


if __name__ == "__main__":
    state_memory = deque(maxlen=9 * EPISODE)
    edge_memory = deque(maxlen=9 * EPISODE)
    play_mark_O = 0
    win_mark_O = 0
    # 환경 생성 및 시드 설정
    env = TicTacToeEnv()
    selfplay = AgentVsAgent()
    # 통계용
    result = {1: 0, 0: 0, -1: 0}
    # play game
    for e in range(EPISODE):
        state = env.reset()
        action_memory = deque(maxlen=9)
        print('-' * 15, '\nepisode: %d' % (e + 1))
        # 선공 정하고 교대로 하기
        selfplay.first_turn = ((OPPONENT + e) % 2)
        if selfplay.first_turn == PLAYER:
            play_mark_O += 1
        # 환경에 알려주기
        env.mark_O = selfplay.first_turn
        done = False
        while not done:
            print("---- BOARD ----")
            print(state[PLAYER] + state[OPPONENT] * 2)
            node = state.copy()
            state_memory.appendleft(node.flatten())
            edge = np.zeros((3, 3, 4), 'float')
            # action 선택하기
            action = selfplay.select_action(state)
            action_memory.appendleft(action)
            edge[action[1]][action[2]][N] += 1
            edge_memory.appendleft(edge)
            # action 진행
            state, reward, done, info = env.step(action)
        if done:
            if reward == 1:
                if env.mark_O == PLAYER:
                    win_mark_O += 1
            # 승부난 보드 보기: 내 착수:1, 상대 착수:2
            print("- FINAL BOARD -")
            print(state[PLAYER] + state[OPPONENT] * 2)
            # 결과 dict에 기록
            result[reward] += 1
            steps = info['steps']
            for i in range(steps):
                if action_memory[i][0] == PLAYER:
                    edge_memory[i][action_memory[i][1]][action_memory[i][2]
                                                        ][W] += reward
                else:
                    edge_memory[i][action_memory[i][1]][action_memory[i][2]
                                                        ][W] -= reward
            selfplay.reset_episode()
            selfplay.agent_player.reset_episode()
            selfplay.agent_oppnent.reset_episode()
        # data save
        if (e + 1) % 4000 == 0:
            print('data saved')
            np.save('data/self_state_memory.npy', state_memory)
            np.save('data/self_edge_memory.npy', edge_memory)
        # 에피소드 통계
    print('-' * 22, '\nWin: %d Lose: %d Draw: %d Winrate: %0.1f%% \
PlayMarkO: %d WinMarkO: %d' %
          (result[1], result[-1], result[0], result[1] / (e + 1) * 100,
           play_mark_O, win_mark_O))
