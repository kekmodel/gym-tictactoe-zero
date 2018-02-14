import tictactoe_env

import xxhash
from collections import deque

import numpy as np
import dill as pickle

np.set_printoptions(suppress=True)

HUMAN = 0
AI = 1
N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()

EPISODE = 5


class ZeroAgent(object):
    def __init__(self, model_path):
        self.tree_memory = self._load_tree(model_path)
        self.action_space = self._action_space()
        self.action_count = None
        self.reset()

    def reset(self):
        self.action_count = 0

    def _load_tree(self, path):
        with open(path, 'rb') as f:
            tree_memory = pickle.load(f)
            return tree_memory

    def _action_space(self):
        action_space = []
        for i in range(3):
            for j in range(3):
                action_space.append([i, j])
        return np.asarray(action_space)

    def select_action(self, state_new):
        print("AI's Turn!")
        self.action_count += 1
        state_reshape = state_new.reshape(9, 3, 3)
        board = state_reshape[0] + state_reshape[4]
        empty_loc = np.argwhere(board == 0)
        node = xxhash.xxh64(state_new.tostring()).hexdigest()
        if node in self.tree_memory:
            edge = self.tree_memory[node]
            pi_memory = self._get_pi(edge)
            if self.action_count == 1:
                print('"stochastic"')
                choice = np.random.choice(9, p=pi_memory)
            else:
                print('"deterministic"')
                pi_max_idx = [i for i, v in enumerate(
                    pi_memory) if v == max(pi_memory)]
                choice = np.random.choice(pi_max_idx)
            move_target = self.action_space[choice]
        else:
            print('"random"')
            move_target = empty_loc[np.random.choice(len(empty_loc))]
        action = np.r_[AI, move_target]
        return tuple(action)

    def _get_pi(self, edge):
        visit_count_memory = []
        for i in range(3):
            for j in range(3):
                visit_count_memory.append(edge[i][j][N])
        pi_memory = visit_count_memory / sum(visit_count_memory)
        return pi_memory


class HumanAgent(object):
    def __init__(self):
        self.action_space = self._action_space()

    def _action_space(self):
        action_space = []
        for i in range(3):
            for j in range(3):
                action_space.append([i, j])
        return np.array(action_space)

    def select_action(self, state):
        print("It's your turn!")
        move_target = input('1 ~ 9: ')
        i = int(move_target) - 1
        action = np.r_[HUMAN, self.action_space[i]]
        return tuple(action)


class HumanVsAi(object):
    def __init__(self):
        self.human = HumanAgent()
        self.ai = ZeroAgent('data/tree_memory_e500k.pkl')
        self.current_turn = None
        self.human_history = None
        self.ai_history = None
        self.state_new = None
        self.reset()

    def reset(self):
        self.current_turn = None
        self.human_history = deque([PLANE] * 4, maxlen=4)
        self.ai_history = deque([PLANE] * 4, maxlen=4)
        self.state_new = None
        self.ai.reset()

    def _convert_state(self, state):
        if self.current_turn == AI:
            self.human_history.appendleft(state[HUMAN].flatten())
        else:
            self.ai_history.appendleft(state[AI].flatten())
        state_new = np.r_[np.array(self.human_history).flatten(),
                          np.array(self.ai_history).flatten(),
                          state[2].flatten()]
        return state_new

    def select_action(self, state):
        self.state_new = self._convert_state(state)
        if self.current_turn == HUMAN:
            action = self.human.select_action(state)
        else:
            action = self.ai.select_action(self.state_new)
        return action


if __name__ == '__main__':
    env = tictactoe_env.TicTacToeEnv()
    manager = HumanVsAi()
    result = {1: 0, 0: 0, -1: 0}
    mode = input('Play mode >> 1.Text 2.Graphic: ')
    if mode == '1':
        for e in range(EPISODE):
            state = env.reset()
            print('=' * 15, '\nepisode: {}'.format(e + 1))
            env.player_color = (0 + e) % 2  # 0 = 'O'
            done = False
            action_count = -1
            while not done:
                action_count += 1
                manager.current_turn = (env.player_color + action_count) % 2
                print('---- BOARD ----')
                print(state[HUMAN] + state[AI] * 2)
                action = manager.select_action(state)
                state, reward, done, _ = env.step(action)
            if done:
                import time
                print('- FINAL BOARD -')
                print(state[HUMAN] + state[AI] * 2)
                time.sleep(2)
                result[reward] += 1
                manager.reset()
    if mode == '2':
        for e in range(EPISODE):
            state = env.reset()
            print('-' * 20, '\nepisode: {}'.format(e + 1))
            env.player_color = (0 + e) % 2  # 0 = 'O'
            done = False
            action_count = -1
            while not done:
                env.render()
                action_count += 1
                manager.current_turn = (env.player_color + action_count) % 2
                print('---- BOARD ----')
                print(state[HUMAN] + state[AI] * 2)
                action = manager.select_action(state)
                state, reward, done, _ = env.step(action)
            if done:
                import time
                env.render()
                print('- FINAL BOARD -')
                print(state[HUMAN] + state[AI] * 2)
                time.sleep(2)
                result[reward] += 1
                manager.reset()
            env.render(close=True)
    print('=' * 20, '\nWin: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%'.format(
        result[1], result[-1], result[0],
        1 / (1 + np.exp(result[-1] / EPISODE) / np.exp(result[1] / EPISODE)) *
        100))
