import tictactoe_env

import xxhash
from collections import deque

import numpy as np
import dill as pickle

np.set_printoptions(suppress=True)

PLAYER = 0
OPPONENT = 1
N, W, Q, P = 0, 1, 2, 3
PLANE = np.zeros((3, 3), 'int').flatten()

EPISODE = 16000


class ZeroAgent(object):
    def __init__(self, model_path, user_type):
        self.tree_memory = self._load_tree(model_path)
        self.action_space = self._action_space()
        self.action_count = None
        self.user_type = user_type
        self.reset()
        self.tau = 1

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
        user = {0: 'Player', 1: 'Opponent'}
        print("{}'s Turn!".format(user[self.user_type]))
        self.action_count += 1
        state_reshape = state_new.reshape(9, 3, 3)
        board = state_reshape[0] + state_reshape[4]
        empty_loc = np.argwhere(board == 0)
        node = xxhash.xxh64(state_new.tostring()).hexdigest()
        if node in self.tree_memory:
            edge = self.tree_memory[node]
            pi_memory = self._get_pi(edge)
            if self.action_count <= self.tau:
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
        action = np.r_[self.user_type, move_target]
        return tuple(action)

    def _get_pi(self, edge):
        visit_count_memory = []
        for i in range(3):
            for j in range(3):
                visit_count_memory.append(edge[i][j][N])
        pi_memory = visit_count_memory / sum(visit_count_memory)
        return pi_memory


class AgentVsAgent(object):
    def __init__(self):
        self.ai_player = ZeroAgent('data/tree_memory_e500k.pkl', PLAYER)
        self.ai_opponent = ZeroAgent('data/tree_memory_e1000k.pkl', OPPONENT)
        self.current_turn = None
        self.plyer_history = None
        self.opponent_history = None
        self.state_new = None
        self.reset()

    def reset(self):
        self.current_turn = None
        self.player_history = deque([PLANE] * 4, maxlen=4)
        self.opponent_history = deque([PLANE] * 4, maxlen=4)
        self.state_new = None
        self.ai_player.reset()
        self.ai_opponent.reset()

    def _convert_state(self, state):
        if self.current_turn == OPPONENT:
            self.player_history.appendleft(state[PLAYER].flatten())
        else:
            self.opponent_history.appendleft(state[OPPONENT].flatten())
        state_new = np.r_[np.array(self.player_history).flatten(),
                          np.array(self.opponent_history).flatten(),
                          state[2].flatten()]
        return state_new

    def select_action(self, state):
        self.state_new = self._convert_state(state)
        if self.current_turn == PLAYER:
            action = self.ai_player.select_action(self.state_new)
        else:
            action = self.ai_opponent.select_action(self.state_new)
        return action


if __name__ == '__main__':
    env = tictactoe_env.TicTacToeEnv()
    manager = AgentVsAgent()
    result = {1: 0, 0: 0, -1: 0}
    for e in range(EPISODE):
        state = env.reset()
        print('=' * 15, '\nepisode: {}'.format(e + 1))
        env.player_color = (0 + e) % 2  # 0 = 'O'
        done = False
        action_count = -1
        step = 0
        while not done:
            step += 1
            action_count += 1
            manager.current_turn = (env.player_color + action_count) % 2
            print('---- BOARD ----')
            print(state[PLAYER] + state[OPPONENT] * 2)
            action = manager.select_action(state)
            state, reward, done, _ = env.step(action)
        if done:
            print('- FINAL BOARD -')
            print(state[PLAYER] + state[OPPONENT] * 2)
            result[reward] += 1
            manager.reset()
    print('=' * 20, '\nWin: {}  Lose: {}  Draw: {}  Winrate: {:0.1f}%'.format(
        result[1], result[-1], result[0],
        1 / (1 + np.exp(result[-1] / EPISODE) / np.exp(result[1] / EPISODE)) *
        100))
