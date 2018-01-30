from neural_network_cpu import PolicyValueNet
from mcts_zero import MCTS
from evaluate import ZeroTree

import numpy as np
from collections import deque

import torch
from torch.autograd import Variable


PLAYER = 0
OPPONENT = 1
MARK_O = 2
N, W, Q, P = 0, 1, 2, 3
EPISODE = 800
SAVE_CYCLE = 1000


state_memory = deque(np.load('data/state_memory_25k.npy'))
state = list(reversed(state_memory))
edge_memory = deque(np.load('data/edge_memory_25k.npy'))
edge = list(reversed(edge_memory))

mcts_train = MCTS()

tree = ZeroTree(state_path='data/state_memory_25k.npy',
                edge_path='data/edge_memory_25k.npy')

pv_net = PolicyValueNet()

for i, s in enumerate(state):
    s_t = s.reshape(9, 3, 3)
    board = s_t[0] + s_t[4] * 2
    s_n = Variable(torch.from_numpy(s_t).float().unsqueeze(0))
    p, v = pv_net(s_n)
    v = v.data.numpy()[0][0]
    pi = tree.get_pi(s_t)
    tmp = edge[i]
    z = 0
    for r in range(3):
        for c in range(3):
            z += tmp[r][c][W]
    print("{}\n{}\n{}\n{}\n{}".format(board, z, v, pi, p))
