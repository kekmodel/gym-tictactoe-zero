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
state = reversed(state_memory)
edge_memory = deque(np.load('data/edge_memory_25k.npy'))
edge = reversed(edge_memory)

mcts_train = MCTS()

tree = ZeroTree(state_path='data/state_memory_25k.npy',
                edge_path='data/edge_memory_25k.npy')

pv_net = PolicyValueNet()

for i, s in enumerate(state):
    s_t = s.reshape(9, 3, 3)
    s_n = Variable(torch.from_numpy(s_t).float().unsqueeze(0))
    p, v = pv_net(s_n)
    v = v.data.numpy().round(decimals=8)
    pi = tree.get_pi(s_t)
    # z = tree.tree_memory(tuple(s_t))
    print("{}\n{}\n{}\n{}".format(s_t, v, pi, p))
