import neural_network_cpu
# import mcts_zero
import evaluate

import numpy as np
from collections import deque

import torch
# import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data


N, W, Q, P = 0, 1, 2, 3

# Hyper Parameters
epochs = 64
batch_size = 32
learning_rate = 0.01


"""data set 생성용
state_memory = deque(np.load('data/state_memory_25k.npy'))
edge_memory = deque(np.load('data/edge_memory_25k.npy'))

tree = evaluate.ZeroTree(state_path='data/state_memory_25k.npy',
                         edge_path='data/edge_memory_25k.npy')

pi_memory = deque(maxlen=len(state_memory))
for state in state_memory:
    pi = tree.get_pi(state)
    pi_memory.append(pi.flatten())
print(pi_memory[100])


reward_memory = deque(maxlen=len(edge_memory))
for edge in edge_memory:
    z = 0
    for r in range(3):
        for c in range(3):
            z += edge[r][c][W]
    reward_memory.append(z)
print(reward_memory[100])

data_set = deque(zip(state_memory, pi_memory, reward_memory))
torch.save(data_set, 'data/zero_data.pkl')
"""
data_set = torch.load('data/zero_data.pkl')
train_data = data.DataLoader(data_set, batch_size=batch_size,
                             shuffle=True, drop_last=True)


pv_net = neural_network_cpu.PolicyValueNet()
optimizer = torch.optim.SGD(pv_net.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=0.0001)

for epoch in range(epochs):
    minibatch = len(data_set)
    for i, (state, pi, reward) in enumerate(train_data):
        state = Variable(state.view(batch_size, 9, 3, 3).float())
        p, v = pv_net(state)
        pi = Variable(pi.view(1, batch_size * 9).float())
        p = p.view(batch_size * 9, 1)
        z = Variable(reward.float())
        optimizer.zero_grad()
        loss = ((z - v).pow(2).sum() -
                torch.matmul(pi, torch.log(p))) / batch_size
        # print(loss)
        optimizer.step()

        if (i + 1) % 64 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, epochs, i + 1,
                     len(data_set) // batch_size, loss.data[0]))

# Save the Model
torch.save(pv_net.state_dict(), 'data/model.pkl')
