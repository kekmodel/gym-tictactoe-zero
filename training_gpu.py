import neural_network
import evaluate

import time
from collections import deque

import numpy as np
import slackweb

import torch
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils import data

start = time.time()
np.set_printoptions(suppress=True)
N, W, Q, P = 0, 1, 2, 3

# Hyper Parameters
epochs = 128
batch_size = 32
learning_rate = 0.2
l2_value = 0.0001
num_channel = 128
num_layer = 5


def make_data_set(state_path, edge_path):
    """data set 생성 함수"""
    state_memory = deque(np.load(state_path))
    edge_memory = deque(np.load(edge_path))

    tree = evaluate.ZeroTree(state_path=state_path, edge_path=edge_path)

    # pi 저장
    pi_memory = deque(maxlen=len(state_memory))
    for state in state_memory:
        pi = tree.get_pi(state)
        pi_memory.append(pi.flatten())

    # reward 저장
    reward_memory = deque(maxlen=len(edge_memory))
    for edge in edge_memory:
        z = 0
        for r in range(3):
            for c in range(3):
                z += edge[r][c][W]
        reward_memory.append(z)

    data_set = deque(zip(state_memory, pi_memory, reward_memory))
    torch.save(data_set, 'data/zero_data_30k.pkl')


# data load
data_set = torch.load('data/zero_data_30k.pkl')
train_data = data.DataLoader(data_set, batch_size=batch_size,
                             shuffle=False, drop_last=True, num_workers=4)

# 신경망 생성 및 최적화 인스턴스 생성
pv_net = neural_network.PolicyValueNet(num_channel).cuda()
optimizer = torch.optim.SGD(
    pv_net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=l2_value)
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=1, min_lr=2e-4, verbose=1)

# spec
spec = {'epochs': epochs, 'batch size': batch_size,
        'optim': 'SGD', **optimizer.defaults}
print(spec)

# train
for epoch in range(epochs):
    val_loss = 0
    for i, (state, pi, reward) in enumerate(train_data):
        state = Variable(state.view(batch_size, 9, 3, 3).float(),
                         requires_grad=True).cuda()
        pi = Variable(pi.view(1, batch_size * 9).float(),
                      requires_grad=True).cuda()
        z = Variable(reward.float(), requires_grad=True).cuda()

        # forward and backward
        optimizer.zero_grad()
        p, v = pv_net(state)
        p = p.view(batch_size * 9, 1)
        loss = ((z - v).pow(2).sum() -
                torch.matmul(pi, torch.log(p))) / batch_size
        loss.backward()
        optimizer.step()
        val_loss += loss.data[0]
        if (i + 1) % 771 == 0:
            statics = ('Epoch [%d/%d]  Loss: [%.4f]' %
                       (epoch + 1, epochs, val_loss / (i + 1)))
            print(statics, " Step: [%d/%d]" %
                  ((i + 1) * batch_size, len(train_data) * batch_size))

    # Save the Model
    finish = round(float(time.time() - start))
    torch.save(pv_net.state_dict(
    ), 'data/model_{}_res{}_ch{}_test.pkl'.format(
        spec['optim'], num_layer, num_channel))
    print(statics, ' in {}s [MacBook]'.format(finish))

    scheduler.step(val_loss[0], epoch)

    # 메시지 보내기
    slack = slackweb.Slack(
        url="https://hooks.slack.com/services/T8P0E384U/B8PR44F1C/\
4gVy7zhZ9teBUoAFSse8iynn")
    slack.notify(text=statics + ' in {}s [MacBook]'.format(finish))
