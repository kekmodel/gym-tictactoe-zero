# -*- coding: utf-8 -*-
import neural_network

import time

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
EPOCH = 64
BATCH_SIZE = 32
LR = 0.2
L2 = 0.0001
CHANNEL = 128

# data load
dataset = torch.load('data/train_dataset_s400_g200.pkl')
train_dataset = data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

# 신경망 생성 및 최적화 인스턴스 생성
pv_net = neural_network.PolicyValueNet(CHANNEL).cuda()
optimizer = torch.optim.SGD(pv_net.parameters(), lr=LR, momentum=0.9, weight_decay=L2)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, min_lr=2e-4, verbose=1)

# spec
spec = {'epoch': EPOCH, 'batch size': BATCH_SIZE, 'optim': 'SGD', **optimizer.defaults}
print(spec)

# train
step = 0
for epoch in range(EPOCH):
    val_loss = 0
    for i, (s, pi, z) in enumerate(train_dataset):
        s = Variable(s.view(BATCH_SIZE, 9, 3, 3).float(), requires_grad=True).cuda()
        pi = Variable(pi.view(1, BATCH_SIZE * 9).float(), requires_grad=True).cuda()
        z = Variable(z.float(), requires_grad=True).cuda()

        # forward and backward
        optimizer.zero_grad()
        p, v = pv_net(s)
        p = p.view(BATCH_SIZE * 9, 1)
        loss = ((z - v).pow(2).sum() -
                torch.matmul(pi, torch.log(p))) / BATCH_SIZE
        loss.backward()
        optimizer.step()
        step += 1
        val_loss += loss.data[0]
        if (i + 1) % 771 == 0:
            statics = ('Epoch [%d/%d]  Loss: [%.4f]' %
                       (epoch + 1, BATCH_SIZE, val_loss / (i + 1)))
            print(statics, " Step: [%d/%d]" %
                  ((i + 1) * BATCH_SIZE, len(train_dataset) * BATCH_SIZE))

    # Save the Model
    finish = round(float(time.time() - start))
    torch.save(pv_net.state_dict(), 'data/model_t{}.pkl'.format(step))
    print(statics, ' in {}s [MacBook]'.format(finish))

    scheduler.step(val_loss[0], epoch)

    # 메시지 보내기
    slack = slackweb.Slack(
        url="https://hooks.slack.com/services/T8P0E384U/B8PR44F1C/\
4gVy7zhZ9teBUoAFSse8iynn")
    slack.notify(text=statics + ' in {}s [MacBook]'.format(finish))
