import math
import torch.nn as nn


# 신경망 클래스: forward 자동 호출
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # convolutional layer
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv_bn = nn.BatchNorm2d(4)
        self.conv_relu = nn.ReLU(inplace=True)

        # residual layer
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(4)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(4)
        # forward엔 여기에 skip connection 추가하기
        self.conv2_relu = nn.ReLU(inplace=True)

        # 정책 헤드: 정책함수 인풋 받는 곳
        self.policy_head = nn.Conv2d(4, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc = nn.Linear(9, 9)
        self.policy_softmax = nn.Softmax(dim=1)

        # 가치 헤드: 가치함수 인풋 받는 곳
        self.value_head = nn.Conv2d(4, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_relu1 = nn.ReLU(inplace=True)
        self.value_fc = nn.Linear(9, 9)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_scalar = nn.Linear(9, 1)
        self.value_out = nn.Tanh()

        # weight 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, state):

        x = self.conv(state)
        x = self.conv_bn(x)
        x = self.conv_relu(x)
        residual = x
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x += residual  # skip connection
        x = self.conv2_relu(x)
        print(x)

        p = self.policy_head(x)
        p = self.policy_bn(p)
        p = self.policy_relu(p)
        p = p.view(p.size(1), -1)  # 텐서 펼치기
        p = self.policy_fc(p)
        p = self.policy_softmax(p)

        v = self.value_head(x)
        v = self.value_bn(v)
        v = self.value_relu1(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        v = self.value_relu2(v)
        v = self.value_scalar(v)
        v = self.value_out(v)

        return p, v
