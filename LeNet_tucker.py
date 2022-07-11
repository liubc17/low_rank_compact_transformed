import torch
import torch.nn as nn
import torch.nn.functional as F
from STN import *


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        # print(self.conv1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 10, 5, bias=False)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)  # with dropout
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


class TuckerLeNet(nn.Module):
    def __init__(self, ch_com_rate=0.5, kernel_size=3, compress_size=3, affine=True, group=True):
        super(TuckerLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = TuckerLayer(6, 10, ch_com_rate=ch_com_rate, kernel_size=kernel_size,
                                 compress_size=compress_size, affine=affine, group=group)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(360, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = x.view(-1, 360)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)  # with dropout
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

