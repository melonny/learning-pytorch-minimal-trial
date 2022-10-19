import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        # 定义第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        # 定义第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义第一个全连接层
        self.fc1 = nn.Linear(1296, 128)
        # 定义第二个全连接层
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # 连接各个cnn各个模块
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 36 * 6 * 6)
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        # 返回运算后的结果
        return x
