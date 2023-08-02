import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class CNN(nn.Module):
    def __init__(self, cfg):
        super(CNN, self).__init__()
        self.h, self.w = cfg['data']['image_size']['h'], cfg['data']['image_size']['w']
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * ((self.h//2)//2) * ((self.w//2)//2), 128)
        self.fc2 = nn.Linear(128, 2) # 3

    def forward(self, x):
        # x = x.permute(0,3,2,1)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,64 * ((self.h//2)//2) * ((self.w//2)//2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
