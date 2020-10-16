import torch
import time
import copy
from torch import nn
import torch.nn.functional as F
from conv_layers import DoubleConv, ConvWithRes

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = ConvWithRes(15, 32, 5)
        self.conv2 = ConvWithRes(32, 64, 5)
        self.conv3 = ConvWithRes(64, 128, 5)
        self.conv4 = DoubleConv(128, 256, 5)
        self.conv5 = DoubleConv(256, 512, 5)
        self.conv6 = DoubleConv(512, 512, 5)
        self.head = nn.Linear(512, 5)
        self.head_pretrain = nn.Conv1d(512, 4, 1, 1)

    def forward_pretrain(self, features, idx):
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x, idx)
        x = self.conv5(x, idx)
        x = self.conv6(x, idx)
        x = self.head_pretrain(x)
        return x
        
    def forward(self, features, masks, idx):
        x = self.conv1(features)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x, idx)
        x = self.conv5(x, idx)
        x = self.conv6(x, idx)
        x = x.permute((0,2,1)).reshape(-1,512)
        masks = masks.flatten()
        x = x[masks]
        x = self.head(x)
        return x
