import torch
import time
import copy
from torch import nn
import torch.nn.functional as F

class EdgeConv(nn.Module):
    '''
    apply 1D conv on pairs of connected nt by the secondary structure
    '''
    def __init__(self, in_f, out_f):
        super(EdgeConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(out_f*2, out_f, 1, 1),
            nn.InstanceNorm1d(out_f),
            nn.ReLU()
        )
    
    def collect_neighbours(self, x, idx):
        idx = idx.clone()
        for i in range(idx.shape[0]):
            idx[i] += i*(idx.shape[1]+1)
        
        sh = x.shape
        x_new = F.pad(x, (0,1)).permute((0,2,1)).reshape((-1, sh[1]))
        neighbours = x_new[idx.flatten()].reshape((sh[0], sh[2], sh[1])).permute((0,2,1))
        return neighbours
    
    def forward(self, x, idx):
        x2 = self.collect_neighbours(x, idx)
        x = torch.cat([x, x2], 1) #concatenate each nt features with the other nt features connected by the secondary structure
        x = self.layer(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_f, out_f, k):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_f, out_f, k, 1, padding = k//2),
            nn.InstanceNorm1d(out_f),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_f, out_f, k):
        super(DoubleConv, self).__init__()
        self.conv = Conv(in_f, out_f, k)
        self.edge_conv = EdgeConv(out_f, out_f)
    
    def forward(self, x, idx):
        x = self.conv(x)
        x = self.edge_conv(x, idx)
        return x

class ConvWithRes(nn.Module):
    def __init__(self, in_f, out_f, k):
        super(ConvWithRes, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_f, out_f, k, 1, padding = k//2),
            nn.InstanceNorm1d(out_f),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_f, out_f, 1, 1),
            nn.InstanceNorm1d(out_f),
        )
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = F.relu(x1 + x2)
        return x
