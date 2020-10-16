import torch
import time
import copy
from torch import nn
import torch.nn.functional as F
from conv_layers import DoubleConv, EdgeConv

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = DoubleConv(15, 256, 5)
        self.lstm = nn.LSTM(256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)
        self.conv2 = EdgeConv(512, 512)
        self.head = nn.Linear(512, 5)
        self.head_pretrain = nn.Conv1d(512, 4, 1, 1)
        
    def init_hidden(self, n):
        return (torch.zeros((2*2,n,256), dtype = torch.float32).cuda(),
                torch.zeros((2*2,n,256), dtype = torch.float32).cuda())
    
    def forward_pretrain(self, features, idx):
        x = self.conv1(features, idx)
        
        x = x.permute((0,2,1))
        sh = x.shape
        h = self.init_hidden(sh[0])
        x, h = self.lstm(x, h)
        x = x.permute((0,2,1))
        
        x = self.conv2(x, idx)
        x = self.head_pretrain(x)
        return x
    
    def forward(self, features, masks, idx):
        x = self.conv1(features, idx)
        
        x = x.permute((0,2,1))
        sh = x.shape
        h = self.init_hidden(sh[0])
        x, h = self.lstm(x, h)
        x = x.permute((0,2,1))

        x = self.conv2(x, idx)
        x = x.permute((0,2,1))
        x = x.reshape(-1,512)
        
        masks = masks.flatten()
        x = x[masks]
        x = self.head(x)
        return x
