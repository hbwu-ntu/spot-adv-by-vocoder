#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .pooling import Self_Attention_Pooling
except:
    from pooling import Self_Attention_Pooling

class Attention_Block(nn.Module):
    def __init__(self, nOut, **kwargs):
        super(Attention_Block, self).__init__()
        self.attention_layer1 = nn.TransformerEncoderLayer(d_model=nOut, nhead=8)
        self.attention_layer2 = nn.TransformerEncoderLayer(d_model=nOut, nhead=8)
        self.attention_layer3 = nn.TransformerEncoderLayer(d_model=nOut, nhead=8)

    def forward(self, x):
        residual = x
        x = self.attention_layer1(x)
        x = self.attention_layer2(x)
        x = self.attention_layer3(x)
        return residual + x
 

class Attention(nn.Module):
    def __init__(self, nOut, **kwargs):
        super(Attention, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=64, out_channels=nOut, dilation=1, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(nOut)
        self.attention_block1 = Attention_Block(nOut=nOut)
        self.attention_block2 = Attention_Block(nOut=nOut)
        self.attention_block3 = Attention_Block(nOut=nOut)
        self.pooling = Self_Attention_Pooling(nOut)
        self.fc1 = nn.Linear(nOut*2, 1024)
        self.fc2 = nn.Linear(1024, nOut)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, time).
        Returns:
            torch.Tensor: Output tensor (#batch, nOut)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.permute(2, 0, 1)
        x = self.attention_block1(x)
        x = self.attention_block2(x)
        x = self.attention_block3(x)
        x = x.permute(1, 2, 0)
        x = self.statistics_pooling(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def statistics_pooling(x):
        """Computes Statistics Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, channels, time).
        Returns:
            torch.Tensor: Output tensor (#batch, channels*2)
        """
        mean = torch.mean(x, axis=2)
        var = torch.var(x, axis=2)
        x = torch.cat((mean, var), axis=1)
        return x



def Speaker_Encoder(nOut=256, **kwargs):
    # Number of filters
    model = Attention(nOut)
    return model

if __name__ == '__main__':
    model = Speaker_Encoder()
    data = torch.randn(10, 64, 100)
    out = model(data)
    print(data.shape)
    print(out.shape)

