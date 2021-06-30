#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .pooling import *
except:
    from pooling import *



class TDNN(nn.Module):
    def __init__(self, nOut, pooling_type="ASP", n_mels=64, **kwargs):
        super(TDNN, self).__init__()
        self.td_layer1 = torch.nn.Conv1d(in_channels=n_mels, out_channels=256, dilation=1, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.td_layer2 = torch.nn.Conv1d(in_channels=256, out_channels=512, dilation=2, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.td_layer3 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=3, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.td_layer4 = torch.nn.Conv1d(in_channels=512, out_channels=512, dilation=1, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.td_layer5 = torch.nn.Conv1d(in_channels=512, out_channels=1500, dilation=1, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm1d(1500)

        if pooling_type == "Temporal_Average_Pooling" or pooling_type == "TAP":
            self.pooling = Temporal_Average_Pooling()
            self.fc1 = nn.Linear(1500, 512)

        elif pooling_type == "Temporal_Statistics_Pooling" or pooling_type == "TSP":
            self.pooling = Temporal_Statistics_Pooling()
            self.fc1 = nn.Linear(1500*2, 512)

        elif pooling_type == "Self_Attentive_Pooling" or pooling_type == "SAP":
            self.pooling = Self_Attentive_Pooling(1500)
            self.fc1 = nn.Linear(1500, 512)

        elif pooling_type == "Attentive_Statistics_Pooling" or pooling_type == "ASP":
            self.pooling = Attentive_Statistics_Pooling(1500)
            self.fc1 = nn.Linear(1500*2, 512)

        else:
            raise ValueError('{} pooling type is not defined'.format(pooling_type))

        self.fc2 = nn.Linear(512, nOut)

    def forward(self, x):
        '''
        x [batch_size, dim, time]
        '''
        x = F.relu(self.td_layer1(x))
        x = self.bn1(x)

        x = F.relu(self.td_layer2(x))
        x = self.bn2(x)

        x = F.relu(self.td_layer3(x))
        x = self.bn3(x)

        x = F.relu(self.td_layer4(x))
        x = self.bn4(x)

        x = F.relu(self.td_layer5(x))
        x = self.bn5(x)

        x = self.pooling(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def Speaker_Encoder(nOut=256, **kwargs):
    model = TDNN(nOut, n_mels=64, **kwargs)
    return model


if __name__ == '__main__':
    model = Speaker_Encoder()
    total = sum([param.nelement() for param in model.parameters()])
    print(total/1e6)
    data = torch.randn(10, 64, 100)
    out = model(data)
    print(data.shape)
    print(out.shape)

