#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
try:
    from .ResNetSE import *
except:
    from ResNetSE import *

def Speaker_Encoder(nOut=256, **kwargs):
    # Number of filters
    num_filters = [64, 128, 256, 512]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model

if __name__ == '__main__':
    model = Speaker_Encoder()
    total = sum([param.nelement() for param in model.parameters()])
    print(total/1e6)
    data = torch.randn(10, 64, 100)
    out = model(data)
    print(data.shape)
    print(out.shape)

