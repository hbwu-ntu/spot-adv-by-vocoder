#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from . import amsoftmax
from . import angleproto

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.amsoftmax = amsoftmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):
        assert x.size()[1] == 2
        nlossS, prec1 = self.amsoftmax(x, label)
        nlossP, _ = self.angleproto(x, None)
        return nlossS+nlossP, prec1
