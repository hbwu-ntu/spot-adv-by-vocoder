#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, embedding_dim, num_classes, **kwargs):
        super(LossFunction, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.criertion = nn.CrossEntropyLoss()

    def forward(self, x, label=None):
        assert len(x.shape) == 3
        label = label.repeat_interleave(x.shape[1])
        x = x.reshape(-1, self.embedding_dim)
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.embedding_dim

        x = self.fc(x)
        loss = self.criertion(x, label)
        prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
