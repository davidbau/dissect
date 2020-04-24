from __future__ import print_function
# based on https://github.com/jiecaoyu/pytorch_imagenet

import os
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms
import numpy

def load_places_alexnet(weight_file):
    model = AlexNet()
    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)
    return model


class AlexNet(nn.Sequential):

    def __init__(self, num_classes=None,
            include_lrn=True, split_groups=True,
            include_dropout=True):
        w = [3, 96, 256, 384, 384, 256, 4096, 4096, 365]
        if num_classes is not None:
            w[-1] = num_classes
        if split_groups is True:
            groups = [1, 2, 1, 2, 2]
        else:
            groups = [1, 1, 1, 1, 1]
        sequence = OrderedDict()
        for name, module in [
                ('conv1', nn.Conv2d(w[0], w[1], kernel_size=11,
                    stride=4,
                    groups=groups[0], bias=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('lrn1', LRN(local_size=5, alpha=0.0001, beta=0.75)),
                ('conv2', nn.Conv2d(w[1], w[2], kernel_size=5, padding=2,
                    groups=groups[1], bias=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('lrn2', LRN(local_size=5, alpha=0.0001, beta=0.75)),
                ('conv3', nn.Conv2d(w[2], w[3], kernel_size=3, padding=1,
                    groups=groups[2], bias=True)),
                ('relu3', nn.ReLU(inplace=True)),
                ('conv4', nn.Conv2d(w[3], w[4], kernel_size=3, padding=1,
                    groups=groups[3], bias=True)),
                ('relu4', nn.ReLU(inplace=True)),
                ('conv5', nn.Conv2d(w[4], w[5], kernel_size=3, padding=1,
                    groups=groups[4], bias=True)),
                ('relu5', nn.ReLU(inplace=True)),
                ('pool5', nn.MaxPool2d(kernel_size=3, stride=2)),
                ('flatten', Vectorize()),
                ('fc6', nn.Linear(w[5] * 6 * 6, w[6], bias=True)),
                ('relu6', nn.ReLU(inplace=True)),
                ('dropout6', nn.Dropout()),
                ('fc7', nn.Linear(w[6], w[7], bias=True)),
                ('relu7', nn.ReLU(inplace=True)),
                ('dropout7', nn.Dropout()),
                ('fc8', nn.Linear(w[7], w[8])) ]:
            if not include_lrn and name.startswith('lrn'):
                continue
            if not include_dropout and name.startswith('drop'):
                continue
            sequence[name] = module
        super(AlexNet, self).__init__(sequence)

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75,
            ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), int(numpy.prod(x.size()[1:])))
        return x

