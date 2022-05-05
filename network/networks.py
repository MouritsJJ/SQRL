"""
The following code is an adapted and modified version of:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""

import torch.nn as nn

from layers.SqRL import *
from layers.polar import *
from layers.Max import *
from constants import *

#Dropout - https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

class Networks(nn.Module):
    def __init__(self, features, num_classes, init_weights):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(640, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = image_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'sqrl':
            layers += [SqRL()]
        elif v == "P":
            layers += [Polar()]
        elif v == "max":
            layers += [Max(v)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3,5), stride=(1,2), padding=0)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "sqrl":  [16, 32, 64, "max"],
    "polar": ["P", 64, 64, "M", 128, 128, "M", "max"],
}


def sqrl(**kwargs):
    return Networks(make_layers(cfgs['sqrl']), **kwargs)

def polar(**kwargs):
    return Networks(make_layers(cfgs['polar']), **kwargs)
