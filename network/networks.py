"""
The following code is an adapted and modified version of:
https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
"""

import torch.nn as nn

from layers.SqRL import *
from layers.polar import *
from layers.Max import *
from constants import *

# Network model
class Networks(nn.Module):
    def __init__(self, features, num_classes, init_weights):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(linear, 512),
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

# Method that creates a neural network based on a configuration
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
            layers += [Max()]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3,3), stride=(1,1), padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# Neural network configurations
cfgs = {
    "sqrl":  [64, 64, "M", 128, 128, "M", "max"],
    "polar": ["P", 64, 64, "M", 128, 128, "M", "max"],
}

# Methods for loading the correct configuration based on the name of the model
def sqrl(**kwargs):
    return Networks(make_layers(cfgs['sqrl']), **kwargs)

def polar(**kwargs):
    return Networks(make_layers(cfgs['polar']), **kwargs)
