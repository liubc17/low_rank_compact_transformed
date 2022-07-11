'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
from STN import *

__all__ = [
    'TuckerVGG', 'Tuckervgg11', 'Tuckervgg11_bn', 'Tuckervgg13', 'Tuckervgg13_bn', 'Tuckervgg16', 'Tuckervgg16_bn',
    'Tuckervgg19_bn', 'Tuckervgg19'
]


class TuckerVGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_classes=10):
        super(TuckerVGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, ch_com_rate=0.5, kernel_size=3, compress_size=3, affine=True, group=True, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if in_channels == 3:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = TuckerLayer(in_channels, v, ch_com_rate=ch_com_rate, kernel_size=kernel_size,
                                     compress_size=compress_size, stride=1, affine=affine, group=group)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def Tuckervgg11(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 11-layer model (configuration "A")"""
    return TuckerVGG(make_layers(cfg['A'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size),
               num_classes=num_classes)


def Tuckervgg11_bn(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return TuckerVGG(make_layers(cfg['A'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size,
                           batch_norm=True), num_classes=num_classes)


def Tuckervgg13(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 13-layer model (configuration "B")"""
    return TuckerVGG(make_layers(cfg['B'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size),
               num_classes=num_classes)


def Tuckervgg13_bn(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return TuckerVGG(make_layers(cfg['B'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size,
                           batch_norm=True), num_classes=num_classes)


def Tuckervgg16(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 16-layer model (configuration "D")"""
    return TuckerVGG(make_layers(cfg['D'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size),
               num_classes=num_classes)


def Tuckervgg16_bn(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return TuckerVGG(make_layers(cfg['D'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size,
                           batch_norm=True), num_classes=num_classes)


def Tuckervgg19(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 19-layer model (configuration "E")"""
    return TuckerVGG(make_layers(cfg['E'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size),
               num_classes=num_classes)


def Tuckervgg19_bn(ch_com_rate=0.5, kernel_size=3, compress_size=3, num_classes=10):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return TuckerVGG(make_layers(cfg['E'], ch_com_rate=ch_com_rate, kernel_size=kernel_size, compress_size=compress_size,
                           batch_norm=True), num_classes=num_classes)
