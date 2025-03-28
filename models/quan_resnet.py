import sys
import torch
import torch.nn as nn
from functools import partial
from typing import Dict, Any, List

from .quantization import quan_Conv2d, quan_Linear

def quan_conv3x3(in_planes, out_planes, stride=1):
    """3x3 quantized convolution with padding"""
    return quan_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def quan_conv1x1(in_planes, out_planes, stride=1):
    """1x1 quantized convolution"""
    return quan_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = quan_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = quan_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = quan_conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = quan_Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, quan_Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                quan_conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def _resnet(
    arch: str,
    layers: List[int],
    progress: bool = True,
    pretrained: bool = False,
    **kwargs: Any
) -> CifarResNet:
    model = CifarResNet(BasicBlock, layers, **kwargs)
    return model

def cifar10_resnet20(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet20', [3, 3, 3], **kwargs)

def cifar10_resnet32(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet32', [5, 5, 5], **kwargs)

def cifar10_resnet44(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet44', [7, 7, 7], **kwargs)

def cifar10_resnet56(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet56', [9, 9, 9], **kwargs)

def cifar100_resnet20(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet20', [3, 3, 3], num_classes=100, **kwargs)

def cifar100_resnet32(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet32', [5, 5, 5], num_classes=100, **kwargs)

def cifar100_resnet44(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet44', [7, 7, 7], num_classes=100, **kwargs)

def cifar100_resnet56(*args, **kwargs) -> CifarResNet:
    return _resnet('resnet56', [9, 9, 9], num_classes=100, **kwargs)
