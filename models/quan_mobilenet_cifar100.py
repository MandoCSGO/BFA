import torch
import torch.nn as nn
from .quantization import quan_Conv2d, quan_Linear

class DepthSeperableConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            quan_Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            quan_Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = quan_Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MobileNet(nn.Module):

    def __init__(self, width_multiplier=1, class_num=100):
        super().__init__()
        alpha = width_multiplier
        self.stem = nn.Sequential(
            BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
            DepthSeperableConv2d(
                int(32 * alpha),
                int(64 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv1 = nn.Sequential(
            DepthSeperableConv2d(
                int(64 * alpha),
                int(128 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperableConv2d(
                int(128 * alpha),
                int(128 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv2 = nn.Sequential(
            DepthSeperableConv2d(
                int(128 * alpha),
                int(256 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperableConv2d(
                int(256 * alpha),
                int(256 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        # downsample
        self.conv3 = nn.Sequential(
            DepthSeperableConv2d(
                int(256 * alpha),
                int(512 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            *[DepthSeperableConv2d(
                int(512 * alpha),
                int(512 * alpha),
                3,
                padding=1,
                bias=False
            ) for _ in range(5)]
        )

        # downsample
        self.conv4 = nn.Sequential(
            DepthSeperableConv2d(
                int(512 * alpha),
                int(1024 * alpha),
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            DepthSeperableConv2d(
                int(1024 * alpha),
                int(1024 * alpha),
                3,
                padding=1,
                bias=False
            )
        )

        self.fc = quan_Linear(int(1024 * alpha), class_num)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet_quan_cifar100(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)
