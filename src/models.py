from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBatchLeaky(nn.Conv2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


class ConvTrBatchLeaky(nn.ConvTranspose2d):
    def __init__(self, lr_slope, *args, **kwargs):
        super(ConvTrBatchLeaky, self).__init__(*args, **kwargs)
        batch_dim = self.weight.data.size(0)
        self.bn = nn.BatchNorm2d(batch_dim)
        self.lr = nn.LeakyReLU(lr_slope)

    def forward(self, x):
        x = super(ConvTrBatchLeaky, self).forward(x)
        return self.lr(self.bn(x))


class EncDecCelebA(nn.Module):

    def __init__(self, in_channels=1, lr_slope=0.2, bias=False):
        super(EncDecCelebA, self).__init__()
        self.lr_slope = lr_slope

        self.conv1 = ConvBatchLeaky(self.lr_slope, in_channels, 256, 4, 2, 1, 1, bias=False)
        self.conv2 = ConvBatchLeaky(self.lr_slope, 256, 512, 4, 2, 1, 1, bias=False)  # 8
        self.conv3 = ConvBatchLeaky(self.lr_slope, 512, 1024, 4, 2, 1, 1, bias=False)  # 4
        self.conv4 = ConvBatchLeaky(self.lr_slope, 1024, 1024, 3, 1, 2, 2, groups=512, bias=False)  # 4
        self.conv5 = ConvBatchLeaky(self.lr_slope, 1024, 1024, 3, 1, 2, 2, groups=512, bias=False)  # 4
        self.conv6 = ConvBatchLeaky(self.lr_slope, 1024, 1024, 3, 1, 2, 2, groups=512, bias=False)  # 4

        self.convT1 = ConvTrBatchLeaky(0.2, 1024 + 1024, 512, 4, 2, 1, bias=bias)  # 8
        self.convT2 = ConvTrBatchLeaky(0.2, 512 + 512, 512, 4, 2, 1, bias=bias)  # 16
        self.convT3 = ConvTrBatchLeaky(0.2, 512, 256, 4, 2, 1, bias=bias)  # 32

        self.convT4 = ConvTrBatchLeaky(0.2, 256, 128, 4, 2, 1, bias=bias)  # 64
        self.convT5 = ConvTrBatchLeaky(0.2, 128, 64, 3, 1, 1, bias=bias)  # 128
        self.convT6 = ConvTrBatchLeaky(0.2, 64, 32, 3, 1, 1, 1, bias=bias)  # 128

        self.convT7 = nn.Conv2d(32, 3, 3, 1, 1, 1, bias=bias)
        self.upsamp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input):
        #Encoder
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        #Decoder
        x = torch.cat([x6, x3], 1)
        x = self.convT1(x)  # 8
        x = torch.cat([x, x2], 1)

        x = self.convT2(x)  # 16
        x = self.convT3(x)  # 32
        x = self.convT4(x)  # 64
        x = self.upsamp(x)

        x = self.convT5(x)  # 128
        x = self.convT6(x)  # 128
        x = F.sigmoid(self.convT7(x))  # 128

        return x
