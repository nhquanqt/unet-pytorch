import os, sys

import time

import torch
import torch.nn as nn

import torchvision.models as models

import numpy as np
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs = 3):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        blocks = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_convs):
            blocks += [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ]
        
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.blocks(x)


class Unet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(Unet, self).__init__()

        self.encoder = models.vgg16()

        self.decoder_3 = Up(1024, 512)
        self.decoder_2 = Up(512, 256)
        self.decoder_1 = Up(256, 128)
        self.decoder_0 = Up(128, 64)
        
        self.projector = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x0 = x
        for i in range(4):
            x0 = self.encoder.features[i](x0)

        x1 = x0
        for i in range(4, 9):
            x1 = self.encoder.features[i](x1)

        x2 = x1
        for i in range(9, 16):
            x2 = self.encoder.features[i](x2)

        x3 = x2
        for i in range(16, 23):
            x3 = self.encoder.features[i](x3)

        x4 = x3
        for i in range(23, 30):
            x4 = self.encoder.features[i](x4)

        x = torch.cat([x4, x4], dim=1)
        x = self.decoder_3(x, x3)
        x = self.decoder_2(x, x2)
        x = self.decoder_1(x, x1)
        x = self.decoder_0(x, x0)

        return self.projector(x)

def main():
    model = Unet(1)

    t_now = time.time()

    for _ in range(10):
        input = torch.randn(1,3,224,224)
        output = model(input)
        print(output.size())
        print(1./(time.time() - t_now))
        t_now = time.time()


if __name__ == "__main__":
    main()