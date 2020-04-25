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

        self.encoder = models.vgg16(pretrained=True)

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


# Ref: https://github.com/usuyama/pytorch-unet
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

def main():
    # model = Unet(1)
    model = ResNetUNet(19)

    t_now = time.time()

    for _ in range(10):
        input = torch.randn(1,3,224,224)
        output = model(input)
        print(output.size())
        print(1./(time.time() - t_now))
        t_now = time.time()


if __name__ == "__main__":
    main()