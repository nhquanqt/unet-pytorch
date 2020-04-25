import os, time, sys
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

from models import Unet
from datasets import Cityscapes

import matplotlib.pyplot as plt

import random

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/home/wan/dataset/cityscapes', help='root directory of cityscapes')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--resize', type=int, default=256, help='resize image')
parser.add_argument('--crop', type=tuple, default=(224,224), help='crop image')
opt = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main():
    model = torch.load('unet.pth').to(device)

    dataset = Cityscapes(opt.root, split='val', resize=opt.resize, crop=opt.crop)

    inputs, labels = random.choice(dataset)

    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    outputs = outputs.detach()

    dst = torch.zeros((outputs[0].size()[1], outputs[0].size()[2]),dtype=torch.float32)

    for i in range(outputs[0].size()[0]):
        dst[outputs[0][i] > 0.5] = i
    
    plt.imshow(dst)
    plt.show()


if __name__ == "__main__":
    main()