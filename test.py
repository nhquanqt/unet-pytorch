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

def classes_to_rgb(tensor, dataset, thresh=0.5):
    dst = torch.zeros((3, tensor.size()[1], tensor.size()[2]),dtype=torch.uint8)
    for class_ in dataset.classes:
        if not class_.ignore_in_eval:
            idx = class_.train_id
            dst[0][tensor[idx] > thresh] = class_.color[0]
            dst[1][tensor[idx] > thresh] = class_.color[1]
            dst[2][tensor[idx] > thresh] = class_.color[2]

    return dst


def main():
    model = torch.load('unet.pth').to(device)

    # dataset = Cityscapes(opt.root, split='val', resize=opt.resize, crop=opt.crop)
    dataset = Cityscapes(opt.root, resize=opt.resize, crop=opt.crop)

    inputs, labels = random.choice(dataset)

    inputs = inputs.unsqueeze(0)
    labels = labels.unsqueeze(0)

    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)
    outputs = outputs.detach()

    gt = classes_to_rgb(labels[0], dataset)
    seg = classes_to_rgb(outputs[0], dataset)

    fig, ax = plt.subplots(1,3)

    ax[0].imshow(inputs[0].permute(1,2,0))
    ax[1].imshow(gt.permute(1,2,0))
    ax[2].imshow(seg.permute(1,2,0))

    plt.show()


if __name__ == "__main__":
    main()