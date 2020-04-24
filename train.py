import os, time, sys
import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import BCELoss

from models import Unet
from datasets import Cityscapes

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/home/wan/dataset/cityscapes', help='root directory of cityscapes')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--resize', type=int, default=256, help='resize image')
parser.add_argument('--crop', type=tuple, default=(224,224), help='crop image')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs')
opt = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def main():
    model = Unet(19).to(device)
    # model = torch.load('unet.pth').to(device)

    dataset = Cityscapes(opt.root, resize=opt.resize, crop=opt.crop)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=1
    )

    criterion = BCELoss().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(opt.n_epochs):
        print('epoch {}'.format(epoch))
        for i, batch in enumerate(dataloader):
            inputs, labels = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(loss)

    torch.save(model, 'unet.pth')

if __name__ == '__main__':
    main()