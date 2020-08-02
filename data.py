import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import pandas as pd
import numpy as np

import os


def get_data(args):
    train_dataset = ImageFolder(args.root, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.CenterCrop((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.bs, shuffle=True)
    test_loader = []
    return train_loader, test_loader


from options import get_args
from util import get_grid

if __name__ == '__main__':
    args = get_args()
    # print(args.bs)
    # print(args.root)
    train_loader, test_loader = get_data(args=args)
    for idx, (x, y) in enumerate(train_loader):
        # print(x)
        # print(x.shape)
        imgs = get_grid(x.numpy(), args, 1, idx)
        # imgs.show()
