import torch
import torchvision.transforms as transforms
import pickle
from typing import Any, Callable, Optional, Tuple
from pathlib import Path 
import numpy as np
from PIL import Image
__all__ = ['GazeToy']


class GazeToy():
    def __init__(self, metadata, is_train=True):
        self.transform = transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.metadata = metadata
        self.is_train = is_train




    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = CIFAR10Dataset(datapath = ['/media/aioz-thang/data3/aioz-thang/jvn/torchism/data/CIFAR10/cifar-10-batches-py/test_batch'], metapath='/media/aioz-thang/data3/aioz-thang/jvn/torchism/data/CIFAR10/cifar-10-batches-py/batches.meta')
    print(len(ds))
    for i, (im, lbl) in enumerate(ds):
        print(im.shape, lbl)
        break
