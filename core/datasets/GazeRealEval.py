import re
import pandas as pd
import torch
import torchvision.transforms as transforms
import pickle
from typing import Any, Callable, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import json
from .GazeReal import GazeReal
from .GazeToy import GazeToy
from tqdm import tqdm

__all__ = ["GazeRealEval"]


class GazeRealEval(GazeReal):
    def __init__(self, metadata, vocab, is_train=True):
        # init super
        super().__init__(metadata, vocab, is_train)

    def __getitem__(self, index):
        img, fixation, fix_masks, transcript, sent_masks = super().__getitem__(index)
        return index, img, fixation, fix_masks, transcript, sent_masks


if __name__ == "__main__":
    ds = GazeRealEval(
        metadata="/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json",
        vocab="/home/ptthang/gaze_sample/data_here/vocab.json",
    )
    print(len(ds))
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=1, shuffle=True)
    # enumerate tqdm for the dataset
    mx, mn = 0, 100000
    for i, (idx, img, fixation, fix_masks, transcript, sent_masks) in enumerate(
        tqdm(dl)
    ):
        if i == 0:
            print(
                img.shape,
                fixation.shape,
                fix_masks.shape,
                transcript.shape,
                sent_masks.shape,
            )
        if mx < fixation.shape[1]:
            mx = fixation.shape[1]
        if mn > fixation.shape[1]:
            mn = fixation.shape[1]
        # continue
        # print(img.shape, fixation.shape, transcript)
        # break # only break for one sample, if it is ok, cmt break and run the whole dataset to make sure it runs

    print(mx, mn)
