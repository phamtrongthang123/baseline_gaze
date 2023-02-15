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
from tqdm import tqdm


# convert torch tensor index to words
def tensor2words(tensor, vocab):
    # tensor [num_sent, sent_len]
    num_sent, sent_len = tensor.shape
    all_sent = []
    for i in range(num_sent):
        sent = []
        for j in range(sent_len):
            idx = str(tensor[i, j].item())
            if vocab["idx2word"][idx] == "<UNK>":
                continue
            if vocab["idx2word"][idx] == "<EOS>":
                break
            sent.append(vocab["idx2word"][idx])
        all_sent.append(" ".join(sent))
    # return list of sentences
    return all_sent


def save2json(dict_res, path):
    with open(path, "w") as f:
        json.dump(dict_res, f)
