from pathlib import Path 
import monai 
import json
from monai.transforms import LoadImage 
from PIL import Image
import torch
import cv2 
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
# import partial
from functools import partial
import multiprocessing
import sys 

p = Path('/media/ptthang/Seagate16T_Sang2/thang/mimic-cxr-2.0.0.physionet.org/files')
loader = LoadImage(image_only=True)
all_dcm = sorted(list(p.glob('**/*.dcm')))
with open("all_dcm.txt", "w") as f: 
    pass
def _load(dcm):
    img = loader(dcm)
    H,W,C = img.shape
    if C != 1:
        print(img.shape, dcm)
        with open("all_dcm.txt", "a") as f: 
            f.write(str(dcm))
    
pool = multiprocessing.Pool()
for i, _ in enumerate(pool.imap_unordered(_load, all_dcm)):
    sys.stderr.write('\\rdownloading %d / %d' % (i + 1, len(all_dcm)))
sys.stderr.write('\\ndone.\\n')
