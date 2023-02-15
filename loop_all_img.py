import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
from tqdm import tqdm

net = models.resnet18(pretrained=True)
with open("data_here/new_metadata.json") as f:
    data = json.load(f)

tfm = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
dicom_ids = list(data.keys())
for dicom_id in tqdm(dicom_ids):
    img_path = data[dicom_id]["img_path_jpg"]
    img = Image.open(img_path)
    img = tfm(img)
    img = img.unsqueeze(0)
    output = net(img)
