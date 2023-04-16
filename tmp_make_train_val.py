import torch
from functorch import vmap
import einops
import json 
from torch.utils.data import DataLoader, random_split
from core.datasets import GazeReal

dataset = GazeReal(metadata= "/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json",
            vocab= "/home/ptthang/gaze_sample/data_here/vocab.json",
            is_train= True )

ratio = 0.5
train_sz = max(1, int(ratio * len(dataset)))
val_sz = len(dataset) - train_sz
train_dataset, val_dataset = random_split(dataset, [train_sz, val_sz])

with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json", "r") as f:
    metadata = json.load(f)

train_metadata = {dataset.dicom_ids[i]: metadata[dataset.dicom_ids[i]] for i in train_dataset.indices}
val_metadata = {dataset.dicom_ids[i]: metadata[dataset.dicom_ids[i]] for i in val_dataset.indices}


print(len(train_metadata))
print(len(val_metadata))

with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata_5.json", "w") as f:
    json.dump(train_metadata, f)

with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata_the_other_5.json", "w") as f:
    json.dump(val_metadata, f)

