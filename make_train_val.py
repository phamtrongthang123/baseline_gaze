import torch
from functorch import vmap
import einops
import json 
from torch.utils.data import DataLoader, random_split
from core.datasets import GazeReal

dataset = GazeReal(metadata= "/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json",
            vocab= "/home/ptthang/gaze_sample/data_here/vocab.json",
            is_train= True )

ratio = 0.7
train_sz = max(1, int(ratio * len(dataset)))
val_sz = len(dataset) - train_sz
train_dataset, val_dataset = random_split(dataset, [train_sz, val_sz])
val_dataset, test_dataset = random_split(val_dataset, [val_sz//2, val_sz - val_sz//2])
with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json", "r") as f:
    metadata = json.load(f)

train_metadata = {dataset.dicom_ids[i]: metadata[dataset.dicom_ids[i]] for i in train_dataset.indices}
val_metadata = {dataset.dicom_ids[i]: metadata[dataset.dicom_ids[i]] for i in val_dataset.indices}
test_metadata = {dataset.dicom_ids[i]: metadata[dataset.dicom_ids[i]] for i in test_dataset.indices}

print(len(train_metadata))
print(len(val_metadata))
print(len(test_metadata))

with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata_train_7.json", "w") as f:
    json.dump(train_metadata, f)

with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata_val_1_5.json", "w") as f:
    json.dump(val_metadata, f)

with open("/home/ptthang/gaze_sample/data_here/reflacx_new_metadata_test_1_5.json", "w") as f:
    json.dump(test_metadata, f)