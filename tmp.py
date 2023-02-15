import torch
from functorch import vmap
import einops

from torch.utils.data import DataLoader, random_split
from core.datasets import GazeReal

a = GazeReal(metadata= "/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json"
            vocab= "/home/ptthang/gaze_sample/data_here/vocab.json"
            is_train= True )

            
