import monai 
import json
from monai.transforms import LoadImage 
from PIL import Image
import torch
import cv2 
import numpy as np
# read jpg with open cv 
img = cv2.imread('53ba5a31-2d6bda0a-abb7ad77-065a0f20-b4fa6b44.jpg', cv2.IMREAD_UNCHANGED)
print(img.max(), img.min(), img.shape, img.dtype)

p = "./5c61db80-672e0874-ba236081-53d2f8ad-ab8d2efb.dcm"
loader = LoadImage(image_only=False)
img, metadata = loader(p)
print(img.shape, metadata.keys())
print(type(img), type(metadata))
# torch tensor to pil image and save 
img = img.numpy().astype(np.uint16)
L = 3200
WW = 1000
low = L - WW/2
high = L + WW/2
img[img<=low] = 0
img[img>=high] = 0
img = img/img.max()*255
print(img.max(), img.min(), img.shape, img.dtype)

cv2.imwrite('5c61db80-672e0874-ba236081-53d2f8ad-ab8d2efb.png', cv2.transpose(img))
# with open('metadata-5c61db80-672e0874-ba236081-53d2f8ad-ab8d2efb.txt', 'w') as f:
#     json.dump(metadata, f, indent=4)

print(metadata)