""" This script preprocesses data and generate new metadata file for preprocessed data.

"""


# resnet input image size is 224x224x3, so we need to resize the image 


import copy
import os
import sys
from typing import Tuple, List, Dict 
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import json 
import torch
from functorch import vmap
from multiprocessing import Pool 


def visualize_gaze(xs, ys, image, points=False, box=False,  alpha=0.5):

    if points:
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        # image = (image / image.max())
        count_ = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for x,y in zip(xs, ys):
            try:
                # i need to filter out invalid data, but let's keep try catch for now
                count_[int(y), int(x)] += 1
            except:
                pass
        # count_ = count_ / count_.max()
        count_ = count_[:, :, np.newaxis]
        count_ = np.repeat(count_, 3, axis=2)
        # make count_ CV_8UC3
        count_ = (count_ * 255).astype(np.uint8)
        # count_color = cv2.applyColorMap(count_, cv2.COLORMAP_JET)
        # gaze is very thin, so jet doesnot work well
        image = cv2.addWeighted(image, alpha, count_, 1-alpha, 0)
        
    if box:
        # plot box
        x0,x1 = xs 
        y0,y1 = ys
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

    return image

def visualize_gaze_with_image(xs, ys, image, points=False, box=False,  alpha=0.5):
    return visualize_gaze(xs, ys, image, points, box,  alpha)

def visualize_gaze_with_image_and_save(xs, ys, image, save_path, points=False, box=False,  alpha=0.5):
    image = visualize_gaze_with_image(xs, ys, image, points, box,  alpha)
    cv2.imwrite(save_path, image)


def split_two_part_odd_even(a,b) -> Tuple[int, int]:
    assert a >= b, "a must be greater than b"
    need_to_pad = a - b 
    if need_to_pad % 2 == 0:
        return need_to_pad // 2, need_to_pad // 2
    else:
        return need_to_pad // 2+1, need_to_pad // 2  

def split_padding(w, h) -> Tuple[int, int, int, int]:
    """Return the padding size for the image, top bottom left right

    Args:
        w (_type_): _description_
        h (_type_): _description_

    Returns:
        Tuple[int, int, int, int]: _description_
    """    
    delta_h1, delta_h2 = 0, 0
    delta_w1, delta_w2 = 0, 0
    if w > h:
        delta_h1, delta_h2 = split_two_part_odd_even(w, h)
    elif w < h:
        delta_w1, delta_w2 = split_two_part_odd_even(h, w)
    elif w == h: 
        return 0,0,0,0
    return delta_h1, delta_h2, delta_w1, delta_w2

def padding_image(img, w,h, get_delta=False) -> np.ndarray:
    """Padding the image to make it square

    Args:
        img (_type_): _description_
        w (_type_): _description_
        h (_type_): _description_
        get_delta (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: _description_
    """    
    delta_h1, delta_h2, delta_w1, delta_w2 = split_padding(w, h)
    img = cv2.copyMakeBorder(img, delta_h1, delta_h2, delta_w1, delta_w2, cv2.BORDER_CONSTANT, value=[0,0,0])
    if not get_delta:
        return img
    return img, delta_h1, delta_h2, delta_w1, delta_w2

def resize_follow_img(v:torch.Tensor, img_old_size: int,img_new_size: int) -> torch.Tensor:
    return v * img_new_size / img_old_size

def preprocess(x: np.ndarray, y:np.ndarray, clip_min=0):
    x = x.clip(min=clip_min)
    y = y.clip(min=clip_min)
    mask = (x> 0) & (y > 0)
    x = x[mask]
    y = y[mask] 
    return x, y 

with open('data_here/gazetoy_metadata.json') as f:
    metadata = json.load(f)

dicom_ids = list(metadata.keys())
saved_dir = 'data_here/new_gaze_data'
os.makedirs(saved_dir, exist_ok=True)
img_jpg_dir = f'{saved_dir}/img_jpg'
os.makedirs(img_jpg_dir, exist_ok=True)
vis_dir = f'{saved_dir}/vis'
os.makedirs(vis_dir, exist_ok=True)
fix_dir = f'{saved_dir}/fixation'
os.makedirs(fix_dir, exist_ok=True)
gaze_dir = f'{saved_dir}/gaze'
os.makedirs(gaze_dir, exist_ok=True)
new_metadata = copy.deepcopy(metadata)
def process_all(dicom_id):
    img_path_jpg = metadata[dicom_id]['img_path_jpg']
    img  = cv2.imread(img_path_jpg, cv2.IMREAD_UNCHANGED)
    h,w = img.shape[:2]
    img, dh1, dh2, dw1, dw2 = padding_image(img, w,h, get_delta=True)
    # resize to 224x224 and repeat 3 channel 
    nw, nh = 224, 224
    old_w, old_h = w, h
    w, h = w + dw1 + dw2, h + dh1 + dh2
    # nw, nh = w,h 
    img = cv2.resize(img, (nh,nw))
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    cv2.imwrite(f'{img_jpg_dir}/{dicom_id}.jpg', img)
    new_metadata[dicom_id]['img_path_jpg'] = f'{img_jpg_dir}/{dicom_id}.jpg'
    vmap_resize_follow_img = vmap(resize_follow_img, in_dims=(0, None, None))
    try:
        if len(metadata[dicom_id]['fixation_reflacx']) > 0 and metadata[dicom_id]['fixation_reflacx'][0] != '':
            for i,(f_path, g_path) in enumerate(zip(metadata[dicom_id]['fixation_reflacx'], metadata[dicom_id]['gaze_reflacx'])):
                fixation_reflacx = pd.read_csv(f_path)
                gaze_reflacx = pd.read_csv(g_path)
                new_gaze_reflacx = {'time': [], 'x': [], 'y': []}
                new_fixation_reflacx = {'start_time': [],'end_time':[], 'x': [], 'y': []}
                new_gaze_reflacx['time'] = gaze_reflacx['timestamp_sample'].values.tolist()
                nx, ny = preprocess(gaze_reflacx['x_position'].values, gaze_reflacx['y_position'].values)
                new_gaze_reflacx['x'] = vmap_resize_follow_img(torch.tensor(nx + dw1), w, nw).tolist()   
                new_gaze_reflacx['y'] = vmap_resize_follow_img(torch.tensor(ny + dh1), h, nh).tolist()
                new_fixation_reflacx['start_time'] = fixation_reflacx['timestamp_start_fixation'].values.tolist()
                new_fixation_reflacx['end_time'] = fixation_reflacx['timestamp_end_fixation'].values.tolist()
                nx, ny = preprocess(fixation_reflacx['x_position'].values, fixation_reflacx['y_position'].values)
                new_fixation_reflacx['x'] = vmap_resize_follow_img(torch.tensor(nx + dw1), w, nw).tolist()
                new_fixation_reflacx['y'] = vmap_resize_follow_img(torch.tensor(ny + dh1), h, nh).tolist()

                visualize_gaze_with_image_and_save(new_fixation_reflacx['x'], new_fixation_reflacx['y'], img,save_path=f'{vis_dir}/{dicom_id}-reflacx.jpg', points=True,alpha=0.2)
                with open(f'{gaze_dir}/{dicom_id}-reflacx-gaze.json', 'w') as f:
                    json.dump(new_gaze_reflacx, f)
                with open(f'{fix_dir}/{dicom_id}-reflacx-fixation.json', 'w') as f:
                    json.dump(new_fixation_reflacx, f)
                if i == 0:
                    new_metadata[dicom_id]['fixation_reflacx'] = [f'{fix_dir}/{dicom_id}-reflacx-fixation.json']
                    new_metadata[dicom_id]['gaze_reflacx'] = [f'{gaze_dir}/{dicom_id}-reflacx-gaze.json']
                else:
                    new_metadata[dicom_id]['fixation_reflacx'].append(f'{fix_dir}/{dicom_id}-reflacx-fixation.json')
                    new_metadata[dicom_id]['gaze_reflacx'].append(f'{gaze_dir}/{dicom_id}-reflacx-gaze.json')

        if len(metadata[dicom_id]['fixation_egd']) > 0:
            fixation_egd = pd.read_csv(metadata[dicom_id]['fixation_egd'])
            gaze_egd = pd.read_csv(metadata[dicom_id]['gaze_egd'])
            new_gaze_egd = {'time': [], 'x': [], 'y': []}
            new_fixation_egd = {'start_time': [],'end_time':[], 'x': [], 'y': []}
            new_fixation_egd['start_time'] = fixation_egd['FPOGS'].values.tolist()
            new_fixation_egd['end_time'] = (fixation_egd['FPOGS'].values + fixation_egd['FPOGD'].values).tolist()
            nx, ny = preprocess(fixation_egd['X_ORIGINAL'].values, fixation_egd['Y_ORIGINAL'].values)
            new_fixation_egd['x'] = vmap_resize_follow_img(torch.tensor(fixation_egd['X_ORIGINAL'].values + dw1), w, nw).tolist()
            new_fixation_egd['y'] = vmap_resize_follow_img(torch.tensor(fixation_egd['Y_ORIGINAL'].values + dh1), h, nh).tolist()
            new_gaze_egd['time'] = gaze_egd['Time (in secs)'].values.tolist()
            nx, ny = preprocess(gaze_egd['X_ORIGINAL'].values, gaze_egd['Y_ORIGINAL'].values)
            new_gaze_egd['x'] = vmap_resize_follow_img(torch.tensor(gaze_egd['X_ORIGINAL'].values + dw1), w, nw).tolist()
            new_gaze_egd['y'] = vmap_resize_follow_img(torch.tensor(gaze_egd['Y_ORIGINAL'].values + dh1), h, nh).tolist()
            visualize_gaze_with_image_and_save(new_fixation_egd['x'], new_fixation_egd['y'], img,save_path=f'{vis_dir}/{dicom_id}egd.jpg', points=True,alpha=0.2)
            with open(f'{gaze_dir}/{dicom_id}-egd-gaze.json', 'w') as f:
                json.dump(new_gaze_egd, f)
            with open(f'{fix_dir}/{dicom_id}-egd-fixation.json', 'w') as f:
                json.dump(new_fixation_egd, f)
            new_metadata[dicom_id]['fixation_egd'] = f'{fix_dir}/{dicom_id}-egd-fixation.json'
            new_metadata[dicom_id]['gaze_egd'] = f'{gaze_dir}/{dicom_id}-egd-gaze.json'
    except Exception as e :
        with open('error.txt', 'a') as f:
            f.write(f'{dicom_id} {str(e)} \n ')

p = Pool() 
# for i, _ in enumerate(p.imap_unordered(process_all, dicom_ids)):
#     sys.stderr.write('\\rprocessing %d / %d\n' % (i + 1, len(dicom_ids)))
# sys.stderr.write('\\ndone.\\n')

for dicom_id in tqdm(dicom_ids):
    process_all(dicom_id)
    

with open(f'data_here/new_metadata.json', 'w') as f:
    json.dump(new_metadata, f)
