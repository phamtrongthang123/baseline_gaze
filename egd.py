from typing import List
import pandas as pd 
import numpy as np 
from pathlib import Path 
import os
class EGD():
    def __init__(self, path='physionet.org/files/egd-cxr/1.0.0'):
        self.path = Path(path)
    def get_all_image_ids(self):
        master_csv = pd.read_csv(self.path/"master_sheet.csv")
        res = []
        for row in master_csv.iterrows():
            image_id = row[1]['path'].split('/')[-1].replace('.dcm', '')
            res.append(image_id)
        return res

    def get_row_from_image_ids(self, image_ids: List[str]):
        master_csv = pd.read_csv(self.path/"master_sheet.csv")
        res = []
        for i,row in master_csv.iterrows():
            image_id = row['path'].split('/')[-1].replace('.dcm', '')
            if image_id in image_ids:
                res.append(row)
        return res
    def pull_labels_from_dicom_id(self, dicom_id,saved_dir):
        """
        dicom_id: dicom id of the image
        saved_dir: directory where the image will be saved
        """
        saved_dir = os.path.join(saved_dir,"egd")
        audio_segmentation_transcripts = os.path.join(saved_dir,"audio_segmentation_transcripts")
        inclusion_exclusion_criteria_outputs = os.path.join(saved_dir,"inclusion_exclusion_criteria_outputs")
        os.makedirs(audio_segmentation_transcripts, exist_ok=True)
        os.makedirs(inclusion_exclusion_criteria_outputs, exist_ok=True)
        os.system('cp {} -rf {}'.format(os.path.join(self.path,'audio_segmentation_transcripts', dicom_id), audio_segmentation_transcripts))

        # for inclusion_exclusion_criteria_outputs 
        chf = pd.read_csv(os.path.join(self.path,'inclusion_exclusion_criteria_outputs', 'CHF.csv'))
        normals = pd.read_csv(os.path.join(self.path,'inclusion_exclusion_criteria_outputs', 'normals.csv'))
        pneumonia = pd.read_csv(os.path.join(self.path,'inclusion_exclusion_criteria_outputs', 'pneumonia.csv'))
        alls = [chf, normals, pneumonia]
        names = ['CHF', 'normals', 'pneumonia']
        for i,p in enumerate(alls):
            rp = p[p['dicom_id'].isin([dicom_id])]
            if rp.empty:
                continue
            else:
                rp.to_csv(os.path.join(inclusion_exclusion_criteria_outputs, f'{names[i]}-{dicom_id}.csv'))
                
        # for bounding boxes
        bounding_boxes = pd.read_csv(os.path.join(self.path, 'bounding_boxes.csv'))
        rp = bounding_boxes[bounding_boxes['dicom_id'].isin([dicom_id])]
        if not rp.empty:
            rp.to_csv(os.path.join(saved_dir, f'bounding_boxes-{dicom_id}.csv'))
        # for eye gaze 
        eye_gaze = pd.read_csv(os.path.join(self.path, 'eye_gaze.csv'))
        rp = eye_gaze[eye_gaze['DICOM_ID'].isin([dicom_id])]
        if not rp.empty:
            rp.to_csv(os.path.join(saved_dir, f'eye_gaze-{dicom_id}.csv'))
        # for fixation 
        fixation = pd.read_csv(os.path.join(self.path, 'fixations.csv'))
        rp = fixation[fixation['DICOM_ID'].isin([dicom_id])]
        if not rp.empty:
            rp.to_csv(os.path.join(saved_dir, f'fixations-{dicom_id}.csv'))
        
if __name__=="__main__":
    # check if egd uses the same images as mimic-cxr-jpg
    egd = EGD("physionet.org/files/egd-cxr/1.0.0")
    master_csv = pd.read_csv(egd.path/"master_sheet.csv")
    print(master_csv.shape)
    for row in master_csv.iterrows():
        image_path = row[1]['path'].replace('files/', './gcloud/mimic-cxr-jpg-2.0.0.physionet.org/files/').replace('.dcm', '.jpg')
        if not Path(image_path).exists():
            print(image_path)
            break