import os
from typing import List
import pandas as pd 
import numpy as np 
from pathlib import Path 
from MIMIC_CXR_JPG import MIMIC_CXR_JPG
from collections import defaultdict
class REFLACX():
    def __init__(self, path="./REFLACX"):
        self.path = Path(path)
        self.gaze_dir = self.path/"gaze_data"
        self.main_dir = self.path/"main_data"

    def get_gaze_data(self, phase):
        return pd.read_csv(self.main_dir/"metadata_phase_{}.csv".format(phase))

    def get_all_gaze_data(self):
        return [self.get_gaze_data(i) for i in range(1, 4)]

    def get_all_image_ids(self):
        metadata_phases = self.get_all_gaze_data()
        res = [] 
        for i,p in enumerate(metadata_phases):
            for row in p.iterrows():
                image_id = row[1]['image'].split('/')[-1].replace('.dcm', '')
                res.append(image_id)
        return res

    def get_row_from_image_ids(self, image_ids: List[str]):
        metadata_phases = self.get_all_gaze_data()
        res = []
        for i,p in enumerate(metadata_phases):
            for j,row in p.iterrows():
                image_id = row['image'].split('/')[-1].replace('.dcm', '')
                if image_id in image_ids:
                    res.append(row)

        return res

    def pull_labels_from_dicom_id(self, dicom_id, saved_dir):
        """
        dicom_id: dicom id of the image
        saved_dir: directory where the labels will be saved
        """
        saved_dir = os.path.join(saved_dir,"reflacx")
        saved_main_data = os.path.join(saved_dir,"main_data")
        saved_gaze_data = os.path.join(saved_dir,"gaze_data")
        # get id from dicom_id and metadata all phases
        metadata_phases = self.get_all_gaze_data()
        res = []
        found = False 
        for i,p in enumerate(metadata_phases):
            for j,row in p.iterrows():
                image_id = row['image'].split('/')[-1].replace('.dcm', '')
                if image_id == dicom_id:
                    res = row
                    found = True 
            if found:
                break
        # save the labels
        if not os.path.exists(saved_main_data):
            os.makedirs(saved_main_data)
        if not os.path.exists(saved_gaze_data):
            os.makedirs(saved_gaze_data)
        id = str(res['id'])
        os.system("cp {} -rf {}".format(os.path.join(self.main_dir,id), os.path.join(saved_main_data,f"{dicom_id}_{id}")))
        os.system("cp {} -rf {}".format(os.path.join(self.gaze_dir,id), os.path.join(saved_gaze_data,f"{dicom_id}_{id}")))

if __name__=="__main__":
    # check if reflacx uses the same images as mimic-cxr-jpg
    reflacx = REFLACX("./REFLACX")
    metadata_phases = [pd.read_csv(reflacx.main_dir/"metadata_phase_{}.csv".format(i)) for i in range(1, 4)]
    column_count = defaultdict(list)
    for i,p in enumerate(metadata_phases):
        for k in p.columns.tolist():
            column_count[k].append(i+1)
        for row in p.iterrows():
            image_path = row[1]['image'].replace('physionet.org/files/mimic-cxr/2.0.0/', './gcloud/mimic-cxr-jpg-2.0.0.physionet.org/').replace('.dcm', '.jpg')
            if not Path(image_path).exists():
                print(image_path)
                break

    print(dict(column_count))