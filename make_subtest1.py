import os 
import sys 
import numpy as np 
from pathlib import Path
import json 
import pandas as pd
from tqdm import tqdm

if __name__=="__main__":
    split = 'val'
    annotationjson = []
    with open(f'data_here/reflacx_new_metadata_{split}.json') as f:
        metadata = json.load(f)
    for dicom_id, val in tqdm(metadata.items()):
        # read transcript 
        reflacx_ids = [] 
        transcripts = [] 
        for reflacx_id, transcript_path in zip(val['reflacx_id'], val['transcript_reflacx']):
            transcript_df = pd.read_csv(transcript_path)
            sentences = " ".join(transcript_df["word"].values).replace(" .", ".")
            transcripts.append(sentences)
            reflacx_ids.append(reflacx_id)
        annotationjson.append({'dicom_id': dicom_id, 'reflacx_id': reflacx_ids, 'transcript': transcripts})
    with open(f'data_here/reflacx_new_metadata_{split}_subtest1.json', 'w') as f:
        json.dump(annotationjson, f)
    