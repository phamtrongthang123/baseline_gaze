import os 
import sys 
import numpy as np 
from pathlib import Path
import json 
import pandas as pd
from tqdm import tqdm
import re
from collections import Counter


def create_vocabulary(ann):
    total_tokens = []

    for dicom_id, example in ann.items():
        tokens = clean_report_mimic_cxr(" ".join(example['transcript'])).split()
        for token in tokens:
            total_tokens.append(token)

    counter = Counter(total_tokens)
    vocab = [k for k, v in counter.items() if v >= 1]
    vocab.sort()
    vocab = ["<UNK>","<PAD>", "<SOS>", "<EOS>", "<MASK>"] + vocab
    token2idx, idx2token = {}, {}
    for idx, token in enumerate(vocab):
        token2idx[token] = idx
        idx2token[idx] = token
    return token2idx, idx2token

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report
if __name__=="__main__":
    split = 'val'
    annotationjson = {}
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
            if mx < len(sentences.split()):
                mx = len(sentences.split())
        annotationjson[dicom_id] ={ 'reflacx_id': reflacx_ids, 'transcript': transcripts}
    with open(f'data_here/reflacx_new_metadata_{split}_subtest1.json', 'w') as f:
        json.dump(annotationjson, f)
    token2idx, idx2token = create_vocabulary(annotationjson)
    with open(f'data_here/reflacx_new_metadata_{split}_subtest1_vocab.json', 'w') as f:
        json.dump({'word2idx': token2idx, 'idx2word': idx2token}, f)
