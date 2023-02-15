import json
from pathlib import Path
import re
from typing import List
import pandas as pd


def parse_sent(sent):
    res = re.sub("[^a-zA-Z-]", " ", sent)
    res = res.strip().lower().split()
    return res


with open("data_here/new_metadata.json", "r") as f:
    data = json.load(f)


def __read(path):
    # get extension from path
    ext = Path(path).suffix
    if ext in [".csv"]:
        return pd.read_csv(path)
    elif ext in [".json"]:
        with open(path, "r") as f:
            return json.load(f)


def __is_reflacx_empty(path_list: List[str]):
    return path_list[0] == "" and len(path_list) == 1


dicom_ids = list(data.keys())
all_words = []
for dicom_id in dicom_ids:
    img_path = data[dicom_id]["img_path_jpg"]
    fixation_path = data[dicom_id]["fixation_reflacx"]
    is_reflacx = True
    if __is_reflacx_empty(fixation_path):
        fixation_path = data[dicom_id]["fixation_egd"]
        is_reflacx = False
    else:
        # we only care about the first fixation per patient for now
        fixation_path = fixation_path[0]
    if is_reflacx:
        for transcript_path in data[dicom_id]["transcript_reflacx"]:
            transcript = __read(transcript_path)
            sentences = [
                parse_sent(x) for x in " ".join(transcript["word"].values).split(".")
            ]
            all_words.extend(parse_sent(" ".join(transcript["word"].values)))
    else:
        transcript_path = data[dicom_id]["transcript_egd"]
        transcript = __read(transcript_path)
        sentences = [parse_sent(x) for x in transcript["full_text"].split(".")]
        all_words.extend(parse_sent(transcript["full_text"]))

mapping_dict = {
    "word2idx": {"<UNK>": 0, "<PAD>": 1, "<SOS>": 2, "<EOS>": 3, "<MASK>": 4},
    "idx2word": {0: "<UNK>", 1: "<PAD>", 2: "<SOS>", 3: "<EOS>", 4: "<MASK>"},
}

for i, word in enumerate(set(all_words)):
    if word not in mapping_dict:
        mapping_dict["word2idx"][word] = len(mapping_dict["word2idx"])
        mapping_dict["idx2word"][len(mapping_dict["idx2word"])] = word
with open("data_here/vocab.json", "w") as f:
    json.dump(mapping_dict, f)
