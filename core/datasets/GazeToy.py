import re
import pandas as pd
import torch
import torchvision.transforms as transforms
import pickle
from typing import Any, Callable, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import json

from tqdm import tqdm

__all__ = ["GazeToy"]


def parse_sent(sent):
    res = re.sub("[^a-zA-Z-]", " ", sent)
    res = res.strip().lower().split()
    return res


class GazeToy(torch.utils.data.Dataset):
    def __init__(self, metadata, vocab, is_train=True):
        self.data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }
        self.metadata = metadata
        self.is_train = is_train
        with open(self.metadata, "r") as f:
            self.data = json.load(f)
        self.dicom_ids = list(self.data.keys())
        self.dicom_ids = self.dicom_ids[:8]
        with open(vocab, "r") as f:
            self.vocab = json.load(f)

        self.max_sent_len = 50
        self.max_fix_len = 400  # reflacx largest is 386, min is 15

    def __getitem__(self, index):
        dicom_id = self.dicom_ids[index]
        img_path = self.data[dicom_id]["img_path_jpg"]
        if self.is_train:
            img = Image.open(img_path)
            img = self.data_transforms["train"](img)
        else:
            img = Image.open(img_path)
            img = self.data_transforms["val"](img)
        fixation_path = self.data[dicom_id]["fixation_reflacx"]
        is_reflacx = True
        if self.__is_reflacx_empty(fixation_path):
            fixation_path = self.data[dicom_id]["fixation_egd"]
            is_reflacx = False
        else:
            # we only care about the first fixation per patient for now
            fixation_path = fixation_path[0]
        fixation_data = self.__read(fixation_path)
        fixation, ftimestamp_s, ftimestamp_e = self.__get_fixation_and_timestamp(
            fixation_data, is_reflacx
        )
        # fixation [fix_len, 2], duration [fix_len,1], torch cat them together to [fix_len, 3]
        fixation = torch.cat((fixation, ftimestamp_e - ftimestamp_s), dim=1)
        fixation[:, 2] = fixation[:, 2] / fixation[:, 2].max()
        fixation, fix_masks = self.__padding_mask(fixation)
        if is_reflacx:
            transcript_path = self.data[dicom_id]["transcript_reflacx"][0]
        else:
            transcript_path = self.data[dicom_id]["transcript_egd"]
        transcript = self.__read(transcript_path)
        # split the transcript into sentences
        transcript, sent_masks, timestamp_s, timestamp_e = self.__split_sentences(
            transcript, is_reflacx
        )
        # convert to index from vocab
        transcript = self.__word2index(transcript)
        # convert to tensor [N_seq, seq_len,]
        transcript = torch.tensor(transcript)
        timestamp_e = torch.tensor(timestamp_e)
        timestamp_s = torch.tensor(timestamp_s)
        sent_masks = torch.tensor(sent_masks)
        # only keep the fixation that is within the transcript timestamp start and end
        # fixation = self.__filter_fixation(fixation, ftimestamp_s, ftimestamp_e, timestamp_s, timestamp_e)
        return img, fixation, fix_masks, transcript, sent_masks

    def __padding_mask(self, fixation):
        lf = len(fixation)
        if lf < self.max_fix_len:
            padding = torch.zeros(self.max_fix_len - lf, 3)
            fixation = torch.cat((fixation, padding), dim=0)
            fix_masks = torch.cat(
                (torch.ones(lf, 1), torch.zeros(self.max_fix_len - lf, 1)), dim=0
            )
        else:
            fixation = fixation[: self.max_fix_len]
            fix_masks = torch.ones(self.max_fix_len, 1)
        return fixation, fix_masks

    def __len__(self):
        return len(self.dicom_ids)

    def __filter_fixation(
        self,
        fixation,
        ftimestamp_s,
        ftimestamp_e,
        timestamp_s,
        timestamp_e,
        threshold=2,
    ):
        """This function will filter out the fixation that is not within the timestamp start and end.
        But the data between fixation and timestamp is mismatched so much, I keep getting empty set, that this function can't be used. This function only works with gaze.
        Args:

        Returns:
            fixation: [N_seq, fix_len, 3]
        """
        # ftimestamp_s == ftimestamp_e == [fix_len, 1], timestamp_s == timestamp_e == [N_seq, 1]
        res = []
        for i in range(len(timestamp_s)):
            tmp = []
            for j in range(len(ftimestamp_s)):
                if (
                    ftimestamp_s[j] >= timestamp_s[i]
                    and ftimestamp_e[j] <= timestamp_e[i]
                ):
                    # t : [ --------- ]
                    # ft:    [ ---- ]
                    tmp.append(fixation[j])
                elif (timestamp_s[i] - ftimestamp_s[j] < threshold) and ftimestamp_e[
                    j
                ] <= timestamp_e[i]:
                    # t :     [ ---- ]
                    # ft:   [ ---- ]
                    tmp.append(fixation[j])
                elif ftimestamp_s[j] >= timestamp_s[i] and (
                    ftimestamp_e[j] - timestamp_e[i] < threshold
                ):
                    # t : [ ---- ]
                    # ft:    [ ---- ]
                    tmp.append(fixation[j])
                elif (timestamp_s[i] - ftimestamp_s[j] < threshold) and (
                    ftimestamp_e[j] - timestamp_e[i] < threshold
                ):
                    # t :   [ ---- ]
                    # ft: [ ------- ]. diff is small
                    tmp.append(fixation[j])
            res.append(torch.stack(tmp, dim=0))
        return torch.cat(res, dim=0)

    def __word2index(self, transcript):
        """This function will convert the word in the transcript to index from the vocab"""
        res = []
        for sent in transcript:
            res.append([self.vocab["word2idx"][word] for word in sent])
        return res

    def __get_fixation_and_timestamp(self, fixation_data, is_reflacx):
        """This function will get the fixation and timestamp from the fixation_data dictionary

        Args:
            fixation_data (_type_): _description_
            is_reflacx (bool): _description_
        """
        x = torch.tensor(fixation_data["x"])
        y = torch.tensor(fixation_data["y"])
        fixation = torch.stack((x, y), dim=1)
        # fixation = fixation / fixation.max()
        start_time = torch.tensor(fixation_data["start_time"]).unsqueeze(1)
        end_time = torch.tensor(fixation_data["end_time"]).unsqueeze(1)
        return fixation, start_time, end_time

    def __padding(self, sentences):
        """This function will pad the sentence to the max_len, and return the padded sentence with mask

        Args:
            sentence (_type_): _description_
        """
        masks = []
        new_sentences = []
        for sentence in sentences:
            # mask = [1.0] * (len(sentence)-1) + [0.0] # ignore end token
            mask = [1.0] * len(sentence)  # count end token
            if len(sentence) < self.max_sent_len:
                ls = len(sentence)
                sentence += ["<PAD>"] * (self.max_sent_len - ls)
                mask += [0.0] * (self.max_sent_len - ls)
            masks.append(mask)
            new_sentences.append(sentence)
        return new_sentences, masks

    def __split_sentences(self, transcript, is_reflacx):
        """This functions splits the transcript data into sentences and returns the sentence (in words list) and timestamp.
        Note that transcript of reflacx is dataframe and egd is dictionary


        Args:
            transcript (_type_): _description_
            is_reflacx (bool): _description_
        """
        timestamp_s = []
        timestamp_e = []
        if is_reflacx:
            sentences = [
                ["<SOS>"] + parse_sent(x) + ["<EOS>"]
                for x in " ".join(transcript["word"].values).split(".")
                if x not in ["", " "]
            ]
            # sentences =  [["<SOS>"] + parse_sent(x) + ["<EOS>"] for x in ' '.join(transcript['word'].values).replace('.','').split('.') if x not in ['', ' '] ]
            sentences, masks = self.__padding(sentences)
            is_start = True
            for _, row in transcript.iterrows():
                word, timestamp_start_word, timestamp_end_word = (
                    row["word"],
                    row["timestamp_start_word"],
                    row["timestamp_end_word"],
                )
                if is_start:
                    timestamp_s.append(timestamp_start_word)
                    is_start = False
                if word == ".":
                    timestamp_e.append(timestamp_end_word)
                    is_start = True
        else:
            sentences = [
                ["<SOS>"] + parse_sent(x) + ["<EOS>"]
                for x in transcript["full_text"].split(".")
                if x not in ["", " "]
            ]
            sentences, masks = self.__padding(sentences)
            is_start = True
            for element in transcript["time_stamped_text"]:
                if is_start:
                    timestamp_s.append(element["begin_time"])
                    is_start = False
                if "." in element["phrase"]:
                    end = element["end_time"]
                    timestamp_e.append(end)
                    is_start = True
        return sentences, masks, timestamp_s, timestamp_e

    def __read(self, path):
        # get extension from path
        ext = Path(path).suffix
        if ext in [".csv"]:
            return pd.read_csv(path)
        elif ext in [".json"]:
            with open(path, "r") as f:
                return json.load(f)

    def __is_reflacx_empty(self, path_list: List[str]):
        return path_list[0] == "" and len(path_list) == 1


if __name__ == "__main__":
    ds = GazeToy(
        metadata="/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json",
        vocab="/home/ptthang/gaze_sample/data_here/vocab.json",
    )
    print(len(ds))
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=1, shuffle=True)
    # enumerate tqdm for the dataset
    mx, mn = 0, 100000
    for i, (img, fixation, fix_masks, transcript, sent_masks) in enumerate(tqdm(dl)):
        if i == 0:
            print(
                img.shape,
                fixation.shape,
                fix_masks.shape,
                transcript.shape,
                sent_masks.shape,
            )
        if mx < fixation.shape[1]:
            mx = fixation.shape[1]
        if mn > fixation.shape[1]:
            mn = fixation.shape[1]
        # continue
        # print(img.shape, fixation.shape, transcript)
        # break # only break for one sample, if it is ok, cmt break and run the whole dataset to make sure it runs

    print(mx, mn)
