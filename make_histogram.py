"""Histogram of the data.
Length: 
    + sentence length
    + word length
    + full text length
    + number of sentences per text
    + gaze length
    + gaze length per sentence
Duplpicate:
    + sentence
    + word
    + full text
"""
import os
from os.path import join as pjoin
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm
import re

saved_dir = "exp/histograms"
os.makedirs(saved_dir, exist_ok=True)


def parse_sent(sent):
    res = re.sub("[^a-zA-Z-]", " ", sent)
    res = res.strip().lower().split()
    return res


# remove empty string element in list
remove_empty_string = lambda x: [i for i in x if i]
# sort the dict by value
sort_dict = lambda x: {
    k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)
}

reflacx_sentence_length = []
reflacx_word_length = []
reflacx_full_text_length = []
reflacx_num_sentences = []
reflacx_gaze_length = []
reflacx_gaze_duration = []
reflacx_gaze_length_per_sentence = []
reflacx_duplicate_sentence = defaultdict(int)
reflacx_duplicate_word = defaultdict(int)


egd_sentence_length = []
egd_word_length = []
egd_full_text_length = []
egd_num_sentences = []
egd_gaze_length = []
egd_gaze_duration = []
egd_gaze_length_per_sentence = []
egd_duplicate_sentence = defaultdict(int)
egd_duplicate_word = defaultdict(int)

with open("data_here/gazetoy_metadata.json") as f:
    metadata = json.load(f)

for dicom_id, v in metadata.items():
    # loop reflacx
    if len(v["reflacx_id"]) != 1:
        # continue
        for transcript_path, gaze_path in zip(
            v["transcript_reflacx"], v["gaze_reflacx"]
        ):
            # read transcript
            transcript = pd.read_csv(transcript_path)
            full_text = " ".join(transcript["word"].to_list()).replace(" .", ".")
            reflacx_full_text_length += [len(parse_sent(full_text))]
            sentences = remove_empty_string(full_text.split("."))
            for sent in sentences:
                reflacx_sentence_length += [len(parse_sent(sent))]
                reflacx_duplicate_sentence[" ".join(parse_sent(sent))] += 1
            sentence_per_text = len(sentences)
            reflacx_num_sentences += [sentence_per_text]
            words = remove_empty_string(full_text.split(" "))
            for word in words:
                reflacx_word_length += [len(word)]
                reflacx_duplicate_word[word] += 1

            is_start = True
            start = -1
            end = -1
            for _, row in transcript.iterrows():
                word, timestamp_start_word, timestamp_end_word = (
                    row["word"],
                    row["timestamp_start_word"],
                    row["timestamp_end_word"],
                )
                if is_start:
                    start = timestamp_start_word
                    is_start = False
                if word == ".":
                    end = timestamp_end_word
                    reflacx_gaze_length_per_sentence += [end - start]
                    is_start = True
            # read gaze
            gaze = pd.read_csv(gaze_path)
            duration = gaze["timestamp_sample"].to_numpy().max()
            reflacx_gaze_duration += [duration]
            reflacx_gaze_length += [len(gaze)]

    # egd
    if v["gaze_egd"] != "":
        # read transcript
        with open(v["transcript_egd"]) as f:
            transcript = json.load(f)
        full_text = transcript["full_text"]
        egd_full_text_length += [len(parse_sent(full_text))]
        sentences = remove_empty_string(full_text.split("."))
        for sent in sentences:
            egd_sentence_length += [len(parse_sent(sent))]
            egd_duplicate_sentence[" ".join(parse_sent(sent))] += 1
        sentence_per_text = len(sentences)
        egd_num_sentences += [sentence_per_text]
        if sentence_per_text == 0:
            with open(pjoin(saved_dir, "egd_no_sentence.txt"), "a") as f:
                f.write(f"{dicom_id}\n")
        words = remove_empty_string(full_text.split(" "))
        for word in words:
            egd_word_length += [len(word)]
            egd_duplicate_word[word] += 1

        start = -1
        end = -1
        is_start = True
        for element in transcript["time_stamped_text"]:
            if is_start:
                start = element["begin_time"]
                is_start = False
            if "." in element["phrase"]:
                end = element["end_time"]
                egd_gaze_length_per_sentence += [end - start]
                is_start = True
        # read gaze
        gaze = pd.read_csv(v["gaze_egd"])
        duration = gaze["Time (in secs)"].max()
        egd_gaze_duration += [duration]
        egd_gaze_length += [len(gaze)]

get_stat = (
    lambda x: "\n mean: "
    + str(np.mean(x).round(decimals=2))
    + " std: "
    + str(np.std(x).round(decimals=2))
    + " median: "
    + str(np.median(x).round(decimals=2))
    + "\n max: "
    + str(np.max(x).round(decimals=2))
    + " min: "
    + str(np.min(x).round(decimals=2))
)
# reflacx_sentence_length = []
# reflacx_word_length = []
# reflacx_full_text_length = []
# reflacx_num_sentences = []
# reflacx_gaze_length = []
# reflacx_gaze_duration = []
# reflacx_gaze_length_per_sentence = []
# plot reflacx hist length and gaze
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs[0, 0].hist(reflacx_sentence_length, bins=100)
axs[0, 0].set_title("reflacx sentence length" + get_stat(reflacx_sentence_length))
axs[0, 1].hist(reflacx_word_length, bins=100)
axs[0, 1].set_title("reflacx word length" + get_stat(reflacx_word_length))
axs[0, 2].hist(reflacx_full_text_length, bins=100)
axs[0, 2].set_title("reflacx full text length" + get_stat(reflacx_full_text_length))
axs[1, 0].hist(reflacx_num_sentences, bins=100)
axs[1, 0].set_title("reflacx num sentences" + get_stat(reflacx_num_sentences))
axs[1, 1].hist(reflacx_gaze_length, bins=100)
axs[1, 1].set_title("reflacx gaze length" + get_stat(reflacx_gaze_length))
axs[1, 2].hist(reflacx_gaze_duration, bins=100)
axs[1, 2].set_title("reflacx gaze duration" + get_stat(reflacx_gaze_duration))
axs[2, 0].hist(reflacx_gaze_length_per_sentence, bins=100)
axs[2, 0].set_title(
    "reflacx gaze length per sentence" + get_stat(reflacx_gaze_length_per_sentence)
)
fig.tight_layout()
# plt.show()
plt.savefig(pjoin(saved_dir, "reflacx_hist.png"))
with open(pjoin(saved_dir, "reflacx_hists.json"), "w") as f:
    res = {
        "reflacx_sentence_length": reflacx_sentence_length,
        "reflacx_word_length": reflacx_word_length,
        "reflacx_full_text_length": reflacx_full_text_length,
        "reflacx_num_sentences": reflacx_num_sentences,
        "reflacx_gaze_length": reflacx_gaze_length,
        "reflacx_gaze_duration": reflacx_gaze_duration,
        "reflacx_gaze_length_per_sentence": reflacx_gaze_length_per_sentence,
        "reflacx_duplicate_sentence": sort_dict(reflacx_duplicate_sentence),
        "reflacx_duplicate_word": sort_dict(reflacx_duplicate_word),
    }
    json.dump(res, f)


# egd_sentence_length = []
# egd_word_length = []
# egd_full_text_length = []
# egd_num_sentences = []
# egd_gaze_length = []
# egd_gaze_duration = []
# egd_gaze_length_per_sentence = []
# plot egd hist length and gaze. Then save the fig

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
axs[0, 0].hist(egd_sentence_length, bins=100)
axs[0, 0].set_title("egd sentence length" + get_stat(egd_sentence_length))
axs[0, 1].hist(egd_word_length, bins=100)
axs[0, 1].set_title("egd word length" + get_stat(egd_word_length))
axs[0, 2].hist(egd_full_text_length, bins=100)
axs[0, 2].set_title("egd full text length" + get_stat(egd_full_text_length))
axs[1, 0].hist(egd_num_sentences, bins=100)
axs[1, 0].set_title("egd num sentences" + get_stat(egd_num_sentences))
axs[1, 1].hist(egd_gaze_length, bins=100)
axs[1, 1].set_title("egd gaze length" + get_stat(egd_gaze_length))
axs[1, 2].hist(egd_gaze_duration, bins=100)
axs[1, 2].set_title("egd gaze duration" + get_stat(egd_gaze_duration))
axs[2, 0].hist(egd_gaze_length_per_sentence, bins=100)
axs[2, 0].set_title(
    "egd gaze length per sentence" + get_stat(egd_gaze_length_per_sentence)
)
fig.tight_layout()
# plt.show()
plt.savefig(pjoin(saved_dir, "egd.png"))

with open(pjoin(saved_dir, "egd_hists.json"), "w") as f:
    res = {
        "egd_sentence_length": egd_sentence_length,
        "egd_word_length": egd_word_length,
        "egd_full_text_length": egd_full_text_length,
        "egd_num_sentences": egd_num_sentences,
        "egd_gaze_length": egd_gaze_length,
        "egd_gaze_duration": egd_gaze_duration,
        "egd_gaze_length_per_sentence": egd_gaze_length_per_sentence,
        "egd_duplicate_sentence": sort_dict(egd_duplicate_sentence),
        "egd_duplicate_word": sort_dict(egd_duplicate_word),
    }
    json.dump(res, f)
