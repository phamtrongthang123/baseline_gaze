# better train on only reflacx or egd first.
import os
import json
from typing import List


def __is_reflacx_empty(path_list: List[str]):
    return path_list[0] == "" and len(path_list) == 1


with open("data_here/new_metadata.json", "r") as f:
    data = json.load(f)

reflacxs = {}
egds = {}
for k, v in data.items():
    if not __is_reflacx_empty(v["fixation_reflacx"]):
        reflacxs[k] = v
    else:
        egds[k] = v


with open("data_here/reflacx_new_metadata.json", "w") as f:
    json.dump(reflacxs, f)

with open("data_here/egd_new_metadata.json", "w") as f:
    json.dump(egds, f)
