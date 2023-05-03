#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_torchism/
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train/gazebaseline7_subtest1_overfit.yaml --gpus 0
