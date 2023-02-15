#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_torchism/
CUDA_VISIBLE_DEVICES=0 python test.py --config configs/val/gazebaseline_toy.yaml --gpus 0
