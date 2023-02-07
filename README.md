# Torchism 
General template for my PyTorch projects.
I will add more examples other than supervised mnist in the future.

## Dependency 
```
pip install -r requirements.txt
```

## Data folder 
meta
```
"dicom_id": {"img_path_jpg", "img_path_dcm", "transcript_egd", "gaze_egd", "transcript_reflacx", "gaze_reflacx"}
```

## Sanity run 
Try using the script in `data_generator` or download mnist data (.csv) from `https://github.com/pjreddie/mnist-csv-png` and save it same as the paths in `configs/train/sample.yaml`. 
Remember to set env variable before run the train.py script, and change the trainer logger to Neptune logger (currently we are using neptune logger)
```bash
export NEPTUNE_API_TOKEN="<key>"
python train.py --config configs/train/sample.yaml --gpus 0
```

Or if you want a quick test, change the neptune logger in the trainer to tensorboard and run this:
```
python train.py --config configs/train/sample.yaml --gpus 0
tensorboard --logdir=runs 
```
