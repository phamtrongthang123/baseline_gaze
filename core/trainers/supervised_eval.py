import os
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from ..utils.random_seed import set_seed
from ..utils.getter import get_instance
from ..utils.meter import AverageValueMeter
from ..utils.device import move_to, detach
from ..utils.exponential_moving_average import ExponentialMovingAverage
from ..utils.decoding import save2json, tensor2words
from ..loggers import TensorboardLogger, NeptuneLogger
from ..metrics import GazeTranscript

__all__ = ["SupervisedEvaluator"]


class SupervisedEvaluator:
    def __init__(self, config):
        super().__init__()

        self.load_config_dict(config)
        self.config = config

        # Train ID
        self.train_id = self.config.get("id", "None")
        self.train_id += "-" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Get arguments
        self.nepochs = self.config["trainer"]["nepochs"]
        self.log_step = self.config["trainer"]["log_step"]
        self.val_step = self.config["trainer"]["val_step"]
        self.debug = self.config["debug"]

        # Instantiate global variables
        self.best_loss = np.inf
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.val_loss = list()
        self.val_metric = {k: list() for k in self.metric.keys()}

        # get dir of path pretrained model using os module
        self.save_dir = os.path.dirname(self.config["pretrained"])
        # Instantiate loggers
        # self.tsboard = TensorboardLogger(path=self.save_dir)
        # self.tsboard = NeptuneLogger(project_name="thesis-master/torchism", name="test", path=self.save_dir, model_params=config)
        self.amp = False
        if "amp" in config:
            self.amp = config["amp"]
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

    def load_config_dict(self, config):
        # Get device
        dev_id = (
            "cuda:{}".format(config["gpus"])
            if torch.cuda.is_available() and config.get("gpus", None) is not None
            else "cpu"
        )
        self.device = torch.device(dev_id)

        # Get pretrained model
        pretrained_path = config["pretrained"]

        pretrained = None
        if pretrained_path != None:
            pretrained = torch.load(pretrained_path, map_location=dev_id)
            for item in ["model"]:
                config[item] = pretrained["config"][item]

        # 2: Define network
        set_seed(config["seed"])
        self.model = get_instance(config["model"]).to(self.device)

        # Train from pretrained if it is not None
        if pretrained is not None:
            self.model.load_state_dict(pretrained["model_state_dict"])

        # 3: Define loss
        set_seed(config["seed"])
        self.criterion = get_instance(config["loss"]).to(self.device)

        # 4: Define Optimizer
        set_seed(config["seed"])
        self.optimizer = get_instance(
            config["optimizer"], params=self.model.parameters()
        )
        if pretrained is not None:
            self.optimizer.load_state_dict(pretrained["optimizer_state_dict"])

        # 5: Define Scheduler
        set_seed(config["seed"])
        self.scheduler = get_instance(
            config["scheduler"],
            optimizer=self.optimizer,
            t_total=config["trainer"]["nepochs"],
        )

        # 6: Define metrics
        set_seed(config["seed"])
        self.metric = {mcfg["name"]: 0.0 for mcfg in config["metric"]}

    @torch.no_grad()
    def val_epoch(self, dataloader):
        dataset = dataloader.dataset
        running_loss = AverageValueMeter()

        self.model.eval()
        print("Evaluating........")
        sentences = []
        dicom_ids = []
        sentences_gt = []
        dicom_ids_gt = []

        progress_bar = tqdm(dataloader)
        for i, (indexes, img, fixation, fix_masks, transcript, sent_masks) in enumerate(
            progress_bar
        ):
            # 1: Load img_inputs and labels
            img = move_to(img, self.device)
            fixation = move_to(fixation, self.device)
            fix_masks = move_to(fix_masks, self.device)
            transcript = move_to(transcript, self.device)
            sent_masks = move_to(sent_masks, self.device)
            transcript_inp, transcript_out = transcript[:, :, :-1], transcript[:, :, 1:]
            sent_masks_inp, sent_masks_out = sent_masks[:, :, :-1], sent_masks[:, :, 1:]
            # 2: Get network outputs

            outs, probs = self.model.generate_greedy(img, fixation, fix_masks)
            # outs = self.model(img, fixation, fix_masks, transcript_inp, sent_masks_inp)
            # 3: Calculate the loss
            loss = self.model.build_loss(probs, transcript_out, sent_masks_out)
            # loss = self.model.build_loss(outs, transcript_out, sent_masks_out)
            # 4: Update loss
            running_loss.add(loss.item())
            # # 5: Update metric
            # outs = detach(outs)
            # lbl = detach(lbl)
            # for m in self.metric.values():
            #     m.update(outs, lbl)

            # store sentences and dicom_ids
            sentences.extend([tensor2words(outs, dataset.vocab)])
            dicom_ids.extend([dataset.dicom_ids[idx] for idx in indexes])
            sentences_gt.extend([tensor2words(transcript_out.squeeze(0), dataset.vocab)])
            dicom_ids_gt.extend([dataset.dicom_ids[idx] for idx in indexes])

        print("+ Evaluation result")
        avg_loss = running_loss.value()[0]
        print("Loss:", avg_loss)
        self.val_loss.append(avg_loss)

        res = {dicom_id: sentence for dicom_id, sentence in zip(dicom_ids, sentences)}
        save2json(res, os.path.join(self.save_dir, "val_result.json"))
        gt_res = {dicom_id: sentence for dicom_id, sentence in zip(dicom_ids_gt, sentences_gt)}
        save2json(gt_res, os.path.join(self.save_dir, "val_result_gt.json"))
        metric_instance = GazeTranscript(ground_truth_filenames=os.path.join(self.save_dir, "val_result_gt.json"), prediction_filename=os.path.join(self.save_dir, "val_result.json"), verbose=True)
        scores = metric_instance.evaluate_transcript()
        for k in self.metric.keys():
            m = scores[k]
            self.val_metric[k].append(m)

    def eval(self, val_dataloader):
        set_seed(self.config["seed"])
        self.val_epoch(dataloader=val_dataloader)
        #
