import os
from datetime import datetime

import torch
import numpy as np
from torch import nn
from tqdm import tqdm

from core.utils.decoding import save2json, tensor2words

from ..utils.random_seed import set_seed
from ..utils.getter import get_instance
from ..utils.meter import AverageValueMeter
from ..utils.device import move_to, detach
from ..utils.exponential_moving_average import ExponentialMovingAverage
from ..loggers import TensorboardLogger, NeptuneLogger
from ..metrics import GazeTranscript

__all__ = ["SupervisedTrainer"]


class SupervisedTrainer:
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

        # Instantiate loggers
        self.save_dir = os.path.join(self.config["trainer"]["log_dir"], self.train_id)
        self.tsboard = TensorboardLogger(path=self.save_dir)
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

    def save_checkpoint(self, epoch, val_loss, val_metric):
        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        torch.save(data, os.path.join(self.save_dir, "last.pth"))

        if val_loss < self.best_loss:
            print(
                f"Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights..."
            )
            torch.save(data, os.path.join(self.save_dir, "best_loss.pth"))
            # Update best_loss
            self.best_loss = val_loss
        else:
            print(f"Loss is not improved from {self.best_loss:.6f}.")

        if self.metric is not None and val_metric is not None:
            for k in self.metric.keys():
                if val_metric[k] > self.best_metric[k]:
                    print(
                        f"{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights..."
                    )
                    torch.save(
                        data, os.path.join(self.save_dir, f"best_metric_{k}.pth")
                    )
                    self.best_metric[k] = val_metric[k]
                else:
                    print(f"{k} is not improved from {self.best_metric[k]:.6f}.")

    def train_epoch(self, epoch, dataloader):
        # 0: Record loss during training process
        running_running_loss = AverageValueMeter()
        running_total_loss = AverageValueMeter()
        running_num_loss = AverageValueMeter()
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        num_loss = AverageValueMeter()
        self.model.train()
        print("Training........")
        progress_bar = tqdm(dataloader)
        for i, (img, fixation, fix_masks, transcript, sent_masks) in enumerate(
            progress_bar
        ):
            # 1: Load img_inputs and labels
            img = move_to(img, self.device)
            fixation = move_to(fixation, self.device)
            fix_masks = move_to(fix_masks, self.device)
            transcript = move_to(transcript, self.device)
            sent_masks = move_to(sent_masks, self.device)

            transcript_inp, transcript_out = (
                transcript[:, :, :-1].clone(),
                transcript[:, :, 1:].clone(),
            )
            sent_masks_inp, sent_masks_out = (
                sent_masks[:, :, :-1].clone(),
                sent_masks[:, :, 1:].clone(),
            )
            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                # 3: Get network outputs
                outs, num = self.model(
                    img, fixation, fix_masks, transcript_inp, sent_masks_inp
                )
                # 4: Calculate the loss
                loss = self.model.build_loss(outs, transcript_out, sent_masks_out)
                num_loss_ = nn.MSELoss()(num, torch.tensor([sent_masks.shape[1]]).type_as(num))



            if self.scaler is not None:
                self.scaler.scale(loss+num_loss_).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 5: Calculate gradients
                # (loss+num_loss_).backward()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                # 6: Performing backpropagation
                self.optimizer.step()
            with torch.no_grad():
                # 7: Update loss
                running_loss.add(loss.item())
                num_loss.add(num_loss_.item())
                total_loss.add(num_loss_.item() + loss.item())
                running_running_loss.add(loss.item())
                running_num_loss.add(num_loss_.item())
                running_total_loss.add(num_loss_.item() + loss.item())

                if (i + 1) % self.log_step == 0 or (i + 1) == len(dataloader):
                    self.tsboard.update_loss(
                        "train", running_total_loss.value()[0], epoch * len(dataloader) + i
                    )
                    running_total_loss.reset()
                    self.tsboard.update_metric("train", "caption_loss", running_running_loss.value()[0], epoch * len(dataloader) + i)
                    self.tsboard.update_metric("train", "num_loss", running_num_loss.value()[0], epoch * len(dataloader) + i)
                    running_running_loss.reset()
                    running_num_loss.reset()

                # 8: Update metric
                # outs = detach(outs)
                # lbl = detach(lbl)
                # for m in self.metric.values():
                #     m.update(outs, lbl)
                # break
        print("+ Training result")
        avg_loss = running_loss.value()[0]
        print("Caption Loss:", avg_loss)
        print("Num Loss:", num_loss.value()[0])
        print("Total Loss:", total_loss.value()[0])

        # for k in self.metric.keys():
        #     m = self.metric[k].value()
        #     self.metric[k].summary()
        #     self.tsboard.update_metric('train', k, m, epoch)

    @torch.no_grad()
    def val_epoch(self, epoch, dataloader):
        dataset = dataloader.dataset
        running_loss = AverageValueMeter()
        total_loss = AverageValueMeter()
        num_loss = AverageValueMeter()

        self.model.eval()
        print("Evaluating........")
        sentences = []
        dicom_ids = []
        sentences_gt = []
        dicom_ids_gt = []
        nums = []
        nums_gt = []

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

            outs, probs, num = self.model.generate_greedy(img, fixation, fix_masks)
            # outs = self.model(img, fixation, fix_masks, transcript_inp, sent_masks_inp)
            # 3: Calculate the loss
            loss = self.model.build_loss(probs, transcript_out, sent_masks_out)
            num_loss_ = nn.MSELoss()(num, torch.tensor([sent_masks.shape[1]]).type_as(num))

            # loss = self.model.build_loss(outs, transcript_out, sent_masks_out)
            # 4: Update loss
            running_loss.add(loss.item())
            num_loss.add(num_loss_.item())
            total_loss.add(loss.item() + num_loss_.item())
            # # 5: Update metric
            # outs = detach(outs)
            # lbl = detach(lbl)
            # for m in self.metric.values():
            #     m.update(outs, lbl)

            # store sentences and dicom_ids
            sentences.extend([tensor2words(outs, dataset.vocab)])
            dicom_ids.extend([dataset.dicom_ids[idx] for idx in indexes])
            nums.extend([num.item() ])
            sentences_gt.extend([tensor2words(transcript_out.squeeze(0), dataset.vocab)])
            dicom_ids_gt.extend([dataset.dicom_ids[idx] for idx in indexes])
            nums_gt.extend([sent_masks.shape[1]])
            # break

        print("+ Evaluation result")
        avg_loss = running_loss.value()[0]
        print("Caption Loss:", avg_loss)
        print("Num Loss:", num_loss.value()[0])
        print("Total Loss:", total_loss.value()[0])
        self.val_loss.append(total_loss.value()[0])
        self.tsboard.update_loss("val", total_loss.value()[0], epoch)
        self.tsboard.update_metric("val", "caption_loss", running_loss.value()[0], epoch)
        self.tsboard.update_metric("val", "num_loss", num_loss.value()[0], epoch)
        res = {dicom_id: sentence for dicom_id, sentence in zip(dicom_ids, sentences)}
        save2json(res, os.path.join(self.save_dir, f"val_result_{epoch}.json"))
        gt_res = {dicom_id: sentence for dicom_id, sentence in zip(dicom_ids_gt, sentences_gt)}
        save2json(gt_res, os.path.join(self.save_dir, "val_result_gt.json"))
        metric_instance = GazeTranscript(ground_truth_filenames=os.path.join(self.save_dir, "val_result_gt.json"), prediction_filename=os.path.join(self.save_dir, f"val_result_{epoch}.json"), verbose=True)
        scores = metric_instance.evaluate_transcript()
        for k in self.metric.keys():
            m = scores[k]
            self.val_metric[k].append(m)
            self.tsboard.update_metric('val', k, m, epoch)
        
        res_num = {dicom_id: num for dicom_id, num in zip(dicom_ids, nums)}
        save2json(res_num, os.path.join(self.save_dir, f"val_result_num_{epoch}.json"))
        gt_res_num = {dicom_id: num for dicom_id, num in zip(dicom_ids_gt, nums_gt)}
        save2json(gt_res_num, os.path.join(self.save_dir, "val_result_num_gt.json"))
    def train(self, train_dataloader, val_dataloader):
        set_seed(self.config["seed"])
        # add graph to tensorboard
        dataiter = iter(train_dataloader)
        img, fixation, fix_masks, transcript, sent_masks = next(dataiter)
        img = move_to(img, self.device)
        fixation = move_to(fixation, self.device)
        fix_masks = move_to(fix_masks, self.device)
        transcript = move_to(transcript, self.device)
        sent_masks = move_to(sent_masks, self.device)

        transcript_inp, transcript_out = transcript[:, :, :-1], transcript[:, :, 1:]
        sent_masks_inp, sent_masks_out = sent_masks[:, :, :-1], sent_masks[:, :, 1:]
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # 3: Get network outputs
            self.tsboard.add_graph(
                self.model, (img, fixation, fix_masks, transcript_inp, sent_masks_inp)
            )
        for epoch in range(self.nepochs):
            print("\nEpoch {:>3d}".format(epoch))
            print("-----------------------------------")

            # Note learning rate
            for i, group in enumerate(self.optimizer.param_groups):
                self.tsboard.update_lr(i, group["lr"], epoch)

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)

            print()

            # 2: Evalutation phase
            if (epoch + 1) % self.val_step == 0:
                #     # 2: Evaluating model
                self.val_epoch(epoch, dataloader=val_dataloader)
                print("-----------------------------------")

                #     # 3: Learning rate scheduling
                #     # unless it is ReduceLROnPlateau, we don't need to pass the metric value
                #     # self.scheduler.step(self.val_loss[-1])
                self.scheduler.step()

                # 4: Saving checkpoints
                if not self.debug:
                    # Get latest val loss here
                    val_loss = self.val_loss[-1]
                    # val_metric = {k: m[-1] for k, m in self.val_metric.items()}
                    self.save_checkpoint(epoch, val_loss, None)
