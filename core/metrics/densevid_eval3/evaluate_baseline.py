# --------------------------------------------------------
# evaluation scripts for dense video captioning, support python 3
# Modified from https://github.com/ranjaykrishna/densevid_eval/tree/9d4045aced3d827834a5d2da3c9f0692e3f33c1c
# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# --------------------------------------------------------

import argparse
import json
import random
import string
import sys
import time
# sys.path.insert(0, './coco-caption') # Hack to allow the import of pycocoeval

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.bleu.bleu import Bleu
from .pycocoevalcap.meteor.meteor import Meteor
from .pycocoevalcap.rouge.rouge import Rouge
from .pycocoevalcap.cider.cider import Cider
Set=set
import numpy as np

def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_length))

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class GazeTranscript(object):
    # PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000, verbose=False):
        # Check that the gt and submission files exist and load them
        if tious:
            if len(tious) == 0:
                raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        # self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.ground_truths_keys = [vid for gt in self.ground_truths for vid in gt ]
        print('available video number', len(set(self.ground_truths_keys) & set(self.prediction.keys())))

        # Set up scorers, if not verbose, we only use the one we're
        # testing on: METEOR
        self.tokenizer = PTBTokenizer()
        if self.verbose:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission...")
        submission = json.load(open(prediction_filename))
        # if not all([field in submission.keys() for field in self.pred_fields]):
        #     raise IOError('Please input a valid ground truth file.')
        # Ensure that every video is limited to the correct maximum number of proposals.
        return submission

    def import_ground_truths(self, filename):
        gts = self.import_prediction(filename)
        return gts

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
        iou = float(intersection) / (union + 1e-8)
        return iou

    def check_gt_exists(self, dicom_id):
        for gt in self.ground_truths:
            if dicom_id in gt:
              return True
        return False

    def get_gt_dicom_ids(self):
        dicom_ids = set(self.ground_truths.keys())
        return list(dicom_ids)

    def evaluate(self):
        aggregator = {}
        self.scores = {}
        for tiou in self.tious:
            scores = self.evaluate_tiou(tiou)
            for metric, score in scores.items():
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(score)
        if True:
        #if self.verbose:
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)

    def evaluate_detection(self, tiou):
        gt_dicom_ids = self.get_gt_dicom_ids()
        # Recall is the percentage of ground truth that is covered by the predictions
        # Precision is the percentage of predictions that are valid
        recall = [0] * len(gt_dicom_ids)
        precision = [0] * len(gt_dicom_ids)
        for vid_i, dicom_id in enumerate(gt_dicom_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if dicom_id not in gt:
                    continue
                refs = gt[dicom_id]
                ref_set_covered = set([])
                pred_set_covered = set([])
                num_gt = 0
                num_pred = 0
                if dicom_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[dicom_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)

                    new_precision = float(len(pred_set_covered)) / (pred_i + 1) 
                    best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        # This method averages the tIoU precision from METEOR, Bleu, etc. across videos 
        res = {}
        gts = {}
        gt_dicom_ids = self.get_gt_dicom_ids()
        
        unique_index = 0

        # video id to unique caption ids mapping
        dicom2capid = {}
        
        cur_res = {}
        cur_gts = {}
        
        
        for dicom_id in gt_dicom_ids:

            dicom2capid[dicom_id] = []

            # If the video does not have a prediction, then we give it no matches
            # We set it to empty, and use this as a sanity check later on
            if dicom_id not in self.prediction:
                pass

            # If we do have a prediction, then we find the scores based on all the
            # valid tIoU overlaps.
            else:
                # For each prediction, we look at the tIoU with ground truth.
                for pred in self.prediction[dicom_id]:
                    has_added = False
                    for gt in self.ground_truths:
                        if dicom_id not in gt:
                            continue
                        gt_captions = gt[dicom_id]
                        for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                            if self.iou(pred['timestamp'], caption_timestamp) >= tiou:
                                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                                cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                                dicom2capid[dicom_id].append(unique_index)
                                unique_index += 1
                                has_added = True

                    # If the predicted caption does not overlap with any ground truth,
                    # we should compare it with garbage.
                    if not has_added:
                        cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                        cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                        dicom2capid[dicom_id].append(unique_index)
                        unique_index += 1

        # Each scorer will compute across all videos and take average score
        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...'%(scorer.method()))
            
            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}
            
            # call tokenizer here for all predictions and gts
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)

            # reshape back
            for vid in dicom2capid.keys():
                res[vid] = {index:tokenize_res[index] for index in dicom2capid[vid]}
                gts[vid] = {index:tokenize_gts[index] for index in dicom2capid[vid]}

            for dicom_id in gt_dicom_ids:

                if len(res[dicom_id]) == 0 or len(gts[dicom_id]) == 0:
                    if type(method) == list:
                        score = [0] * len(method)
                    else:
                        score = 0
                else:
                    score, scores = scorer.compute_score(gts[dicom_id], res[dicom_id])
                all_scores[dicom_id] = score

            # print(all_scores.values())
            if type(method) == list:
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
                    if self.verbose:
                        print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method[m], output[method[m]]))
            else:
                output[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method, output[method]))
        return output

    def evaluate_transcript(self):
        # This method averages the scores from METEOR, Bleu, etc. across transcript sentences
        res = {}
        gts = {}
        gt_dicom_ids = self.get_gt_dicom_ids()
        
        unique_index = 0

        # video id to unique caption ids mapping
        dicom2capid = {}
        
        cur_res = {}
        cur_gts = {}
        
        
        for dicom_id in gt_dicom_ids:

            dicom2capid[dicom_id] = []

            # If the image does not have a prediction, then we give it no matches
            # We set it to empty, and use this as a sanity check later on
            if dicom_id not in self.prediction:
                pass

            else:
                for pred, gt in zip(self.prediction[dicom_id], self.ground_truths[dicom_id]):
                    cur_res[unique_index] = [{'caption': remove_nonascii(pred)}]
                    cur_gts[unique_index] = [{'caption': remove_nonascii(gt)}]
                    dicom2capid[dicom_id].append(unique_index)
                    unique_index += 1

        # Each scorer will compute across all videos and take average score
        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...'%(scorer.method()))
            
            # For each video, take all the valid pairs (based from tIoU) and compute the score
            all_scores = {}
            
            # call tokenizer here for all predictions and gts
            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)

            # reshape back
            for uid in dicom2capid.keys():
                res[uid] = {index:tokenize_res[index] for index in dicom2capid[uid]}
                gts[uid] = {index:tokenize_gts[index] for index in dicom2capid[uid]}

            for dicom_id in gt_dicom_ids:

                if len(res[dicom_id]) == 0 or len(gts[dicom_id]) == 0:
                    if type(method) == list:
                        score = [0] * len(method)
                    else:
                        score = 0
                else:
                    score, scores = scorer.compute_score(gts[dicom_id], res[dicom_id])
                all_scores[dicom_id] = score

            # print(all_scores.values())
            if type(method) == list:
                scores = np.mean(list(all_scores.values()), axis=0)
                for m in range(len(method)):
                    output[method[m]] = scores[m]
                    if self.verbose:
                        print("Calculated: %s: %0.3f" % ( method[m], output[method[m]]))
            else:
                output[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    print("Calculated: %s: %0.3f" % ( method, output[method]))
        self.scores = output
        return output

def main(args):
    # Call coco eval
    evaluator = GazeTranscript(ground_truth_filenames=args.references,
                             prediction_filename=args.submission,
                             tious=args.tious,
                             max_proposals=args.max_proposals_per_video,
                             verbose=args.verbose)
    evaluator.evaluate_transcript()
    # evaluator.scores['tiou'] = args.tious
    return evaluator.scores


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str,  default='/home/ptthang/gaze_sample/runs/sample-label-smoothing-pe2d-023_02_15-16_36_10-resume-2023_02_18-13_02_27/val_result.json',
                        help='sample submission file for ActivityNet Captions Challenge.')
    parser.add_argument('-r', '--references', type=str, nargs='+', default='/home/ptthang/gaze_sample/runs/sample-label-smoothing-pe2d-023_02_15-16_36_10-resume-2023_02_18-13_02_27/val_result_gt.json',
                        help='reference files with ground truth captions to compare results against. delimited (,) str')
    parser.add_argument('--tious', type=float, nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                        help='Choose the tIoUs to average over.')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                        help='maximum propoasls per video.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print intermediate steps.')
    args = parser.parse_args()

    score = main(args)
    print('json: {} \n args: {} \n score: {}'.format(args.submission,args,score))
    avg_eval_score = {key: np.array(value).mean() for key, value in score.items()}
    print('avg:\n{}'.format(avg_eval_score))
