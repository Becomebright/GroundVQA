#! /usr/bin/env python
"""
Script to evaluate performance of any model for Ego4d Episodic Memory.

Natural Language Queries (NLQ)
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import json

import numpy as np
import torch
import terminaltables



def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    )
    candidates = np.stack(
        [
            np.repeat(s_times[:, None], repeats=num_units, axis=1),
            np.repeat(e_times[None, :], repeats=num_units, axis=0),
        ],
        axis=2,
    ).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(
        num_units, num_units
    )
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = (
        np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    )
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


def display_results(results, mIoU, thresholds, topK, title=None):
    display_data = [
        [f"Rank@{ii}\nmIoU@{jj}" for ii in topK for jj in thresholds] + ["mIoU"]
    ]
    results *= 100
    mIoU *= 100
    display_data.append(
        [
            f"{results[jj][ii]:.02f}"
            for ii in range(len(topK))
            for jj in range(len(thresholds))
        ]
        + [f"{mIoU:.02f}"]
    )
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(thresholds) * len(topK)):
        table.justify_columns[ii] = "center"
    return table.table


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def evaluate_nlq_performance(
    predictions, ground_truth, thresholds, topK, per_instance=False
):
    results = [[[] for _ in topK] for _ in thresholds]
    average_IoU = []
    for pred, gt in zip(predictions, ground_truth):
        # Compute overlap and recalls.
        overlap = compute_IoU(pred, gt)
        average_IoU.append(np.mean(np.sort(overlap[0])[-3:]))
        for tt, threshold in enumerate(thresholds):
            for rr, KK in enumerate(topK):
                results[tt][rr].append((overlap > threshold)[:KK].any())

    mean_results = np.array(results).mean(axis=-1)
    mIoU = np.mean(average_IoU)
    if per_instance:
        per_instance_results = {
            "overlap": overlap,
            "average_IoU": average_IoU,
            "results": results,
        }
        return mean_results, mIoU, per_instance_results
    else:
        return mean_results, mIoU


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]
    

class ReferringRecall(object):
    thresholds = np.array([0.3, 0.5])
    topK = np.array([1, 5])
    def __init__(
            self,
            dataset="ego4d",
            gt_file="/remote-home/share/Ego4D_dsz/v1/annotations/nlq_val.json"
    ):
        self.dataset = dataset
        self.gt_file = gt_file
        print(self.gt_file)
        if self.dataset == "ego4d":
            with open(self.gt_file) as file_id:
                self.gt_dict, self.num_gt_queries = self.load_gt_from_json(json.load(file_id))
        else:
            self.gt_dict = {}
            for d in load_jsonl(self.gt_file):
                # print(d)
                self.gt_dict[d['query_id']] = d["timestamps"]
            self.num_gt_queries = len(self.gt_dict)

    def load_gt_from_json(self, ground_truth):
        gt_dict = {}
        num_gt_queries = 0

        for video_datum in ground_truth["videos"]:
            for clip_datum in video_datum["clips"]:
                clip_uid = clip_datum["clip_uid"]
                for ann_datum in clip_datum["annotations"]:
                    key = (clip_uid, ann_datum["annotation_uid"])
                    gt_dict[key] = ann_datum
                    num_gt_queries += len(ann_datum["language_queries"])

        return gt_dict, num_gt_queries

    def compute_IoU(self, pred, gt):
        """Compute the IoU given predicted and ground truth windows."""
        assert isinstance(pred, list) and isinstance(gt, list)
        if len(pred) == 0:  # FIXME: I don't know why, maybe coincidence, that the PtTransformerRegHead produces all 0 offsets at the start of training
            return [0]
        pred_is_list = isinstance(pred[0], list)
        gt_is_list = isinstance(gt[0], list)
        if not pred_is_list:
            pred = [pred]
        if not gt_is_list:
            gt = [gt]
        pred, gt = np.array(pred), np.array(gt)
        inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
        inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
        inter = np.maximum(0.0, inter_right - inter_left)
        union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
        union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
        union = np.maximum(0.0, union_right - union_left)
        overlap = 1.0 * inter / union
        if not gt_is_list:
            overlap = overlap[:, 0]
        if not pred_is_list:
            overlap = overlap[0]
        return overlap

    def display_results_anet(self, results, title=None):
        display_data = [
            [f"Rank@{ii}\nmIoU@{jj:.1f}" for ii in self.topK for jj in self.thresholds]
        ]
        results *= 100
        display_data.append(
            [
                f"{results[ii][jj]:.02f}"
                for ii in range(len(self.topK))
                for jj in range(len(self.thresholds))
            ]
        )
        table = terminaltables.AsciiTable(display_data, title)
        for ii in range(len(self.thresholds) * len(self.topK)):
            table.justify_columns[ii] = "center"
        return table.table

    def display_results(self, results, title=None):
        display_data = [
            [f"Rank@{ii}\nmIoU@{jj}" for ii in self.topK for jj in self.thresholds]
        ]
        results *= 100

        display_data.append(
            [
                f"{results[jj][ii]:.02f}"
                for ii in range(len(self.topK))
                for jj in range(len(self.thresholds))
            ]
        )
        table = terminaltables.AsciiTable(display_data, title)
        for ii in range(len(self.thresholds) * len(self.topK)):
            table.justify_columns[ii] = "center"
        return table.table

    def evaluate(self, predictions, verbose=True):
        """Evalutes the performances."""

        results = [[[] for _ in self.topK] for _ in self.thresholds]
        average_IoU = []
        num_instances = 0

        for pred_datum in predictions:
            key = (pred_datum["clip_uid"], pred_datum["annotation_uid"])
            assert key in self.gt_dict, f"{key} Instance not present!"
            query_id = pred_datum["query_idx"]
            gt_datum = self.gt_dict[key]
            gt_query_datum = gt_datum["language_queries"][query_id]

            # Compute overlap and recalls.
            overlap = self.compute_IoU(
                pred_datum["predicted_times"],
                [[gt_query_datum["clip_start_sec"], gt_query_datum["clip_end_sec"]]],
            )
            average_IoU.append(overlap[0])

            for tt, threshold in enumerate(self.thresholds):
                for rr, KK in enumerate(self.topK):
                    results[tt][rr].append((overlap > threshold)[:KK].any())
            num_instances += 1

        mean_results = np.array(results).mean(axis=-1)

        score_str = None
        if verbose:
            print(f"Evaluated: {num_instances} / {self.num_gt_queries} instances")
            score_str = self.display_results(mean_results)
            print(score_str, flush=True)

        return mean_results, score_str

    def _iou(self, candidates, gt):
        start, end = candidates[:, 0].float(), candidates[:, 1].float()
        s, e = gt[0].float(), gt[1].float()
        inter = end.min(e) - start.max(s)
        union = end.max(e) - start.min(s)
        return inter.clamp(min=0) / union

    def evaluate_anet(
            self, submission, verbose=True):

        iou_metrics = torch.tensor(self.thresholds)
        num_iou_metrics = len(iou_metrics)

        recall_metrics = torch.tensor(self.topK)
        max_recall = recall_metrics.max()
        num_recall_metrics = len(recall_metrics)
        recall_x_iou = torch.zeros((num_recall_metrics, len(iou_metrics)))

        for k in submission:
            # print(k)
            gt_grounding = torch.tensor(self.gt_dict[k['query_id']])
            pred_moments = torch.tensor(k["predicted_times"][:max_recall])
            mious = self._iou(pred_moments, gt_grounding)
            mious_len = len(mious)
            bools = mious[:, None].expand(mious_len, num_iou_metrics) > iou_metrics
            for i, r in enumerate(recall_metrics):
                recall_x_iou[i] += bools[:r].any(dim=0)

        recall_x_iou /= len(submission)

        if verbose:
            print(f"Evaluated: {len(submission)} / {self.num_gt_queries} instances")
            score_str = self.display_results_anet(recall_x_iou)
            print(score_str, flush=True)

        return recall_x_iou


def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.
    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.
    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU