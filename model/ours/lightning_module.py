import json
import copy
import random

import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import OneCycleLR

from eval import calc_metrics
from eval_nlq import ReferringRecall


class TestLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = instantiate(config.model, max_v_len=config.dataset.max_v_len)

    def test_step(self, batch, batch_idx):
        nlq_results, answer_tokens = self.model.generate(**batch)
        pred_answer = self.tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
        return {
            'question': batch['q_text'],
            'video_id': batch['video_id'],
            'answer': batch['a_text'] if 'a_text' in batch else '',
            'pred_answer': pred_answer,
            'nlq_results': nlq_results,
            'query_id': batch['query_id'],
            'sample_ratio': batch['sample_ratio'],
            'task': batch['task'],
            'clip_uid': batch['video_id']
        }

    def test_epoch_end(self, outputs):
        self.save_nlq_results(outputs)

    def save_nlq_results(self, preds):
        # aggregate preds
        pred_dict = {
            "version": "1.0",
            "challenge": "ego4d_nlq_challenge",
            "results": []
        }
        for batch_pred in preds:
            for i in range(len(batch_pred['video_id'])):
                qid = batch_pred['query_id'][i]
                annotation_uid, query_idx = qid.split('_')
                query_idx = int(query_idx)
                clip_uid = batch_pred['clip_uid'][i]
                sample_ratio = batch_pred['sample_ratio'][i]
                predicted_times = [
                    [segment[0] / sample_ratio, segment[1] / sample_ratio] 
                    for segment in batch_pred['nlq_results'][i]['segments'].cpu().detach().tolist()
                ]
                
                pred_dict['results'].append({
                    'clip_uid': clip_uid,
                    'annotation_uid': annotation_uid,
                    'query_idx': query_idx,
                    'predicted_times': predicted_times
                })

        with open('nlq_eval_results/nlq_v2.json', 'w') as f:
            json.dump(pred_dict, f)


class LightningModule(pl.LightningModule):
    def __init__(self, config, total_steps):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = instantiate(config.model, max_v_len=config.dataset.max_v_len)
        self.nlq_evaluator = ReferringRecall(
            dataset="ego4d",
            gt_file=config.dataset.nlq_val_anno
        )
        self._log_indices = {}
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        total_loss, ce_loss, time_loss = self.model(**batch)
        self.log('total_loss', total_loss, rank_zero_only=True)
        self.log('ce_loss', ce_loss, rank_zero_only=True)
        self.log('time_loss', time_loss, rank_zero_only=True)
        return {
            'loss': total_loss,
        }
    
    def validation_step(self, batch, batch_idx):
        nlq_results, answer_tokens = self.model.generate(**batch)
        pred_answer = self.tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
        return {
            'question': batch['q_text'],
            'video_id': batch['video_id'],
            'answer': batch['a_text'] if 'a_text' in batch else '',
            'pred_answer': pred_answer,
            'nlq_results': nlq_results,
            'query_id': batch['query_id'],
            'sample_ratio': batch['sample_ratio'],
            'task': batch['task']
        }
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def _log_some_outputs(self, outputs, name):
        num_val_steps_to_log, num_samples_per_batch_to_log = 5, 3  # Could be configurable via cfg
        steps_to_log_indices = random.sample(range(len(outputs)), k=min(len(outputs), num_val_steps_to_log))
        self._log_indices[name] = {
            'steps': steps_to_log_indices, 
            'samples': [
                random.sample(
                    range(len(outputs[step]['answer'])),
                    k=min(len(outputs[step]['answer']), 
                    num_samples_per_batch_to_log))
                for step in steps_to_log_indices
            ]
        }
        for i, step in enumerate(steps_to_log_indices):
            indices = self._log_indices[name]['samples'][i]
            for b in indices:
                sample = (
                    f'Video: "{outputs[step]["video_id"][b]}". \n'
                    f'Question: "{outputs[step]["question"][b]}". \n'
                    f'Target: "{outputs[step]["answer"][b]}". \n'
                    f'Output: "{outputs[step]["pred_answer"][b]}"'
                )
                self.logger.experiment.add_text(f'{name} {str(i * len(indices) + b)}', sample,
                                                global_step=self.global_step)

    def aggregate_metrics(self, outputs, prefix):
        # evaluate CloseQA
        all_hypos = []
        all_targets = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] == 'CloseQA':
                    all_hypos.append(output['pred_answer'][i])
                    all_targets.append(output['answer'][i])
        if len(all_hypos) > 0:
            num_correct = 0
            for hypo, target in zip(all_hypos, all_targets):
                if hypo == target:
                    num_correct += 1
            acc = num_correct / len(all_targets) * 100
            metrics = {f'{prefix}_close_acc': acc}
        else:
            metrics = {}

        # evaluate OpenQA
        all_hypos = []
        all_targets = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] == 'OpenQA':
                    all_hypos.append(output['pred_answer'][i])
                    all_targets.append(output['answer'][i])
        if len(all_hypos) > 0:
            open_qa_metrics = calc_metrics(all_hypos, [[x] for x in all_targets], test=prefix=='test')
            for k, v in open_qa_metrics.items():
                metrics[f'{prefix}_{k}'] = v

        # evalute NLQ
        nlq_preds = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] != 'NLQ':
                    continue
                qid = output['query_id'][i]
                temp_list = qid.split("_")
                sample_ratio = output['sample_ratio'][i]
                new_prediction = [
                    [   segment[0] / sample_ratio,
                        segment[1] / sample_ratio,
                        score  ] 
                    for segment, score in zip(
                        output['nlq_results'][i]['segments'].cpu().detach().tolist(),
                        output['nlq_results'][i]['scores'].cpu().detach().tolist(),
                )]
                nlq_preds.append({
                    'query_idx': int(temp_list[1]),
                    'annotation_uid': temp_list[0],
                    'predicted_times': new_prediction,
                    'clip_uid': output['video_id'][i]
                })
        if len(nlq_preds) > 0:
            performance, score_str = self.nlq_evaluator.evaluate(nlq_preds, verbose=False)
            metrics[f'{prefix}_R1_03'] = performance[0, 0] * 100
            metrics[f'{prefix}_R5_03'] = performance[0, 1] * 100
            metrics[f'{prefix}_R1_05'] = performance[1, 0] * 100
            metrics[f'{prefix}_R5_05'] = performance[1, 1] * 100
            metrics[f'{prefix}_Mean_R1'] = (performance[0, 0] + performance[1, 0]) * 100 / 2

        # # save predictions
        # results = []
        # for output in outputs:
        #     for i in range(len(output['video_id'])):
        #         results.append({
        #             'query_id': output['query_id'][i],
        #             'pred_answer': output['pred_answer'][i],
        #             'gt_answer': output['answer'][i],
        #             'pred_window': (output['nlq_results'][i]['segments'].cpu().detach() / output['sample_ratio'][i]).tolist(),
        #             'gt_window': self.nlq_evaluator.gt_dict[(output['video_id'][i], output['query_id'][i].split('_')[0])]["language_queries"][int(output['query_id'][i].split('_')[1])]
        #         })
        # with open('analysis/VLG_OpenQA.json', 'w') as f:
        #     json.dump(results, f)

        return metrics

    # def training_epoch_end(self, outputs):
        # self._log_some_outputs(outputs, 'train')
        # metrics = self.aggregate_metrics(outputs, prefix='train')
        # self.log_dict(metrics, sync_dist=True)

    def validation_epoch_end(self, outputs):
        def _mean(key):
            return torch.stack([data[key] for data in outputs]).mean()

        # self._log_some_outputs(outputs, 'val')
        metrics = self.aggregate_metrics(outputs, prefix='val')
        metrics.update({
            f'val_{name}': _mean(name) for name in outputs[0].keys() if 'loss' in name
        })
        self.log_dict(metrics, sync_dist=True)

    def test_epoch_end(self, outputs):
        # self._log_some_outputs(outputs, 'test')
        metrics = self.aggregate_metrics(outputs, prefix='test')
        self.log_dict(metrics, sync_dist=True)
        if self.config.trainer.save_nlq_results is not None:
            src = 'data/joint/annotations.QaEgo4D_test_close.json'
            dst = self.config.trainer.save_nlq_results
            self.save_nlq_results(src, dst, outputs)

    def save_nlq_results(self, src, dst, preds):
        # aggregate preds
        pred_dict = {}
        for batch_pred in preds:
            for i in range(len(batch_pred['video_id'])):
                qid = batch_pred['query_id'][i]
                sample_ratio = batch_pred['sample_ratio'][i]
                pred_start = batch_pred['nlq_results'][i]['segments'][0].cpu().detach().tolist()[0] / sample_ratio
                pred_end = batch_pred['nlq_results'][i]['segments'][0].cpu().detach().tolist()[1] / sample_ratio
                assert qid not in pred_dict
                pred_dict[qid] = {
                    'pred_start_sec': pred_start,
                    'pred_end_sec': pred_end
                }

        save_results = []
        for src_data in json.load(open(src)):
            pred_data = pred_dict[src_data['sample_id']]
            save_data = copy.deepcopy(src_data)
            save_data['moment_start_frame'] = pred_data['pred_start_sec'] * 30
            save_data['moment_end_frame'] = pred_data['pred_end_sec'] * 30
            save_results.append(save_data)
        with open(dst, 'w') as f:
            json.dump(save_results, f)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.config.optim.optimizer,
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.optim.optimizer.lr
        )
        if self.config.optim.lr_scheduler:
            lr_scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.optim.optimizer.lr,
                total_steps=self.total_steps,
                anneal_strategy='linear'
            )
            return {
                'optimizer': optimizer, 
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer