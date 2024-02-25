import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any

# import bert_score
import sentence_transformers
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenize import tokenize
# from sacrebleu.metrics import BLEU, BLEUScore
from torchmetrics.functional import sacre_bleu_score
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Check whether to use
# - https://github.com/Maluuba/nlg-eval
# - https://github.com/hwanheelee1993/KPQA
def calc_metrics(predictions: List[str], gold_annotations: List[List[str]], test=False) -> Dict[str, Any]:
    """
    Calculate metrics.

    Parameters
    ----------
    predictions : list[str]
        The list of predictions
    gold_annotations : list[list[str]]
        A list with the same length as predictions.
        Each element is a list of possible target candidates for the corresponding prediction.
        All elements should have the same length.
    """
    if len(predictions) != len(gold_annotations):
        raise ValueError(f'{len(predictions)} != {len(gold_annotations)}')
    ref_count = len(gold_annotations[0])
    if any(len(refs) != ref_count for refs in gold_annotations):
        raise ValueError(f'All refs should have the same length {ref_count}!')

    acc = _calc_accuracy(predictions, gold_annotations)
    # bleu = _calc_bleu(predictions, gold_annotations)
    rouge = _calc_rouge(predictions, gold_annotations)
    meteor = _calc_meteor(predictions, gold_annotations)
    # bert_score = _calc_bertscore(predictions, gold_annotations)
    # wups = _calc_wups(predictions, gold_annotations)
    if test:
        sts = SentenceTransformerSimilarity()
        sts_score = sts.calc_st_similarity(predictions, gold_annotations)

    return {
        'plain_acc': acc,
        # **bleu,
        'ROUGE': rouge['rougeL']['f'],
        **_flatten_dict(rouge, prefix='ROUGE.'),
        'METEOR': meteor,
        'SentenceSimilarity': sts_score if test else 0.
        # 'BERTSCORE': bert_score,
        # 'WUPS': wups
    }


""" Sentence Transformer """
class SentenceTransformerSimilarity:
    def __init__(self):
        self.model = sentence_transformers.SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def _calc_similarity(self, pred, gts):
        pred_emb = self.model.encode(pred)
        gts_emb = self.model.encode(gts)
        score = sentence_transformers.util.dot_score(pred_emb, gts_emb)[0,0].cpu()
        return float(score)

    def calc_st_similarity(self, predictions, gold_annotations):
        total_score = 0.
        for pred, gts in zip(predictions, gold_annotations):
            score = self._calc_similarity(pred, gts)
            total_score += score
        return total_score / len(predictions)


""" WUPS """
# ====================================================
# @Time    : 13/9/20 4:19 PM
# @Author  : Xiao Junbin
# @Email   : junbin@comp.nus.edu.sg
# @File    : metrics.py
# ====================================================


def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0: return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0: return 0.0

    #match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1*word_sim
    return word_sim

def wups(words1, words2, alpha):
    """

    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0: continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim

def get_wups(pred, truth, alpha=0):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value

def _calc_wups(predictions, gold_annotations):
    wups = 0
    for pred, gt in zip(predictions, gold_annotations):
        wups += get_wups(pred, gt[0])
    wups /= len(predictions)
    return wups
""" WUPS """


# def _calc_bertscore(predictions, gold_annotations):
#     references = [x[0] for x in gold_annotations]
#     P, R, F1 = bert_score.score(
#         predictions, references, lang='en',
#         model_type='microsoft/deberta-xlarge-mnli',
#     )
#     return float(F1.mean())


def _calc_accuracy(predictions, gold_annotations):
    correct = 0
    for pred, possible_refs in zip(predictions, gold_annotations):
        if any(ref == pred for ref in possible_refs):
            correct += 1
    total = len(predictions)
    return correct / total


def _calc_meteor(predictions, gold_annotations):
    score = AverageMeter()
    for pred, possible_refs in zip(predictions, gold_annotations):
        pred = tokenize(pred, None)
        # https://github.com/cmu-mtlab/meteor/blob/master/src/edu/cmu/meteor/util/Normalizer.java
        possible_refs = [tokenize(x, None) for x in possible_refs]
        score.update(meteor_score(possible_refs, pred))
    return score.avg


def _calc_rouge(predictions, gold_annotations) -> Dict[str, Dict[str, float]]:
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge = defaultdict(lambda: defaultdict(AverageMeter))
    for pred, possible_refs in zip(predictions, gold_annotations):
        sample_result = {}
        for ref in possible_refs:
            single_ref_result = rouge_scorer.score(ref, pred)
            for k, scores in single_ref_result.items():
                existing_result_dict = sample_result.setdefault(k, {})
                if existing_result_dict.get('f', -1) < scores.fmeasure:
                    existing_result_dict.update(f=scores.fmeasure, p=scores.precision, r=scores.recall)
        for k, best_scores in sample_result.items():
            rouge[k]['p'].update(best_scores['p'])
            rouge[k]['r'].update(best_scores['r'])
            rouge[k]['f'].update(best_scores['f'])
    return {
        rouge_type: {
            measure: score.avg
            for measure, score in results.items()
        } for rouge_type, results in rouge.items()
    }


def _calc_bleu(predictions, gold_annotations) -> Dict[str, float]:
    return {
        'BLEU': sacre_bleu_score(predictions, gold_annotations, n_gram=1)
    }
    # refs_transposed = [
    #     [refs[i] for refs in gold_annotations]
    #     for i in range(len(gold_annotations[0]))
    # ]
    # bleu: BLEUScore = BLEU().corpus_score(predictions, refs_transposed)
    # return {
    #     'BLEU': bleu.score,
    #     'BLEU.bp': bleu.bp,
    #     'BLEU.ratio': bleu.ratio,
    #     'BLEU.hyp_len': float(bleu.sys_len),
    #     'BLEU.ref_len': float(bleu.ref_len),
    # }


def _flatten_dict(d, prefix=''):
    result = {}
    for k, v in d.items():
        my_key = prefix + k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=my_key + '.'))
        else:
            result[my_key] = v
    return result


def main():
    parser = ArgumentParser('Eval output file')
    parser.add_argument('--gold_answers', type=str, required=True,
                        help='Path to answers.json, containing mapping from sample_id to answer')
    parser.add_argument('eval_file', type=str,
                        help='JSON File to evaluate. Should contain mapping from sample_id '
                             'to hypothesis or array of hypotheses')
    args = parser.parse_args()

    gold_answers = json.loads(Path(args.gold_answers).read_text())
    hypotheses = json.loads(Path(args.eval_file).read_text())
    if isinstance(next(iter(hypotheses.values())), list):
        hypotheses = {k: v[0] for k, v in hypotheses.items()}
    assert len(hypotheses.keys() - gold_answers.keys()) == 0, 'No gold answer for some hypotheses'

    gold_and_hypo = [(gold_answers[k], hypotheses[k]) for k in hypotheses.keys()]
    hypo_list = [h for g, h in gold_and_hypo]
    gold_list = [[g] for g, h in gold_and_hypo]
    metrics = calc_metrics(hypo_list, gold_list)

    pprint(metrics)


if __name__ == '__main__':
    # main()

    # debug
    st = SentenceTransformerSimilarity()
    score = st._calc_similarity('inside the drawer', ['inside the drawer'])
    print(score)  # 1.0

    score = st._calc_similarity('inside the drawer', ['on the table'])
    print(score)  # 0.49

    score = st._calc_similarity('inside the drawer', ['in the drawer'])
    print(score)  # 0.93

    # mean_score = st.calc_st_similarity(
    #     ['floor', '3'],
    #     [['on the ground'], ['two']]
    # )
    # print(mean_score)
