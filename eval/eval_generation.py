import re
from collections import Counter

import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu


# from nltk.translate.meteor_score import single_meteor_score


class GenerationEvaluator:
    def __init__(self, tokenizer, log_file_path):
        self.tokenizer = tokenizer

        self.reset_metric()
        if log_file_path:
            self.log_file = open(log_file_path, 'w', buffering=1)
            self.log_cnt = 0

        self.all_decoded_preds = []
        self.all_decoded_labels = []

    def evaluate(self, preds, labels, log=False):
        """
        methods that compute a set of evaluation metrics given the generated text strings and groundtruth labels
        @param preds: a list of generated text strings
        @param labels: a list of groundtruth labels
        @param log: log to file
        @return: None
        """
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [decoded_pred.replace('</s>', '').replace('<s>', '') for decoded_pred in
                         decoded_preds]
        decoded_preds = [pred.strip() for pred in decoded_preds]

        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [decoded_label.replace('</s>', '').replace('<s>', '') for decoded_label in
                          decoded_labels]
        decoded_labels = [label.strip() for label in decoded_labels]

        if log and hasattr(self, 'log_file'):
            for pred, label in zip(decoded_preds, decoded_labels):
                self.log_file.write(json.dumps({
                    'pred': pred,
                    'label': label
                }, ensure_ascii=False) + '\n')

        self.collect_ngram(decoded_preds)
        self.compute_bleu(decoded_preds, decoded_labels)
        self.compute_word_f1(decoded_preds, decoded_labels)
        # self.compute_meteor_score(decoded_preds, decoded_labels)
        self.sent_cnt += len([pred for pred in decoded_preds if len(pred) > 0])
        self.all_decoded_preds.extend(decoded_preds)
        self.all_decoded_labels.extend(decoded_labels)

    def collect_ngram(self, strs):
        """
        method that compute n-grams (n=1,4) of a set of strings.
        @param strs: a list of text strings
        @return: None
        """
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        """
        method that compute bleu score between generated texts and labels
        @param preds: a list of generated text strings
        @param labels: a list of groundtruth labels
        @return: None
        """
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)

    def compute_word_f1(self, preds, labels):
        """
        method that compute word-level f1 score between generated text string and labels
        @param preds: a list of generated text strings
        @param labels: a list of groundtruth labels
        @return: none
        """
        golden_char_total = 0.0
        pred_char_total = 0.0
        hit_char_total = 0.0
        for response, golden_response in zip(preds, labels):
            common = Counter(response) & Counter(golden_response)
            hit_char_total += sum(common.values())
            golden_char_total += len(golden_response)
            pred_char_total += len(response)
        p = hit_char_total / pred_char_total if pred_char_total > 0 else 0
        r = hit_char_total / golden_char_total if golden_char_total > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        self.metric["f1"] = f1

    def compute_bert_score(self, preds, labels):
        """
        method that compute the bert score between the generated text strings and groundtruth labels.
        @param preds: a list of generated text strings
        @param labels: a list of groundtruth labels
        @return: None
        """

    def compute_meteor_score(self, preds, labels):
        """
        methods can compute the meteor score between generated texts and groundtruth labels.
        @param preds: list of generated texts trings
        @param labels: list of groundtruth labels
        @return: None
        """
        for pred, label in zip(preds, labels):
            # tokenizer the predicted text and groundtruth label
            pred, label = pred.split(), label.split()
            self.metric["meteor"] += single_meteor_score(pred, label)

    def report(self):
        """
        methods that compute the final values of metrics
        @return: None
        """
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k:
                    v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report, self.all_decoded_preds, self.all_decoded_labels

    def reset_metric(self):
        """
        method that inits default values for the metrics
        @return: None
        """
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            "f1": 0,
            "meteor": 0,
            'dist@1': set(),
            'dist@2': set(),
            'dist@3': set(),
            'dist@4': set(),
        }
        self.sent_cnt = 0
        self.all_decoded_labels = []
        self.all_decoded_preds = []
