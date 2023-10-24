import math
import torch
from sklearn.metrics import precision_recall_fscore_support


class PolicyEvaluator:
    def __init__(self, device=torch.device('cpu')):
        self.metric = {}
        self.device = device
        self.reset_metric()

    def evaluate(self, logits, labels):
        self.metric['acc'] += self.compute_acc(logits, labels)
        self.metric['count'] += logits.shape[0]

    def compute_acc(self, logits, labels):
        return int(torch.eq(logits.argmax(-1), labels).sum())

    def reset_metric(self):
        self.metric['acc'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            report[k] = torch.tensor(v / self.metric['count'], device=self.device)[None]
        return report

    @staticmethod
    def compute_categorical_acc(preds, labels):
        count = 0
        for (pred, label) in list(zip(preds, labels)):
            if pred.lower().strip() == label.lower().strip():
                count += 1
        return count / len(labels)

    @staticmethod
    def compute_precision_recall_f1_metrics(preds, labels, average='macro'):
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
        return p, r, f1
