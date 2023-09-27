import math

import torch


class PolicyEvaluator:
    def __init__(self, device=torch.device('cpu')):
        self.metric = {}
        self.device= device
        self.reset_metric()

    def evaluate(self, logits, labels):
        self.metric['acc'] += self.compute_acc(logits, labels)
        self.metric['count'] += logits.shape[0]

    def compute_acc(self, logits, labels):
        return int(torch.eq(logits.argmax(), labels).sum())

    def reset_metric(self):
        self.metric['acc'] = 0
        self.metric['count'] = 0

    def report(self):
        report = {}
        for k, v in self.metric.items():
            report[k] = torch.tensor(v, device=self.device)[None]
        return report
