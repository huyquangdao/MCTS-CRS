import torch
import torch.nn as nn


class PolicyModel(nn.Module):

    def __init__(self, plm, n_goals, hidden_size, lm_size, dropout=0.5):
        super(PolicyModel, self).__init__()

        self.plm = plm
        self.n_goals = n_goals
        self.hidden_size = hidden_size
        self.lm_size = lm_size
        self.drop_out = nn.Dropout(p=dropout)
        self.proj_layer = nn.Linear(lm_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, n_goals)

    def forward(self, inputs):
        cls_token = self.plm(**inputs)[0]
        hidden = torch.relu(self.proj_layer(cls_token))
        logits = self.out_layer(hidden)
        return logits
