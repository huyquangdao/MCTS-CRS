import os

import gym
import torch

from dyna_gym.default_policy.default_policy import DefaultPolicy
from dyna_gym.envs.utils import simulate_conversation
from dataset.data_utils import convert_example_to_feature_for_goal_prediction


class OfflinePolicy(DefaultPolicy):

    def __init__(
            self,
            env: gym.Env,
            horizon: int,
            tokenizer,
            policy_model,
            max_sequence_length = 512,
            padding='max_length',
            pad_to_multiple_of=True,
            goal2id=None,
            generation_args: dict = {},
    ):
        super().__init__(env, horizon)
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_sequence_length = max_sequence_length
        self.generate_args = generation_args
        self.goal2id = goal2id

    def get_top_k_tokens(self, state, k=1):

        # convert state to input features
        input_ids, _ = convert_example_to_feature_for_goal_prediction(self.tokenizer, state, self.max_sequence_length,
                                                                      self.goal2id)
        # padding the input features
        input_features = self.tokenizer.pad(
            [input_ids], padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v)

        # compute policy with offline policy model.
        logits = self.policy_model(input_features)

        # compute top-k predictions
        topk_probs, topk_indices = torch.topk(logits, k, sorted=True)
        topk_probs = topk_probs.tolist()
        topk_indices = topk_indices.tolist()

        return topk_indices, topk_probs

    def get_predicted_sequence(self, state, horizon: int = None):
        pass
