import os
from collections import defaultdict

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
            generation_model,
            generation_tokenizer,
            policy_tokenizer,
            policy_model,
            max_sequence_length=512,
            max_gen_length=50,
            padding='max_length',
            pad_to_multiple_of=True,
            goal2id=None,
            device=None,
            generation_args: dict = {},
    ):
        super().__init__(env, horizon)
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_sequence_length = max_sequence_length
        self.max_gen_length = max_gen_length
        self.generate_args = generation_args
        self.goal2id = goal2id
        self.device = device

    def get_top_k_tokens(self, state, top_k=3):
        """
        method that get top-k predictions based on some certain policy
        currently implememented based on the offline planning policy
        @param state: the current state of the env
        @param top_k: number of predictions.
        @return: top_k indices and top_k probabilities
        """
        input_features = defaultdict(list)
        # convert state to input features
        input_ids, _ = convert_example_to_feature_for_goal_prediction(self.policy_tokenizer, state,
                                                                      self.max_sequence_length,
                                                                      self.goal2id)

        input_features['input_ids'] = input_ids

        # padding the input features
        input_features = self.policy_tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)

        # compute policy with offline policy model.
        logits = self.policy_model(input_features)

        # convert logits to probabilities
        all_probs = torch.softmax(logits, dim=-1)
        # compute top-k predictions
        topk_probs, topk_indices = torch.topk(all_probs, top_k, sorted=True)
        topk_probs = topk_probs.tolist()
        topk_indices = topk_indices.tolist()

        return topk_indices[0], topk_probs[0]

    def get_predicted_sequence(self, state, horizon: int = 5):
        """
        Default policy to simulate an entire conversation starting from the input state
        @param state: the current state of the conversation
        @param horizon: the maximum number of conversation turns
        @return: the last system response
        """
        last_generated_resp = simulate_conversation(generation_model=self.generation_model,
                                                    generation_tokenizer=self.generation_tokenizer,
                                                    policy_model=self.policy_model,
                                                    policy_tokenizer=self.policy_tokenizer,
                                                    state=state,
                                                    horizon=horizon,
                                                    max_sequence_length=self.max_sequence_length,
                                                    max_gen_length=self.max_gen_length,
                                                    padding=self.padding,
                                                    pad_to_multiple_of=self.pad_to_multiple_of,
                                                    goal2id=self.goal2id,
                                                    device=self.device)
        return last_generated_resp
