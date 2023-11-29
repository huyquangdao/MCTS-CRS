import os
from collections import defaultdict

import gym
import torch

from dyna_gym.default_policy.default_policy import DefaultPolicy
# from dyna_gym.envs.utils import simulate_conversation
# from dataset.data_utils import convert_example_to_feature_for_goal_prediction
from baselines.rtcp.utils import convert_example_to_feature_for_rtcp_goal_topic_prediction


class RTCPOfflinePolicy(DefaultPolicy):

    def __init__(
            self,
            env: gym.Env,
            horizon: int,
            generation_model,
            generation_tokenizer,
            know_generation_model,
            know_tokenizer,
            policy_tokenizer,
            policy_model,
            max_sequence_length=512,
            max_gen_length=50,
            padding='max_length',
            pad_to_multiple_of=True,
            goal2id=None,
            device=None,
            terminated_act=None,
            generation_args: dict = {},
    ):
        super().__init__(env, horizon)
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.know_generation_model = know_generation_model
        self.know_tokenizer = know_tokenizer
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_sequence_length = max_sequence_length
        self.max_gen_length = max_gen_length
        self.generate_args = generation_args
        self.goal2id = goal2id
        self.device = device
        self.terminated_act = terminated_act

    def get_top_k_tokens(self, state, top_k=3):
        """
        method that get top-k predictions based on some certain policy
        currently implememented based on the offline planning policy
        @param state: the current state of the env
        @param top_k: number of predictions.
        @return: top_k indices and top_k probabilities
        """
        context_input_features = defaultdict(list)
        path_input_features = defaultdict(list)

        # convert state to input features
        context_ids, path_ids, _, _ = convert_example_to_feature_for_rtcp_goal_topic_prediction(
            tokenizer=self.policy_tokenizer,
            instance=state,
            max_sequence_length=self.max_sequence_length
        )

        context_input_features['input_ids'] = context_ids
        path_input_features['input_ids'] = path_ids

        # padding the context features
        context_input_features = self.policy_tokenizer.pad(
            context_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in context_input_features.items():
            if not isinstance(v, torch.Tensor):
                context_input_features[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)

        # padding the path features
        path_input_features = self.policy_tokenizer.pad(
            path_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in path_input_features.items():
            if not isinstance(v, torch.Tensor):
                path_input_features[k] = torch.as_tensor(v, device=self.device).unsqueeze(0)

        # label goal, topic. Just using for computational convenience.
        labels_goal = torch.LongTensor([0]).to(self.device)
        labels_topic = torch.LongTensor([0]).to(self.device)

        batch = {
            "context": context_input_features,
            "path": path_input_features,
            "labels_goal": labels_goal,
            "labels_topic": labels_topic
        }
        # predict action
        outputs = self.policy_model(batch)
        goal_logits = outputs['goal_logits']
        topic_logits = outputs['topic_logits']

        # convert logits to probabilities
        all_goal_probs = torch.softmax(goal_logits, dim=-1)
        all_topic_probs = torch.softmax(topic_logits, dim=-1)

        # combining goal and topic probabilities
        all_probs = all_goal_probs.unsqueeze(-1).repeat(1, 1, all_topic_probs.size(-1)) * all_topic_probs.unsqueeze(
            1).repeat(1, all_goal_probs.size(-1), 1)
        all_probs = all_probs.view(all_probs.size(0), -1)

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
        # generated_conversation = simulate_conversation(generation_model=self.generation_model,
        #                                                generation_tokenizer=self.generation_tokenizer,
        #                                                know_generation_model=self.know_generation_model,
        #                                                know_tokenizer=self.know_tokenizer,
        #                                                policy_model=self.policy_model,
        #                                                policy_tokenizer=self.policy_tokenizer,
        #                                                state=state,
        #                                                horizon=horizon,
        #                                                max_sequence_length=self.max_sequence_length,
        #                                                max_gen_length=self.max_gen_length,
        #                                                padding=self.padding,
        #                                                pad_to_multiple_of=self.pad_to_multiple_of,
        #                                                goal2id=self.goal2id,
        #                                                terminated_action=self.terminated_act,
        #                                                device=self.device)
        return None
