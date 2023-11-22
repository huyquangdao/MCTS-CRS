import os
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.base import BaseTorchDataset
from config.config import IGNORE_INDEX


class UnimindTorchDataset(BaseTorchDataset):

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input instances for unimind model
        @param instances: a set of input instances.
        @param convert_example_to_feature: a dictionary which contains key and values where values are functions
        @return: a set of processed input instances.
        """
        if isinstance(convert_example_to_feature, dict):
            processed_instances = []
            # loop overall instances.
            for instance in instances:
                # loop overall functions.
                for key, func in convert_example_to_feature.items():
                    input_ids, label = func(self.tokenizer, instance, self.max_sequence_length, self.max_target_length,
                                            is_test=self.is_test, is_gen=self.is_gen)

                    new_instance = {
                        "input_ids": input_ids,
                        "label": label
                    }
                    processed_instances.append(new_instance)

            return processed_instances
        else:
            return super().preprocess_data(instances, convert_example_to_feature)


class GPTTorchDataset(BaseTorchDataset):

    def collate_fn(self, batch):
        """
        method that construct tensor-kind of inputs for DialogGPT, GPT2 models.
        @param batch: the input batch
        @return: a dictionary of torch tensors.
        """
        input_features = defaultdict(list)
        labels_gen = []
        context_length_batch = []
        for instance in batch:
            # construct the input for decoder-style of pretrained language model
            # the input is the concaternation of dialogue context and response
            # the label is similar to the input but we mask all position corresponding to the dialogue context
            # the label will be shifted to the right direction in the model
            input_features['input_ids'].append(instance['input_ids'])
            context_length_batch.append(len(instance['input_ids']))
            labels_gen.append(instance['label'])

        # padding the input features
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # labels for response generation task, for computing the loss function
        labels = input_features['input_ids']
        labels = [[token_id if token_id != self.tokenizer.pad_token_id else IGNORE_INDEX for token_id in resp] for resp
                  in labels]

        labels = torch.as_tensor(labels, device=self.device)

        # labels for response generation task, for computing generation metrics.
        labels_gen = pad_sequence(
            [torch.tensor(label, dtype=torch.long) for label in labels_gen],
            batch_first=True, padding_value=IGNORE_INDEX)
        labels_gen = labels_gen.to(self.device)

        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device)

        new_batch = {
            "context": input_features,
            "labels": labels,
            "labels_gen": labels_gen,
            "context_len": context_length_batch
        }
        return new_batch


class RTCPTorchDataset(BaseTorchDataset):

    def __init__(self, tokenizer, instances, goal2id=None, topic2id=None, max_sequence_length=512, padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False):
        """
        constructor for the BaseTorchDataset Class
        @param tokenizer: an huggingface tokenizer
        @param instances: a list of instances
        @param goal2id: a dictionary which maps goal to index.
        @param max_sequence_length: the maximum length of the input sequence.
        @param padding: type of padding
        @param pad_to_multiple_of: pad to multiple instances
        @param device: device to allocate the data, eg: cpu or gpu
        @param convert_example_to_feature: a function that convert raw instances to
        corresponding inputs and labels for the model.
        @param max_target_length the maximum number of the target sequence (response generation only)
        @param is_test True if inference step False if training step
        @param is_gen True if response generation else False
        """
        self.topic2id = topic2id
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.goal2id = goal2id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.device = device
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.is_gen = is_gen
        self.instances = self.preprocess_data(instances, convert_example_to_feature)

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the RTCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        for instance in instances:
            context_ids, path_ids, label_goal, label_topic = convert_example_to_feature(self.tokenizer, instance,
                                                                                        self.max_sequence_length,
                                                                                        self.goal2id,
                                                                                        self.topic2id)
            new_instance = {
                "context_ids": context_ids,
                "path_ids": path_ids,
                "label_goal": label_goal,
                "label_topic": label_topic
            }
            processed_instances.append(new_instance)
        return processed_instances

    def collate_fn(self, batch):
        """
        method that construct tensor-kind of inputs for DialogGPT, GPT2 models.
        @param batch: the input batch
        @return: a dictionary of torch tensors.
        """
        context_input_features = defaultdict(list)
        path_input_features = defaultdict(list)
        labels_goal = []
        labels_topic = []
        for instance in batch:
            context_input_features['context_ids'].append(instance['context_ids'])
            path_input_features['path_ids'].append(instance['path_ids'])
            labels_goal.append(instance['label_goal'])
            labels_topic.append(instance['label_topic'])

        # padding the context features
        context_input_features = self.tokenizer.pad(
            context_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in context_input_features.items():
            if not isinstance(v, torch.Tensor):
                context_input_features[k] = torch.as_tensor(v, device=self.device)

        # padding the path features
        path_input_features = self.tokenizer.pad(
            path_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in path_input_features.items():
            if not isinstance(v, torch.Tensor):
                path_input_features[k] = torch.as_tensor(v, device=self.device)

        labels_goal = torch.LongTensor(labels_goal).to(self.device)
        labels_topic = torch.LongTensor(labels_topic).to(self.device)

        new_batch = {
            "context": context_input_features,
            "path": path_input_features,
            "labels_goal": labels_goal,
            "labels_topic": labels_topic
        }
        return new_batch
