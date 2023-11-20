import os
from collections import defaultdict

import torch
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

        input_features = defaultdict(list)
        # labels = []
        context_length_batch = []
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            context_length_batch.append(len(instance['input_ids']))
            # labels.append(instance['label'])

        # padding the input features
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # labels for response generation task
        labels = input_features['input_ids']
        labels = [[token_id if token_id != self.tokenizer.pad_token_id else IGNORE_INDEX for token_id in resp] for resp
                  in labels]
        labels = torch.as_tensor(labels, device=self.device)

        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device)

        new_batch = {
            "context": input_features,
            "labels": labels,
            "context_len": context_length_batch
        }
        return new_batch
