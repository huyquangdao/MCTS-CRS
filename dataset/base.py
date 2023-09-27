from __future__ import print_function
from collections import defaultdict

import torch

from torch.utils.data import Dataset as TorchDataset
from dataset.data_utils import convert_example_to_feature


class BaseTorchDataset(TorchDataset):

    def __init__(self, tokenizer, instances, goal2id, max_sequence_length=512, padding_ids=0, pad_to_multiple_of=True):
        super(BaseTorchDataset, self).__init__()
        self.instances = instances
        self.max_sequence_length = max_sequence_length
        self.padding_ids = padding_ids
        self.tokenizer = tokenizer
        self.goal2id = goal2id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]
        return instance

    def collate_fn(self, batch):
        input_features = defaultdict(list)
        labels = []
        for instance in batch:
            input_ids = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length)
            input_features['input_ids'].append(input_ids)
            labels.append(self.goal2id[instance['goal']])

        input_features = self.tokenizer.pad(
            input_features, padding=self.padding_ids, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        labels = torch.Tensor(labels)
        batch['context'] = input_features
        batch['labels'] = labels
        return input_features


class Dataset:

    def __init__(self, train_data_path, dev_data_path, test_data_path, save_train_convs=True):
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_data_path = test_data_path

        self.topics = []
        self.goals = []
        self.save_train_convs = save_train_convs
        self.train_convs = None

        self.train_instances = self.pipeline(self.train_data_path)
        self.dev_instances = self.pipeline(self.dev_data_path)
        self.test_instances = self.pipeline(self.test_data_path)

        self.goals = list(set(self.goals))
        self.topics = list(set(self.topics))

    def return_infor(self):
        """function that returns information about the dataset

        Returns:
            _type_: dictionary
        """
        infor_dict = {
            "num_topics": len(self.topics),
            "num_goals": len(self.goals),
            "train_instances": len(self.train_instances),
            "dev_instances": len(self.dev_instances),
            "test_instances": len(self.test_instances)

        }
        return infor_dict

    def read_data(self, data_path):
        """function that reads the data from input file

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def process_data(self, data):
        """Function that process the data given the read data.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def repurpose_dataset(self, data):
        """Function that convert the original dataset from goal-driven setting to target-driven setting.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()

    def pipeline(self, data_path):
        """method that employs that data pipeline including read_data, repurpose_data and progress_data
        """
        data = self.read_data(data_path=data_path)
        data = self.repurpose_dataset(data)
        if self.save_train_convs and 'train' in data_path:
            self.train_convs = data
        data = self.process_data(data)
        return data

    def construct_instances(self, conv_id, conv):
        raise NotImplementedError()
