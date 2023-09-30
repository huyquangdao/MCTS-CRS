from __future__ import print_function
from collections import defaultdict

import torch

from torch.utils.data import Dataset as TorchDataset


class BaseTorchDataset(TorchDataset):

    def __init__(self, tokenizer, instances, goal2id=None, max_sequence_length=512, padding='max_length',
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
        super(BaseTorchDataset, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.goal2id = goal2id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.device = device
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.is_gen = is_gen
        self.instances = self.__preprocess_data(instances, convert_example_to_feature)

    def __len__(self):
        """
        method that returns the number of instances in the dataset.
        @return: an integer which is the number of instances in the training dataset.
        """
        return len(self.instances)

    def __getitem__(self, idx):
        """
        function that return an instance from the set of all instances.
        @param idx: the index of the returned instances.
        @return: an instance.
        """
        instance = self.instances[idx]
        return instance

    def collate_fn(self, batch):

        input_features = defaultdict(list)
        labels = []
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            labels.append(instance['label'])

        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device)

        labels = torch.LongTensor(labels).to(self.device)
        new_batch = {
            "context": input_features,
            "labels": labels
        }
        return new_batch

    def __preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            if not self.is_gen:
                input_ids, label = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length,
                                                              self.goal2id)
            else:
                input_ids, label = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length,
                                                              self.max_target_length, self.is_test)
            new_instance = {
                "input_ids": input_ids,
                "label": label
            }
            processed_instances.append(new_instance)
        return processed_instances


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
        """
        method that converts a conversation to a list of inputs and their corresponding outputs.
        @param conv_id: the index of the conversation
        @param conv: the conversation
        @return: a list of instances
        """
        raise NotImplementedError()
