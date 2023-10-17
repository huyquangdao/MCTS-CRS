import os

from dataset.base import BaseTorchDataset


class UnimindTorchDataset(BaseTorchDataset):

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
        super(UnimindTorchDataset, self).__init__(
            tokenizer,
            instances,
            goal2id,
            max_sequence_length,
            padding,
            pad_to_multiple_of,
            device,
            convert_example_to_feature,
            max_target_length,
            is_test,
            is_gen
        )

    def __preprocess_data(self, instances, convert_example_to_feature):

        assert isinstance(convert_example_to_feature, dict) is True
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
