import os
from dataset.base import BaseTorchDataset


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
