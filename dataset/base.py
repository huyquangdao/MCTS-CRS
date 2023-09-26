
from __future__ import print_function

class Dataset:
    
    def __init__(self, train_data_path, dev_data_path, test_data_path):
        self.train_data_path = train_data_path
        self.dev_data_path = dev_data_path
        self.test_data_path = test_data_path
        
        self.topics = []
        self.goals = []
        
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
        data = self.read_data(data_path= data_path)
        data = self.repurpose_dataset(data)
        data = self.process_data(data)
        return data

    def construct_instances(self, conv_id, conv):
        raise NotImplementedError()             
                            