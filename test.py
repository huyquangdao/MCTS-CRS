from dataset.durecdial import DuRecdial
from dyna_gym.envs.utils import generate_sys_resp, get_user_resp
from dataset.data_utils import convert_example_to_feature


if __name__ == '__main__':
    
    train_data_path = 'data/DuRecDial/data/en_train.txt'    
    dev_data_path = 'data/DuRecDial/data/en_dev.txt'
    test_data_path = 'data/DuRecDial/data/en_test.txt'
    
    durecdial = DuRecdial(train_data_path=train_data_path,
                          dev_data_path=dev_data_path,
                          test_data_path=test_data_path)
    
    convert_example_to_feature(None, durecdial.train_instances[0])