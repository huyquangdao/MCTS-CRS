from dataset.durecdial import DuRecdial
from dyna_gym.envs.utils import generate_sys_resp, get_user_resp


if __name__ == '__main__':
    
    train_data_path = 'data/DuRecDial/data/en_train.txt'    
    dev_data_path = 'data/DuRecDial/data/en_dev.txt'
    test_data_path = 'data/DuRecDial/data/en_test.txt'
    
    durecdial = DuRecdial(train_data_path=train_data_path,
                          dev_data_path=dev_data_path,
                          test_data_path=test_data_path)
    
    resp = generate_sys_resp(durecdial.train_instances[0], action='Movie recommendation')    
    user_resp = get_user_resp(durecdial.train_instances[0], resp)
    
    print(resp)
    print(user_resp)