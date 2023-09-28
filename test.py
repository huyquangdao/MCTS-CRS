from dataset.durecdial import DuRecdial
from dyna_gym.envs.utils import generate_sys_resp, get_user_resp, update_state
from dataset.data_utils import randomly_sample_demonstrations

# from dataset.data_utils import convert_example_to_feature
# from dataset.data_utils import randomly_sample_demonstrations

if __name__ == '__main__':
    train_data_path = 'data/DuRecDial/data/en_train.txt'
    dev_data_path = 'data/DuRecDial/data/en_dev.txt'
    test_data_path = 'data/DuRecDial/data/en_test.txt'

    durecdial = DuRecdial(train_data_path=train_data_path,
                          dev_data_path=dev_data_path,
                          test_data_path=test_data_path,
                          save_train_convs=True)

    # simulate a conversation:
    demonstrations = randomly_sample_demonstrations(
        all_convs=durecdial.train_convs,
        instance=durecdial.test_instances[0]
    )
    durecdial.test_instances[0]['demonstration'] = demonstrations[0]
    state = durecdial.test_instances[0]
    while True:
        resp = generate_sys_resp(state, action='Movie recommendation')
        user_resp = get_user_resp(state, resp)
        print('[System]: ', resp)
        print('[USER]: ', user_resp)
        #update the state
        state = update_state(state, None, resp, user_resp)

