from collections import defaultdict
import copy

import openai
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset.data_utils import convert_list_to_str, convert_dict_to_str, convert_example_to_feature_for_goal_prediction, \
    convert_example_to_feature_for_response_generation

API_KEY = ""
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY
IGNORE_INDEX = -100


def reformat_demonstration(demonstration, is_agent_start=False):
    """
    function that reformat the demonstrative conversation
    @param demonstration: the given conversation
    @param is_agent_start: True if the system starts the conversation else False
    @return: the reformated demonstrative conversation
    """
    new_demonstration = []
    role = 0
    if is_agent_start:
        role = -1
    for utt in demonstration:
        if role % 2 == 0:
            new_demonstration.append({'role': 'user', 'content': utt})
        elif role == -1 or role % 2 != 0:
            new_demonstration.append({'role': 'assistant', 'content': utt})
        role += 1
    return new_demonstration


def generate_sys_resp(state, action):
    """ Generate a system response using ChatGPT.
    Args:
        state (_type_): the current state which consists of task background, pre_topics and prev_goals
        action (_type_): a chosen goal.
    """
    demonstration = state['demonstration']['conversation']
    knowledge_str = convert_list_to_str(state['knowledge'])
    user_profile_str = convert_dict_to_str(state['task_background']['user_profile'])
    target_topic = state['task_background']['target_topic']
    system_instruction_1 = f"""You are a recommender. You will be given a set of relevant knowledge 
        deliminated by triple backticks ```{knowledge_str}``` and information about the user
        deliminated by the following triple backticks ```{user_profile_str}```. Your task is to generate 
        a response following the action ```{action}``` using the given knowledge and user profile. If the action 
        is recommendation, then you need recommend the item {target_topic} to the user. 
        The following is an example conversation between a recommender and an user.
    """.replace('\n', '')
    # the first instruction prompt
    messages = [
        {"role": "system", "content": system_instruction_1},
    ]
    # 1-shot demonstration
    for utt in reformat_demonstration(demonstration,
                                      is_agent_start=state['demonstration']['goal_type_list'][0] == 'Greetings'):
        messages.append(utt)

    system_instruction_2 = """
    The following is a new conversation between a recommender (you) and an user.
    """
    # the second instruction prompt
    messages.append(
        {"role": "system", "content": system_instruction_2},
    )
    # current conversation
    for utt in state['dialogue_context']:
        messages.append(utt)

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=50
    )
    return response.choices[0]['message']['content']


def get_user_resp(state, sys_response):
    demonstration = state['demonstration']['conversation']
    target_topic = state['task_background']['target_topic']
    seeker_instruction_1 = '''You are an user chatting with a recommender for recommendation. 
    Your target items: {}. You must follow the instructions below during chat.
    You should never directly tell the target item title. 
    The following is an example conversation between a recommender and an user.
    '''.format(target_topic)
    # the first instruction prompt
    messages = [
        {"role": "system", "content": seeker_instruction_1},
    ]
    # 1-shot demonstration
    for utt in reformat_demonstration(demonstration,
                                      is_agent_start=state['demonstration']['goal_type_list'][0] == 'Greetings'):
        # switch role
        if utt['role'] == 'user':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        messages.append(utt)

    seeker_instruction_2 = """
    The following is a new conversation between a recommender and an user (you).
    """
    # the second instruction prompt
    messages.append(
        {"role": "system", "content": seeker_instruction_2},
    )
    # current conversation
    for utt in state['dialogue_context']:
        # switch role
        if utt['role'] == 'user':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        messages.append(utt)

    # the new generate response.
    messages.append(
        {'role': 'user', 'content': sys_response}
    )

    # getting the response.
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=50
    )
    return response.choices[0]['message']['content']


def update_state(state, action, sys_response, user_response):
    """function that updates the state of the conversation
    in order to update the state, we need to simulate an user's response using ChatGPT.
    Args:
        state (_type_): the current state of the conversation.
        action (_type_): the chosen goal.
        response (_type_): the generated user response.
    """
    # update state
    new_state = copy.deepcopy(state)
    new_state['dialogue_context'].append(
        {"role": "assistant", "content": sys_response}
    )
    new_state['dialogue_context'].append(
        {"role": "user", "content": user_response}
    )
    new_state['pre_goals'].append(action)
    return new_state


def check_terminated_condition(system_resp, target):
    """
    function check if the target item appears in the system response.
    @param system_resp: a generate utterance from the system
    @param target: the target item
    @return: True if the target appear in the generated response else False
    """
    return target.lower() in system_resp


def generate_sys_response_with_plm(generation_model, tokenizer, action, state, max_sequence_length, max_gen_length=50,
                                   pad_to_multiple_of=True, padding='max_length', device=None):
    """
    function that generates a system response with a finetuned pretrained language model
    @param generation_model: the finetuned huggingface pretrained PLM
    @param tokenizer: a huggingface tokenizer.
    @param action: the predicted action
    @param state:  the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param pad_to_multiple_of: True if we pad to multiple instances.
    @param padding: type of padding default = 'max length"
    @param device: device to allocate tensors
    @return: a generated system response
    """
    # convert state to input feature
    input_features = defaultdict(list)

    # assign the predicted action to the input state
    state['pred_goal'] = action

    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_response_generation(tokenizer=tokenizer, instance=state,
                                                                      max_sequence_length=max_sequence_length,
                                                                      is_test=True)
    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # forward the input features through the model
    gen_seqs = generation_model.generate(
        **input_features,
        max_new_tokens=max_gen_length,
        no_repeat_ngram_size=3
    )
    # remove special tokens
    gen_resp_ids = []
    for gen_seq in gen_seqs:
        gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
        gen_resp_ids.append(gen_seq)

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]


def predict_action(policy_model, tokenizer, state, max_sequence_length, goal2id=None, pad_to_multiple_of=True,
                   padding='max_length', device=None):
    """
    function that predicts an action given the input state
    @param policy_model: the offline policy model
    @param tokenizer: a huggingface tokenizer.
    @param state: the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence
    @param goal2id: a dictionary that map goals to indices
    @param pad_to_multiple_of: pad to multiple instances
    @param padding: type of padding
    @param device: device to allocate tensors
    @return: a predicted action
    """
    input_features = defaultdict(list)
    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_goal_prediction(tokenizer, state, max_sequence_length,
                                                                  goal2id)

    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # compute policy with offline policy model.
    logits = policy_model(input_features)
    pred_action_id = logits.argmax(-1).detach().cpu().numpy().tolist()[0]
    id2goal = {v: k for k, v in goal2id.items()}
    action = id2goal[pred_action_id]
    return action


def simulate_conversation(generation_model, generation_tokenizer, policy_model, policy_tokenizer, state, horizon=5,
                          max_sequence_length=512, max_gen_length=50, padding='max_length',
                          pad_to_multiple_of=True, goal2id=None, device=None):
    """
    function that simulates a conversation between an user and a system starting from a given input state.
    @param generation_model: a response generation used to produce a system response
    @param generation_tokenizer: a huggingface tokenizer used for response generation
    @param policy_model: a prediction model used to produce a system action
    @param policy_tokenizer: a huggingface tokenizer
    @param state: the current state of the env
    @param horizon: the maximum number of turn in the simulated conversation
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param padding: type of padding
    @param pad_to_multiple_of: if pad to multiple instances.
    @param goal2id: a dictionary that convert goals to indices.
    @param device the device to allocate tensors
    @return: the last generated system response.
    """
    is_terminal = False
    i = 0
    start_state = copy.deepcopy(state)
    while (not is_terminal) and i < horizon:

        # predict system action using the offline policy model
        action = predict_action(policy_model,
                                policy_tokenizer,
                                start_state,
                                max_sequence_length,
                                goal2id,
                                pad_to_multiple_of,
                                padding,
                                device=device)

        # generate the system response using chatgpt
        # later it will be replaced by the generated response by BART.
        # system_resp = get_user_resp(start_state, action)
        system_resp = generate_sys_response_with_plm(generation_model=generation_model,
                                                     tokenizer=generation_tokenizer,
                                                     action=action,
                                                     state=start_state,
                                                     max_sequence_length=max_sequence_length,
                                                     max_gen_length=max_gen_length,
                                                     pad_to_multiple_of=pad_to_multiple_of,
                                                     padding=padding,
                                                     device=device)
        # check the terminated condition
        if check_terminated_condition(system_resp, state['task_background']['target_topic']):
            is_terminal = True

        # simulate user response.
        user_resp = get_user_resp(start_state, system_resp)

        # update state
        start_state = update_state(start_state, action, system_resp, user_resp)

        i += 1

    # return the last system resp.
    return system_resp
