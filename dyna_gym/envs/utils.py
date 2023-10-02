import copy

import openai

from dataset.data_utils import convert_list_to_str, convert_dict_to_str

API_KEY = ""
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY


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


def simulate_conversation(state, horizon=5):
    """
    Simulate a conversation starting from the given state
    @param state: the current state of the conversation
    @param horizon the maximum of turns in the simulated conversation
    @return: an ended conversation which starts from the input state.
    """
    is_terminal = False
    i = 0
    while not is_terminal or i < horizon:
        system_resp = get_user_resp(state)

