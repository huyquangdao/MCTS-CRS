import openai
API_KEY = ""
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY

def convert_list_to_str(lis):
    """function that converts a list of elements to a text string

    Args:
        lis (_type_): list of lists, each element in the list is a list with 3 elements.

    Returns:
        _type_: a text string.
    """
    output_str = ""
    for k in lis:
        if len(k) > 0:
            tmp = f"{k[0]} {k[1]} {k[2]} "
            output_str += tmp
    return output_str

def generate_sys_resp(state, action):
    """ Generate a system response using ChatGPT.
    Args:
        state (_type_): the current state which consists of task background, pre_topics and prev_goals
        action (_type_): a chosen goal.
    """
    knowledge_str = convert_list_to_str(state['task_background']['knowledge_base'])
    user_profile_str = convert_list_to_str(state['task_background']['user_profile'])
    target_topic = state['task_background']['target_topic']
    system_instruction = f"""You are an AI research assistant. You will be given a set of relevant knowledge
        deliminated by triple backticks ```{knowledge_str}``` and information about the user
        deliminated by the following triple backticks ```{user_profile_str}```. Your task is to generate 
        a response following the action {action} using the given knowledge and user profile. If the action 
        is recommendation, then you need recommend the item {target_topic} to the user.
    """
    messages = [
        {"role": "system", "content": system_instruction},
    ]
    for utt in state['dialogue_context']:
        messages.append(utt)
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0]['message']['content']


def get_user_resp(state, sys_response, logit_bias = None):
    if logit_bias is None:
        logit_bias = {}
    target_topic = state['task_background']['target_topic']
    seeker_instruction = '''You are a seeker chatting with a recommender for recommendation. 
    Your target items: {}. 
    You must follow the instructions below during chat.
    You should never directly tell the target item title.
    You should respond to the recommender's response: {}
    '''.format(target_topic, sys_response)
    response = openai.Completion.create(
        model='text-davinci-003', 
        prompt=seeker_instruction, 
        temperature=0, 
        max_tokens=128, 
        stop='Recommender',
        logit_bias=logit_bias,
    )
    return response['choices'][0]['text']

def update_state(state, action, sys_response, user_response):
    """function that updates the state of the conversation
    in order to update the state, we need to simulate an user's response using ChatGPT.
    Args:
        state (_type_): the current state of the conversation.
        action (_type_): the chosen goal.
        response (_type_): the generated user response.
    """
    ### update state
    state['dialogue_context'].append(
        {"role": "assistant", "content": sys_response}
    )
    state['dialogue_context'].append(
        {"role": "user", "content": user_response}
    )
    state['prev_goals'].append(action)
    return state