import os
import copy
from eval.base import BaseOnlineEval
from baselines.rtcp.utils import predict_action_rtcp, generate_response_rtcp

from dyna_gym.envs.utils import generate_knowledge_with_plm, \
    generate_sys_response_with_plm, get_system_response_with_LLama


class RTCPOnlineEval(BaseOnlineEval):
    """
    Online evaluation class for the original RTCP implementation
    """

    def __init__(self, target_set, terminal_act, policy_model, response_model, policy_tokenizer, generation_tokenizer,
                 horizon, goal2id, topic2id,
                 device=None, max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True
                 ):
        """
        constructor for class unimind online eval
        @param target_set:  the target item set
        @param terminal_act: the terminated action, default is say goodbye
        @param policy_model: the topic generation model
        @param response_model: the response generation model
        @param policy_tokenizer: huggingface tokenizer
        @param generation_tokenizer: huggingface tokenizer
        @param horizon: the maximum number of conversational turns
        @param goal2id: dictionary that maps goals to indices
        @param topic2id: dictionary that maps topics to indices
        @param device:  the device
        @param max_sequence_length: maximum number of token in the input sequence
        @param pad_to_multiple_of:  type of padding
        @param padding: type of padding
        @param max_gen_length: maximum number of token in the generated response
        @param model_generation_args:
        @param should_plot_tree:
        """
        super().__init__(target_set, terminal_act, horizon)
        self.policy_model = policy_model
        self.response_model = response_model
        self.policy_tokenizer = policy_tokenizer
        self.generation_tokenizer = generation_tokenizer
        self.horizon = horizon
        self.goal2id = goal2id
        self.topic2id = topic2id
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.max_gen_length = max_gen_length
        self.model_generation_args = model_generation_args
        self.should_plot_tree = should_plot_tree

    def init_agent(self):
        pass

    def update(self, state, system_response, system_action, system_topic, user_response):
        # update state
        new_state = copy.deepcopy(state)
        new_state['dialogue_context'].append(
            {"role": "assistant", "content": system_response}
        )
        new_state['dialogue_context'].append(
            {"role": "user", "content": user_response}
        )
        new_state['pre_goals'].append(system_action)
        new_state['pre_topics'].append(system_topic)
        return new_state

    def run(self, init_state):
        """
        method that employs one online evaluation
        @param init_state:
        @return: a generated conversation between user and system
        """
        is_terminated = False
        count = 0
        generated_conversation = []
        state = init_state
        while not is_terminated and count < self.horizon:

            # generate system response and action
            system_resp, system_act, system_topic = self.pipeline(state)

            # generate user response
            user_resp = self.get_user_resp(state, system_resp)

            # check the terminated condition
            if self.check_terminated_condition(system_act):
                is_terminated = True

            # update the state of the conversation
            state = self.update(state, system_resp, system_act, system_topic, user_resp)

            # update count
            count += 1

            # update the simulated conversation
            generated_conversation.extend([
                {'role': 'system', 'content': system_resp},
                {'role': 'user', 'content': user_resp}
            ])
        return generated_conversation

    def pipeline(self, state):

        # greedily predict the system action using the offline policy model
        action, topic = predict_action_rtcp(policy_model=self.policy_model,
                                            policy_tokenizer=self.policy_tokenizer,
                                            state=state,
                                            max_sequence_length=self.max_sequence_length,
                                            goal2id=self.goal2id,
                                            topic2id=self.topic2id,
                                            pad_to_multiple_of=self.pad_to_multiple_of,
                                            padding=self.padding,
                                            device=self.device)

        # generate the system response using chatgpt
        # later it will be replaced by the generated response by BART.
        # system_resp = get_user_resp(start_state, action)
        system_resp = generate_response_rtcp(generation_model=self.response_model,
                                             generation_tokenizer=self.generation_tokenizer,
                                             action=action,
                                             topic=topic,
                                             state=state,
                                             goal2id=self.goal2id,
                                             topic2id=self.topic2id,
                                             max_sequence_length=self.max_sequence_length,
                                             max_gen_length=self.max_gen_length,
                                             pad_to_multiple_of=self.pad_to_multiple_of,
                                             padding=self.padding,
                                             device=self.device)
        return system_resp, action, topic


class RTCPBartOnlineEval(BaseOnlineEval):
    """
    Online evaluation class for RTCP with Bart knowledge and text generation model
    """

    def __init__(self, target_set, terminal_act, use_llm_score, n, epsilon, use_demonstration, generation_model,
                 generation_tokenizer,
                 know_generation_model,
                 know_generation_tokenizer,
                 policy_model, policy_tokenizer, horizon, goal2id, topic2id, device=None,
                 max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50
                 ):
        """
        constructor for class MCTSCRSOnlineEval
        @param target_set:
        @param generation_model:
        @param generation_tokenizer:
        @param know_generation_model:
        @param know_generation_tokenizer:
        @param policy_model:
        @param policy_tokenizer:
        @param horizon:
        @param goal2id:
        @param device:
        @param max_sequence_length:
        @param max_gen_length:
        """

        super().__init__(target_set, terminal_act, horizon, use_llm_score, epsilon, n, use_demonstration)
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.know_generation_model = know_generation_model
        self.know_generation_tokenizer = know_generation_tokenizer
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.topic2id = topic2id
        self.horizon = horizon
        self.goal2id = goal2id
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.max_gen_length = max_gen_length

    def update(self, state, system_response, system_action, user_response):
        # update state
        new_state = copy.deepcopy(state)
        new_state['dialogue_context'].append(
            {"role": "assistant", "content": system_response}
        )
        new_state['dialogue_context'].append(
            {"role": "user", "content": user_response}
        )
        goal, topic = system_action
        new_state['pre_goals'].append(goal)
        new_state['pre_topics'].append(topic)
        return new_state

    def check_terminated_condition(self, system_action):
        """
        function that check if the conversation is terminated
        @param system_action: the input system action (goal)
        @return: True if the conversation is terminated else False
        """
        return system_action[0] == self.terminal_act

    def pipeline(self, state):
        """
        method that perform one system pipeline including action prediction, knowledge generation and response generation
        @param state: the current state of the conversation
        @return: generated system response and predicted system action
        """
        # predict action with RTCP policy
        action = predict_action_rtcp(policy_model=self.policy_model,
                                     policy_tokenizer=self.policy_tokenizer,
                                     state=state,
                                     max_sequence_length=self.max_sequence_length,
                                     goal2id=self.goal2id,
                                     topic2id=self.topic2id,
                                     pad_to_multiple_of=self.pad_to_multiple_of,
                                     padding=self.padding,
                                     device=self.device)

        # generate relevant knowledge
        knowledge = generate_knowledge_with_plm(generation_model=self.know_generation_model,
                                                tokenizer=self.know_generation_tokenizer,
                                                action=action,
                                                state=state,
                                                max_sequence_length=self.max_sequence_length,
                                                max_gen_length=self.max_gen_length,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                padding=self.padding,
                                                device=self.device)

        # generate the system response using chatgpt
        # later it will be replaced by the generated response by BART.
        # system_resp = get_user_resp(start_state, action)
        system_resp = generate_sys_response_with_plm(generation_model=self.generation_model,
                                                     tokenizer=self.generation_tokenizer,
                                                     action=action,
                                                     knowledge=knowledge,
                                                     state=state,
                                                     max_sequence_length=self.max_sequence_length,
                                                     max_gen_length=self.max_gen_length,
                                                     pad_to_multiple_of=self.pad_to_multiple_of,
                                                     padding=self.padding,
                                                     device=self.device)
        return system_resp, action


class RTCPLLamaOnlineEval(BaseOnlineEval):
    """
    Online evaluation class for RTCP with Llama
    """

    def __init__(self, target_set, terminal_act, use_llm_score, n, epsilon, use_demonstration,
                 policy_model, policy_tokenizer, horizon, goal2id, topic2id, device=None,
                 max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50
                 ):
        """
        constructor for class MCTSCRSOnlineEval
        @param target_set:
        @param policy_model:
        @param policy_tokenizer:
        @param horizon:
        @param goal2id:
        @param device:
        @param max_sequence_length:
        @param max_gen_length:
        """

        super().__init__(target_set, terminal_act, horizon, use_llm_score, epsilon, n, use_demonstration)
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.topic2id = topic2id
        self.horizon = horizon
        self.goal2id = goal2id
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.max_gen_length = max_gen_length

    def update(self, state, system_response, system_action, user_response):
        # update state
        new_state = copy.deepcopy(state)
        new_state['dialogue_context'].append(
            {"role": "assistant", "content": system_response}
        )
        new_state['dialogue_context'].append(
            {"role": "user", "content": user_response}
        )
        goal, topic = system_action
        new_state['pre_goals'].append(goal)
        new_state['pre_topics'].append(topic)
        return new_state

    def check_terminated_condition(self, system_action):
        """
        function that check if the conversation is terminated
        @param system_action: the input system action (goal)
        @return: True if the conversation is terminated else False
        """
        return system_action[0] == self.terminal_act

    def pipeline(self, state):
        """
        method that perform one system pipeline including action prediction, knowledge generation and response generation
        @param state: the current state of the conversation
        @return: generated system response and predicted system action
        """
        # predict action with RTCP policy
        action = predict_action_rtcp(policy_model=self.policy_model,
                                     policy_tokenizer=self.policy_tokenizer,
                                     state=state,
                                     max_sequence_length=self.max_sequence_length,
                                     goal2id=self.goal2id,
                                     topic2id=self.topic2id,
                                     pad_to_multiple_of=self.pad_to_multiple_of,
                                     padding=self.padding,
                                     device=self.device)

        """
        note: you need to implement the following
         get_system_response_with_LLama function that produces a text string as output.
        the inputs are (1): the current state of the conversation, which contains dialogue context, previous goals
        previous topics and a demonstration.
        (2) the predicted action which is a tuple of two elements, one is a goal, the other is the topic.
        For prompt and instructions, please take a look at the generate_sys_resp function in dynagym/envs/utils.py file
        """
        system_resp = get_system_response_with_LLama(state, action)
        return system_resp, action
