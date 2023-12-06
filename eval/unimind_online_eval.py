import os
import copy

from eval.base import BaseOnlineEval
from baselines.unimind.utils import predict_action_unimind, predict_topic_unimind, generate_response_unimind

from dyna_gym.envs.utils import generate_knowledge_with_plm, \
    generate_sys_response_with_plm, get_system_response_with_LLama


class UnimindOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, goal_model, topic_model, response_model, tokenizer, horizon,
                 reward_func,
                 device=None, max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True

                 ):
        """
        constructor for class unimind online eval
        @param target_set:  the target item set
        @param terminal_act: the terminated action, default is say goodbye
        @param goal_model: the goal generation model
        @param topic_model: the topic generation model
        @param response_model: the response generation model
        @param tokenizer: huggingface tokenizer
        @param horizon: the maximum number of conversational turns
        @param reward_func: a predefined reward function
        @param device:  the device
        @param max_sequence_length: maximum number of token in the input sequence
        @param pad_to_multiple_of:  type of padding
        @param padding: type of padding
        @param max_gen_length: maximum number of token in the generated response
        @param model_generation_args:
        @param should_plot_tree:
        """
        super().__init__(target_set, terminal_act, horizon)
        self.goal_model = goal_model
        self.topic_model = topic_model
        self.response_model = response_model
        self.tokenizer = tokenizer
        self.horizon = horizon
        self.reward_func = reward_func
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
        action = predict_action_unimind(generation_model=self.goal_model,
                                        tokenizer=self.tokenizer,
                                        state=state,
                                        max_sequence_length=self.max_sequence_length,
                                        pad_to_multiple_of=self.pad_to_multiple_of,
                                        padding=self.padding,
                                        device=self.device)

        # generate topic
        topic = predict_topic_unimind(generation_model=self.topic_model,
                                      tokenizer=self.tokenizer,
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
        system_resp = generate_response_unimind(generation_model=self.response_model,
                                                tokenizer=self.tokenizer,
                                                action=action,
                                                topic=topic,
                                                state=state,
                                                max_sequence_length=self.max_sequence_length,
                                                max_gen_length=self.max_gen_length,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                padding=self.padding,
                                                device=self.device)
        return system_resp, action, topic


class UnimindBartOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, use_llm_score, n, k, epsilon, use_demonstration, goal_model,
                 goal_model_tokenizer, topic_model, topic_model_tokenizer,
                 know_generation_model,
                 know_generation_tokenizer,
                 generation_model, generation_tokenizer, horizon,
                 reward_func,
                 device=None, max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50
                 ):
        """
        constructor for class unimind online eval
        @param target_set:  the target item set
        @param terminal_act: the terminated action, default is say goodbye
        @param goal_model: the goal generation model
        @param topic_model: the topic generation model
        @param response_model: the response generation model
        @param tokenizer: huggingface tokenizer
        @param horizon: the maximum number of conversational turns
        @param reward_func: a predefined reward function
        @param device:  the device
        @param max_sequence_length: maximum number of token in the input sequence
        @param pad_to_multiple_of:  type of padding
        @param padding: type of padding
        @param max_gen_length: maximum number of token in the generated response
        @param model_generation_args:
        @param should_plot_tree:
        """

        super().__init__(target_set, terminal_act, horizon, use_llm_score, epsilon, n, use_demonstration, k)
        self.goal_model = goal_model
        self.topic_model = topic_model
        self.goal_tokenizer = goal_model_tokenizer
        self.topic_tokenizer = topic_model_tokenizer
        self.know_generation_model = know_generation_model
        self.know_generation_tokenizer = know_generation_tokenizer
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.horizon = horizon
        self.reward_func = reward_func
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.max_gen_length = max_gen_length

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
        goal = predict_action_unimind(generation_model=self.goal_model,
                                      tokenizer=self.goal_tokenizer,
                                      state=state,
                                      max_sequence_length=self.max_sequence_length,
                                      pad_to_multiple_of=self.pad_to_multiple_of,
                                      padding=self.padding,
                                      device=self.device)

        # generate topic
        topic = predict_topic_unimind(generation_model=self.topic_model,
                                      tokenizer=self.topic_tokenizer,
                                      action=goal,
                                      state=state,
                                      max_sequence_length=self.max_sequence_length,
                                      max_gen_length=self.max_gen_length,
                                      pad_to_multiple_of=self.pad_to_multiple_of,
                                      padding=self.padding,
                                      device=self.device)

        action = (goal, topic)

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

        return system_resp, action, topic
