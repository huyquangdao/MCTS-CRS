import os
import copy

from eval.base import BaseOnlineEval
from baselines.unimind.utils import predict_action_unimind, predict_topic_unimind, generate_response_unimind


class UnimindOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, model, tokenizer, horizon, reward_func,
                 device=None, max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True

                 ):
        super().__init__(target_set, terminal_act, horizon)
        self.model = model
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
        action = predict_action_unimind(generation_model=self.model,
                                        tokenizer=self.tokenizer,
                                        state=state,
                                        max_sequence_length=self.max_sequence_length,
                                        pad_to_multiple_of=self.pad_to_multiple_of,
                                        padding=self.padding,
                                        device=self.device)

        # generate topic
        topic = predict_topic_unimind(generation_model=self.model,
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
        system_resp = generate_response_unimind(generation_model=self.model,
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
