import os
import copy
from eval.base import BaseOnlineEval

from baselines.bart.utils import generate_response_bart


class BartOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, response_model, tokenizer, horizon,
                 reward_func, device=None, max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True
                 ):
        """
        constructor for class unimind online eval
        @param target_set:  the target item set
        @param terminal_act: the terminated action, default is say goodbye
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

    def update(self, state, system_response, user_response):
        """
        method that updates the state of the conversation
        @param state: the state of the conversation
        @param system_response: the generated system response
        @param user_response: the generated user response
        @return: the new updated state.
        """
        # update state
        new_state = copy.deepcopy(state)
        new_state['dialogue_context'].append(
            {"role": "assistant", "content": system_response}
        )
        new_state['dialogue_context'].append(
            {"role": "user", "content": user_response}
        )
        return new_state

    def run(self, init_state):
        """
        method that employs one online evaluation
        @param init_state:
        @return: a generated conversation between user and system
        """
        count = 0
        generated_conversation = []
        state = init_state
        # Bart solely generate the response without any planning strategies.
        # Therefore we terminate the conversation if its length is more than a certain threshold.
        while count < self.horizon:
            # generate system response and action
            system_resp = self.pipeline(state)

            # generate user response
            user_resp = self.get_user_resp(state, system_resp)

            # update the state of the conversation
            state = self.update(state, system_resp, user_resp)

            # update count
            count += 1

            # update the simulated conversation
            generated_conversation.extend([
                {'role': 'system', 'content': system_resp},
                {'role': 'user', 'content': user_resp}
            ])
        return generated_conversation

    def pipeline(self, state):
        system_resp = generate_response_bart(generation_model=self.response_model,
                                             tokenizer=self.tokenizer,
                                             state=state,
                                             max_sequence_length=self.max_sequence_length,
                                             max_gen_length=self.max_gen_length,
                                             pad_to_multiple_of=self.pad_to_multiple_of,
                                             padding=self.padding,
                                             device=self.device)
        return system_resp
