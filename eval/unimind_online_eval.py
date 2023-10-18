import os

from eval.base import BaseOnlineEval
from baselines.unimind.utils import predict_action_unimind, predict_topic_unimind, generate_response_unimind


class UnimindOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, model, tokenizer, horizon, reward_func,
                 device=None, max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True

                 ):
        super().__init__(target_set, terminal_act)
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
        return system_resp, action
