import os

from dyna_gym.envs.utils import simulate_conversation


class BaseOnlineEval(object):

    def __init__(self, target_set, terminal_act):
        self.terminal_act = terminal_act
        self.target_set = target_set

    def pipeline(self, state):
        raise NotImplementedError()

    def get_user_resp(self, state, system_resp):
        raise NotImplementedError()

    def init_state(self, state):
        raise NotImplementedError()

    def update(self, state, system_response, system_action, user_response):
        raise NotImplementedError()

    def run(self, init_state):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def check_terminated_condition(self, system_action):
        return system_action == self.terminal_act

    def compute_metrics(self, generated_conversations):
        raise NotImplementedError()
