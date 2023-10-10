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

    def compute_metrics(self, generated_conversation, target_item):
        """
        method that compute the dialogue-level SR and avg number of conversational turn
        @param generated_conversation: set of generated conversations between user and system
        @param target_item: set of target item
        @return: dialogue-level SR and averaged number of conversational turn
        """
        sr = self.is_successful(generated_conversation, target_item)
        turn = self.compute_turn(generated_conversation, target_item)
        return int(sr), turn

    def is_successful(self, generated_conversation, target_item):
        """
        method that check if the system successfully recommended the target item to the user.
        @param generated_conversation: the generated conversation between user and system
        @param target_item: the targeted item
        @return: True if success else False
        """
        for utt in generated_conversation:
            if utt['role'] == 'system' and target_item.lower() in utt['content'].lower():
                return True
        return False

    def compute_turn(self, generated_conversation):
        """
        method that compute the number of turn needed to end the conversation
        @param generated_conversation: the generated conversation between user and system
        @return: a int number which stands for the number of conversational turn
        """
        return len(generated_conversation)
