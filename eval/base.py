import os

from dyna_gym.envs.utils import simulate_conversation, update_state, get_user_resp


class BaseOnlineEval(object):

    def __init__(self, target_set, terminal_act, horizon):
        self.terminal_act = terminal_act
        self.target_set = target_set
        self.horizon = horizon

    def pipeline(self, state):
        raise NotImplementedError()

    def init_agent(self):
        raise NotImplementedError()

    def get_user_resp(self, state, system_resp):
        """
        method that simulates the user response
        @param state: the current state of the conversation
        @param system_resp: the generated system response
        @return: the generated user response
        """
        return get_user_resp(state, system_resp)

    def init_state(self, target_item):
        """
        method that create the initial state of a conversation
        we assume the user start a conversation.
        @param target_item:
        @return:
        """
        state = {
            "task_background": {
                "target_topic": target_item['topic'],
                "target_goal": target_item['goal']
            },
            "demonstration": target_item["demonstration"],
            "dialogue_context": [{"role": "user", "content": "Hello !"}],
            "goal": "Greetings",  # will not affect anything, only including it for code convenience
            "knowledge": "",  # will not affect anything, only including it for code convenience
            "response": "",  # will not affect anything, only including it for code convenience
            "pre_goals": []
        }
        return state

    def update(self, state, system_response, system_action, user_response):
        """
        method that update the state of the conversation
        @param state: the current state of the conversation
        @param system_response: the generated system response
        @param system_action: the predicted system action
        @param user_response: the generated user response
        @return:
        """
        new_state = update_state(state,
                                 action=system_action,
                                 sys_response=system_response,
                                 user_response=user_response
                                 )
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
            system_resp, system_act = self.pipeline(state)

            # generate user response
            user_resp = self.get_user_resp(state, system_resp)

            # check the terminated condition
            if self.check_terminated_condition(system_act):
                is_terminated = True

            # update the state of the conversation
            state = self.update(state, system_resp, system_act, user_resp)

            # update count
            count += 1

            # update the simulated conversation
            generated_conversation.extend([
                {'role': 'system', 'content': system_resp},
                {'role': 'user', 'content': user_resp}
            ])
        return generated_conversation

    def eval(self):
        """
        method that perform online evaluation on a predefined set of items
        @return: computed metrics
        """
        avg_sr = []
        avg_turn = []
        for target_item in self.target_set:
            initial_state = self.init_state(target_item)
            generated_conversation = self.run(initial_state)
            sr, turn = self.compute_metrics(generated_conversation, target_item['topic'])
            avg_sr.append(sr)
            avg_turn.append(turn)
        return sum(avg_sr) / len(self.target_set), sum(avg_turn) / len(self.target_set)

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
        turn = self.compute_turn(generated_conversation)
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
