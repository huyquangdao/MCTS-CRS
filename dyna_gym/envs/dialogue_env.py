from collections import OrderedDict
import gym
import torch
from dyna_gym.envs.utils import predict_action, generate_sys_response_with_plm, generate_knowledge_with_plm, \
    get_user_resp, update_state


class DialogueEnv(gym.Env):
    """
    Langauge generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Transition: the next state is the current state concatenated with the action.
    Reward: an external function that evaluates a state (pass rate for programs, alignment score for natural language, etc.)
    Terminal state: the program reaches the maximum length or the terminal token is generated.
    """

    def __init__(self, generation_model, generation_tokenizer, know_generation_model, know_tokenizer,
                 terminal_act, horizon=5, max_sequence_length=512, max_gen_length=50, pad_to_multiple_of=True,
                 padding='max_length', device=None,
                 reward_func=None, goal2id=None):
        """

        @param generation_model:
        @param generation_tokenizer:
        @param know_generation_model:
        @param know_generation_tokenizer:
        @param terminal_act:
        @param horizon:
        @param max_sequence_length:
        @param max_gen_length:
        @param pad_to_multiple_of:
        @param padding:
        @param device:
        @param reward_func:
        @param goal2id:
        """
        self.terminal_act = terminal_act
        self.horizon = horizon
        self.get_reward = reward_func
        self.goal2id = goal2id
        self.id2goal = {v: k for k, v in self.goal2id.items()}
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.know_generation_model = know_generation_model
        self.know_tokenizer = know_tokenizer
        self.max_sequence_length = max_sequence_length
        self.max_gen_length = max_gen_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.device = device

    def reset(self, state):
        self.state = state
        return self.state

    def transition(self, state, action, is_model_dynamic=False):
        """Transition method used to update the state of the MDP process
        Args:
            state (_type_): the current state
            action (_type_): the chosen dialogue action
            is_model_dynamic (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: next state, action, reward and flag which indicates whether we terminate the process.
        """
        # generate a response (which can be either user or system response)
        # given the current state and the chosen action.
        action = self.id2goal[action]

        # generate relevant knowledge
        knowledge = generate_knowledge_with_plm(generation_model=self.know_generation_model,
                                                tokenizer=self.know_tokenizer,
                                                action=action,
                                                state=state,
                                                max_sequence_length=self.max_sequence_length,
                                                max_gen_length=self.max_gen_length,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                padding=self.padding,
                                                device=self.device)

        # generate system response
        resp = generate_sys_response_with_plm(generation_model=self.generation_model,
                                              tokenizer=self.generation_tokenizer,
                                              action=action,
                                              knowledge=knowledge,
                                              state=state,
                                              max_sequence_length=self.max_sequence_length,
                                              max_gen_length=self.max_gen_length,
                                              pad_to_multiple_of=self.pad_to_multiple_of,
                                              padding=self.padding,
                                              device=self.device)

        # generate the corresponding user response using a simulator
        user_resp = get_user_resp(state, resp)

        simulated_conversation = [
            {'role': 'system', 'content': resp, 'goal': action},
            {'role': 'user', 'content': user_resp}
        ]

        if action == self.terminal_act or len(state['dialogue_context']) > self.horizon:
            # either the text finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(simulated_conversation, self.state['task_background']['target_topic'],
                                     self.state['task_background']['target_goal'])
        else:
            reward = 0  # no intermediate reward

        # update the current state.
        new_state = update_state(state, action, resp, user_resp)
        return new_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)
        return self.state, reward, done, {}

    def equality_operator(self, s1, s2):
        # s1 and s2 are dictionaries
        return s1 == s2
