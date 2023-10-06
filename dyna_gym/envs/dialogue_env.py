from collections import OrderedDict
import gym
import torch
from dyna_gym.envs.utils import generate_sys_resp, get_user_resp, update_state


class DialogueEnv(gym.Env):
    """
    Langauge generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Transition: the next state is the current state concatenated with the action.
    Reward: an external function that evaluates a state (pass rate for programs, alignment score for natural language, etc.)
    Terminal state: the program reaches the maximum length or the terminal token is generated.
    """

    def __init__(self, terminal_act, horizon, reward_func, goal2id):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_act = terminal_act
        self.horizon = horizon
        self.get_reward = reward_func
        self.goal2id = goal2id
        self.id2goal = {v: k for k, v in self.goal2id.items()}

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
        resp = generate_sys_resp(state, action)

        # generate the corresponding user response using a simulator
        user_resp = get_user_resp(state, resp)

        simulated_conversation = [
            {'role': 'system', 'content': resp},
            {'role': 'user', 'content': user_resp}
        ]

        if action == self.terminal_act or len(state['dialogue_context']) > self.horizon:
            # either the text finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(simulated_conversation, self.state['task_background']['target_topic'])
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
