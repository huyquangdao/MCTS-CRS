import os
import pickle
import copy

from dyna_gym.envs.utils import update_state, predict_action, generate_knowledge_with_plm, \
    generate_sys_response_with_plm, get_user_resp
from eval.base import BaseOnlineEval
from dyna_gym.pipelines.uct_for_dialogue_planning import uct_for_dialogue_planning_pipeline


class MCTSCRSOnlineEval(BaseOnlineEval):

    def __init__(self, target_set, terminal_act, generation_model, generation_tokenizer, know_generation_model,
                 know_generation_tokenizer,
                 policy_model, policy_tokenizer, horizon, reward_func, uct_args, goal2id, device=None,
                 max_sequence_length=512, offline_policy=False, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, model_generation_args=None, should_plot_tree=True
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
        @param reward_func:
        @param uct_args:
        @param goal2id:
        @param device:
        @param max_sequence_length:
        @param offline_policy:
        @param max_gen_length:
        @param model_generation_args:
        @param should_plot_tree:
        """

        super().__init__(target_set, terminal_act, horizon)
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.know_generation_model = know_generation_model
        self.know_generation_tokenizer = know_generation_tokenizer
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.horizon = horizon
        self.reward_func = reward_func
        self.uct_args = uct_args
        self.goal2id = goal2id
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.offline_policy = offline_policy
        self.max_gen_length = max_gen_length
        self.model_generation_args = model_generation_args
        self.should_plot_tree = should_plot_tree

        self.mcts_agent = self.init_agent()

    def init_agent(self):
        """
        method that initializes the Monte-Carlo Tree Search agent
        @return: the constructed state.
        """

        mcts_agent = uct_for_dialogue_planning_pipeline(
            generation_model=self.generation_model,
            generation_tokenizer=self.generation_tokenizer,
            know_generation_model=self.know_generation_model,
            know_tokenizer=self.know_generation_tokenizer,
            policy_model=self.policy_model,
            policy_tokenizer=self.policy_tokenizer,
            horizon=self.horizon,
            reward_func=self.reward_func,
            uct_args=self.uct_args,
            goal2id=self.goal2id,
            device=self.device,
            max_sequence_length=self.max_sequence_length,
            max_gen_length=self.max_gen_length,
            model_generation_args=self.model_generation_args,
            should_plot_tree=True,  # plot the tree after generation
        )

        return mcts_agent

    def pipeline(self, state):
        """
        method that perform one system pipeline including action prediction, knowledge generation and response generation
        @param state: the current state of the conversation
        @return: generated system response and predicted system action
        """
        if not self.offline_policy:
            # predict system action using monte-carlo tree search
            action = self.mcts_agent(state)
        else:
            # predict the system action using the offline policy model
            action = predict_action(self.policy_model,
                                    self.policy_tokenizer,
                                    state,
                                    self.max_sequence_length,
                                    self.goal2id,
                                    self.pad_to_multiple_of,
                                    self.padding,
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
