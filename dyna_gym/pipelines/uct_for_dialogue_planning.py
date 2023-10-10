from datetime import datetime
from typing import Callable, Sequence

import gym
import torch
import transformers

from dyna_gym.agents import uct
from dyna_gym.default_policy.offline_policy import OfflinePolicy
from dyna_gym.utils.tree_search_utils import print_tree


def uct_for_dialogue_planning_pipeline(
        generation_model,
        generation_tokenizer,
        know_generation_model,
        know_tokenizer,
        policy_model,
        policy_tokenizer: transformers.PreTrainedTokenizer,
        horizon: int = 5,
        terminal_act: str = 'Say goodbye',
        max_sequence_length=512,
        max_gen_length=50,
        reward_func: Callable = None,
        uct_args: dict = {},
        model_generation_args: dict = {},
        goal2id: dict = {},
        device=None,
        should_plot_tree: bool = False
) -> Callable:
    """
    function that implements the pipeline for MCTS dialogue planning
    @param generation_model: the response generation model
    @param generation_tokenizer: the tokenizer for response generation model
    @param know_generation_model: the knowledge generation model
    @param know_tokenizer: the tokenizer for the knowledge generation model
    @param policy_model: the policy model
    @param policy_tokenizer: the tokenizer for the policy model
    @param horizon: the maximum number of conversational turns during rollouts
    @param terminal_act: the terminated action
    @param max_sequence_length: the maximum number of tokens in the input sequence
    @param max_gen_length: the maximum number of tokens in the generated output
    @param reward_func: the reward function
    @param uct_args: parameters for UCT criteria.
    @param model_generation_args: params for generation model.
    @param goal2id: a dictionary that map goals to indices.
    @param device: the device which we run the models.
    @param should_plot_tree:
    @return: a function.
    """
    reward_func_ = reward_func
    env = gym.make(
        'DialogueEnv-v0',
        generation_model=generation_model,
        generation_tokenizer=generation_tokenizer,
        know_generation_model=know_generation_model,
        know_tokenizer=know_tokenizer,
        terminal_act=terminal_act,
        horizon=horizon,
        reward_func=reward_func_,
        goal2id=goal2id,
        device=device,
        max_sequence_length=max_sequence_length,
        max_gen_length=max_gen_length,
    )

    default_policy = OfflinePolicy(
        env=env,
        horizon=horizon,
        generation_model=generation_model,
        generation_tokenizer=generation_tokenizer,
        know_generation_model=know_generation_model,
        know_tokenizer=know_tokenizer,
        policy_model=policy_model,
        policy_tokenizer=policy_tokenizer,
        max_sequence_length=max_sequence_length,
        max_gen_length=max_gen_length,
        generation_args=model_generation_args,
        goal2id=goal2id,
        terminated_act=terminal_act,
        device=device
    )

    agent = uct.UCT(
        default_policy=default_policy,
        **uct_args
    )

    id2goal = {v: k for k, v in goal2id.items()}

    # Run
    def generate(initial_state):
        env.reset(initial_state)
        # do all rollouts in one step
        env.step(agent.act(env, done=False))
        # print tree
        # print_tree(agent.root, policy_tokenizer)
        # optionally, plot the tree and save to a pdf file
        if should_plot_tree:
            # plot (and print) the tree
            from dyna_gym.utils.tree_search_utils import plot_tree
            filename = f"tree-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            plot_tree(agent.root, policy_tokenizer, filename)
            print(f"Tree plotted and saved to {filename}.pdf")

        optimal_action = id2goal[agent.opt_act]
        # clear for the next generation call
        agent.reset()

        return optimal_action

    return generate
