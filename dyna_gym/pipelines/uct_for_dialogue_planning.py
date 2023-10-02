from datetime import datetime
from typing import Callable, Sequence

import gym
import torch
import transformers

from dyna_gym.agents import uct
from dyna_gym.default_policy.offline_policy import OfflinePolicy
from dyna_gym.utils.tree_search_utils import print_tree


def uct_for_dialogue_planning_pipeline(
        policy_model,
        tokenizer: transformers.PreTrainedTokenizer,
        horizon: int = 5,
        terminal_act: str = 'Say goodbye',
        max_sequence_length = 512,
        reward_func: Callable = None,
        uct_args: dict = {},
        model_generation_args: dict = {},
        should_plot_tree: bool = False,
        reward_func_input_is_state: bool = False,
) -> Callable:
    """
    A wrapped UCT agent for HuggingFace transformer.

    Args:
        model_name: The name of a HuggingFace transformer model. If provided, will load the model and tokenizer.
        model: A HuggingFace transformer model.
        tokenizer: A HuggingFace tokenizer.
        horizon: The maximum number of steps to take.
        reward_func: A function that evaluate the reward of a sequence.
        value_func: A function that evaluate the value of a sequence.
        uct_args: Arguments for the UCT agent.
        model_generation_args: Arguments for the model generation.
        should_plot_tree: Whether to plot the tree after generation.
        reward_func_input_is_state: Whether the input of the reward function is (token ids, attention masks) or tokenized text.
    """
    reward_func_ = reward_func
    env = gym.make(
        'DialogueEnv-v0',
        terminal_token=terminal_act,
        horizon=horizon,
        reward_func=reward_func_,
    )

    default_policy = OfflinePolicy(
        env=env,
        horizon=horizon,
        policy_model=policy_model,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        generation_args=model_generation_args,
    )

    agent = uct.UCT(
        default_policy=default_policy,
        **uct_args
    )

    # Run
    def generate(initial_state):

        env.reset(initial_state)
        # do all rollouts in one step
        env.step(agent.act(env, done=False))
        # print tree
        print_tree(agent.root, tokenizer)
        # optionally, plot the tree and save to a pdf file
        if should_plot_tree:
            # plot (and print) the tree
            from dyna_gym.utils.tree_search_utils import plot_tree
            filename = f"tree-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            plot_tree(agent.root, tokenizer, filename)
            print(f"Tree plotted and saved to {filename}.pdf")

        results = {
            'output_ids': agent.rolled_out_trajectories,
            'rewards': agent.rolled_out_rewards,
        }
        # clear for the next generation call
        agent.reset()
        return results

    return generate
