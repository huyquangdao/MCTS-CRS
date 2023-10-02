import os
import argparse

import transformers
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer

from dyna_gym.pipelines import uct_for_dialogue_planning_pipeline
from dyna_gym.models.policy import PolicyModel, load_model
from dataset.durecdial import DuRecdial
from config.config import special_tokens_dict, DURECDIALGOALS
from dataset.data_utils import randomly_sample_demonstrations


# define a reward function based on sentiment of the generated text
def reward_func(sentence, target):
    if target.lower() in sentence.lower():
        return 3.0
    else:
        return -3.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--train_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--max_sequence_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--horizon', type=int, default=5, help="max length of both encoder and decoder input.")
    parser.add_argument('--rollouts', type=int, default=20, help="number of rollout in MCT")
    parser.add_argument('--width', type=int, default=3, help="abc")
    parser.add_argument('--gamma', type=float, default=1., help="abc")
    parser.add_argument('--alg', type=str, default='p_uct', help="criterion for the selection step")
    parser.add_argument('--model_path', type=str, help="criterion for the selection step")
    # model
    parser.add_argument("--plm_model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--lm_size", type=int)
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse argments
    args = parse_args()

    # arguments for the UCT agent
    uct_args = dict(
        rollouts=args.rollouts,
        gamma=args.gamma,
        width=args.width,
        alg=args.alg,  # or p_uct
    )

    # will be passed to huggingface model.generate()
    model_generation_args = dict()

    plm_model = args.plm_model
    model_path = args.model_path
    model_name = 'policy.pth'
    lm_size = args.lm_size
    hidden_size = args.hidden_size

    dataset = DuRecdial(
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        test_data_path=args.test_data_path,
        save_train_convs=True  # for demonstration retrieval
    )
    goal2id = {k: v for v, k in enumerate(DURECDIALGOALS)}
    plm = AutoModel.from_pretrained(args.plm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)
    plm.resize_token_embeddings(len(tokenizer))

    model = PolicyModel(
        plm=plm,
        n_goals=len(dataset.goals),
        hidden_size=args.hidden_size,
        lm_size=args.lm_size
    )

    model = load_model(model, os.path.join(model_path, model_name))
    pipeline = uct_for_dialogue_planning_pipeline(
        policy_model=model,
        tokenizer=tokenizer,
        horizon=args.horizon,
        reward_func=reward_func,
        uct_args=uct_args,
        max_sequence_length=args.max_sequence_length,
        model_generation_args=model_generation_args,
        should_plot_tree=True,  # plot the tree after generation
    )

    initial_state = dataset.test_instances[0]
    # sample a demonstration for user simulator:
    demonstrations = randomly_sample_demonstrations(
        all_convs=dataset.train_convs,
        instance=initial_state
    )
    initial_state['demonstration'] = demonstrations[0]

    outputs = pipeline(initial_state=initial_state)
