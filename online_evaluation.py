import math
import os
import argparse

import torch
import transformers
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration

from dyna_gym.pipelines import uct_for_dialogue_planning_pipeline
from dyna_gym.models.policy import PolicyModel, load_model
from dataset.durecdial import DuRecdial
from config.config import special_tokens_dict, DURECDIALGOALS
from dataset.data_utils import create_target_set, load_binary_file, save_binary_file

from dyna_gym.envs.utils import reward_func, random_seed
from eval.mcts_eval_online import MCTSCRSOnlineEval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument('--num_items', default=10, type=int, help="max length of both encoder and decoder input.")
    parser.add_argument("--train_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--max_sequence_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_gen_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--horizon', type=int, default=5, help="max length of both encoder and decoder input.")
    parser.add_argument('--rollouts', type=int, default=20, help="number of rollout in MCT")
    parser.add_argument('--width', type=int, default=3, help="abc")
    parser.add_argument('--gamma', type=float, default=1., help="abc")
    parser.add_argument('--alg', type=str, default='p_uct', help="criterion for the selection step")
    parser.add_argument('--policy_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--generation_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--know_generation_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--target_set_path', type=str, help="criterion for the selection step")
    # model
    parser.add_argument("--plm_policy_model", type=str)
    parser.add_argument("--policy_tokenizer", type=str)
    parser.add_argument("--plm_generation_model", type=str)
    parser.add_argument("--generation_tokenizer", type=str)
    parser.add_argument("--plm_know_generation_model", type=str)
    parser.add_argument("--know_generation_tokenizer", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--offline_policy", action="store_true", help="whether to use offline policy")
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

    random_seed(args.seed)

    device = torch.device('cuda:0')
    # arguments for the UCT agent
    uct_args = dict(
        rollouts=args.rollouts,
        gamma=args.gamma,
        width=args.width,
        alg=args.alg,  # or p_uct
    )

    # will be passed to huggingface model.generate()
    model_generation_args = dict()

    plm_policy_model = args.plm_policy_model
    policy_model_path = args.policy_model_path
    policy_model_name = 'policy.pth'
    lm_size = args.lm_size
    hidden_size = args.hidden_size

    dataset = DuRecdial(
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        test_data_path=args.test_data_path,
        save_train_convs=True  # for demonstration retrieval
    )
    # goal2id = {k: v for v, k in enumerate(DURECDIALGOALS)}
    goal2id = load_binary_file(os.path.join(policy_model_path, "goal2id.pkl"))

    # create and load the weights for policy model
    policy_plm = AutoModel.from_pretrained(plm_policy_model)
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer)
    policy_tokenizer.add_special_tokens(special_tokens_dict)
    policy_plm.resize_token_embeddings(len(policy_tokenizer))

    policy_model = PolicyModel(
        plm=policy_plm,
        n_goals=len(dataset.goals),
        hidden_size=args.hidden_size,
        lm_size=args.lm_size
    )

    policy_model = load_model(policy_model, os.path.join(policy_model_path, policy_model_name))
    policy_model.to(device)

    # create and load the weights for knowledge generation model
    plm_know_generation_model = args.plm_know_generation_model
    know_generation_model_path = args.know_generation_model_path
    know_generation_model_name = 'know_generation.pth'
    know_generation_model = BartForConditionalGeneration.from_pretrained(plm_know_generation_model)

    know_generation_tokenizer = AutoTokenizer.from_pretrained(args.know_generation_tokenizer)
    know_generation_tokenizer.add_special_tokens(special_tokens_dict)
    know_generation_model.resize_token_embeddings(len(know_generation_tokenizer))
    know_generation_model = load_model(know_generation_model,
                                       os.path.join(know_generation_model_path, know_generation_model_name))
    know_generation_model.to(device)

    # create and load the weights for generation model
    plm_generation_model = args.plm_generation_model
    generation_model_path = args.generation_model_path
    generation_model_name = 'response_generation.pth'
    generation_model = BartForConditionalGeneration.from_pretrained(plm_generation_model)

    generation_tokenizer = AutoTokenizer.from_pretrained(args.generation_tokenizer)
    generation_tokenizer.add_special_tokens(special_tokens_dict)
    generation_model.resize_token_embeddings(len(generation_tokenizer))
    generation_model = load_model(generation_model, os.path.join(generation_model_path, generation_model_name))

    generation_model.to(device)

    if not os.path.exists(args.target_set_path):
        os.mkdir(args.target_set_path)

    if os.path.exists(os.path.join(args.target_set_path, "target.pkl")):
        target_set = load_binary_file(os.path.join(args.target_set_path, "target.pkl"))
    else:
        # create the target item set.
        target_set = create_target_set(dataset.train_convs, dataset.test_instances, num_items=args.num_items)
        save_binary_file(target_set, os.path.join(args.target_set_path, "target.pkl"))

    terminal_act = "Say goodbye"

    mcts_online_eval = MCTSCRSOnlineEval(
        target_set=target_set,
        terminal_act=terminal_act,
        generation_model=generation_model,
        generation_tokenizer=generation_tokenizer,
        know_generation_model=know_generation_model,
        know_generation_tokenizer=know_generation_tokenizer,
        policy_model=policy_model,
        policy_tokenizer=policy_tokenizer,
        horizon=args.horizon,
        reward_func=reward_func,
        uct_args=uct_args,
        goal2id=goal2id,
        device=device,
        offline_policy=args.offline_policy,
        max_sequence_length=args.max_sequence_length,
        max_gen_length=args.max_gen_length,
        model_generation_args=model_generation_args,
        should_plot_tree=True,  # plot the tree after generation
    )

    # compute online evaluation metrics
    sr, avg_turn = mcts_online_eval.eval()

    print("Success rate:", sr)
    print("Avg turn: ", avg_turn)
