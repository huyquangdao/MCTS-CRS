import math
import os
import argparse

import torch
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration

from dyna_gym.models.policy import PolicyModel, load_model
from dataset.durecdial import DuRecdial
from config.config import special_tokens_dict, DURECDIALGOALS
from dataset.data_utils import create_target_set, load_binary_file, save_binary_file

from dyna_gym.envs.utils import reward_func, random_seed
from eval.unimind_online_eval import UnimindBartOnlineEval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument('--num_items', default=10, type=int, help="max length of both encoder and decoder input.")
    parser.add_argument("--train_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--max_sequence_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_gen_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--horizon', type=int, default=5, help="max length of both encoder and decoder input.")
    parser.add_argument('--model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--target_set_path', type=str, help="criterion for the selection step")

    parser.add_argument("--use_llm_score", action="store_true", help="whether to use llm based assessment")
    parser.add_argument("--n", default=5, type=int, help="whether to use llm based assessment")
    parser.add_argument("--k", default=3, type=int, help="whether to use llm based assessment")
    parser.add_argument("--epsilon", default=1.0, type=float, help="whether to use llm based assessment")
    parser.add_argument("--use_demonstration", action="store_true", help="whether to use llm based assessment")

    # model
    parser.add_argument("--policy_plm_model", type=str)
    parser.add_argument("--policy_model_path", type=str, help="Where to store the final model.")
    parser.add_argument("--policy_tokenizer", type=str)

    parser.add_argument('--generation_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--know_generation_model_path', type=str, help="criterion for the selection step")

    # know generation model
    parser.add_argument("--plm_know_generation_model", type=str)
    parser.add_argument("--know_generation_tokenizer", type=str)

    # generation model
    parser.add_argument("--plm_generation_model", type=str)
    parser.add_argument("--generation_tokenizer", type=str)

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

    # will be passed to huggingface model.generate()
    model_generation_args = dict()
    dataset = DuRecdial(
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        test_data_path=args.test_data_path,
        save_train_convs=True  # for demonstration retrieval
    )

    # create and load the weights for generation model
    plm_model = args.policy_plm_model
    model_path = args.model_path
    model_name = 'unimind.pth'
    model = BartForConditionalGeneration.from_pretrained(plm_model)

    tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # goal model
    goal_model = load_model(model, os.path.join(model_path, "goal", model_name))
    goal_model.to(device)

    # topic model
    topic_model = load_model(model, os.path.join(model_path, "topic", model_name))
    topic_model.to(device)

    # loading the knowledge generation model
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

    # loading the response generation model
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
    unimind_online_eval = UnimindBartOnlineEval(
        target_set=target_set,
        terminal_act=terminal_act,
        use_llm_score=args.use_llm_score,
        epsilon=args.epsilon,
        n=args.n,
        k=args.k,
        use_demonstration=args.use_demonstration,
        goal_model=goal_model,
        goal_model_tokenizer=tokenizer,
        topic_model=topic_model,
        topic_model_tokenizer=tokenizer,
        generation_model=generation_model,
        generation_tokenizer=generation_tokenizer,
        know_generation_model=know_generation_model,
        know_generation_tokenizer=know_generation_tokenizer,
        horizon=args.horizon,
        reward_func=reward_func,
        device=device,
        max_sequence_length=args.max_sequence_length,
        max_gen_length=args.max_gen_length
    )

    # policy model / target_set_id / text_generation_model/ generated_conversations.txt
    saved_file_path = os.path.join(args.policy_model_path, f"target_set_{args.seed}")
    if not os.path.exists(saved_file_path):
        os.mkdir(saved_file_path)

    saved_file_path = os.path.join(saved_file_path, "bart")
    if not os.path.exists(saved_file_path):
        os.mkdir(saved_file_path)
    saved_file_path = os.path.join(saved_file_path, "generated_conversations.txt")

    # compute online evaluation metrics
    srk, sr, avg_turn = unimind_online_eval.eval(saved_file_path=saved_file_path)

    print(f"success rate @ {args.k}: ", srk)
    print("Success rate:", sr)
    print("Avg turn: ", avg_turn)
