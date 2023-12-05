import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, GPT2LMHeadModel, GPT2Config, AutoModel, \
    BartForConditionalGeneration

from dyna_gym.models.policy import save_model, load_model
from baselines.rtcp.prefix_tuning import PrefixTuningTemplate
from baselines.rtcp.gen_model import PromptGPT2
from dataset.durecdial import DuRecdial
from config.config import special_tokens_dict, PAD_TOKEN
from dataset.data_utils import load_binary_file, create_target_set, save_binary_file
from eval.rtcp_online_eval import RTCPLLamaOnlineEval
from baselines.rtcp.policy import PolicyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--goal_outpath", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    parser.add_argument("--target_set_path", type=str, help="Where to store the final model.")

    # data
    parser.add_argument("--train_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_sequence_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_target_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_gen_length', default=50, type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--top_k', default=0, type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--top_p', default=0.0, type=float, help="max length of both encoder and decoder input.")
    parser.add_argument('--temperature', default=1, type=float, help="max length of both encoder and decoder input.")
    parser.add_argument('--num_items', default=10, type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--horizon', type=int, default=5, help="max length of both encoder and decoder input.")

    parser.add_argument("--use_llm_score", action="store_true", help="whether to use llm based assessment")
    parser.add_argument("--n", default=5, type=int, help="whether to use llm based assessment")
    parser.add_argument("--k", default=3, type=int, help="whether to use llm based assessment")
    parser.add_argument("--epsilon", default=1.0, type=float, help="whether to use llm based assessment")
    parser.add_argument("--use_demonstration", action="store_true", help="whether to use llm based assessment")

    # paths to pretrained models.
    parser.add_argument('--policy_model_path', type=str, help="criterion for the selection step")

    # policy moodel
    parser.add_argument("--plm_policy_model", type=str)
    parser.add_argument("--policy_tokenizer", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--lm_size", type=int)
    parser.add_argument("--ffn_size", type=int)
    parser.add_argument("--fc_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = DuRecdial(
        train_data_path=args.train_data_path,
        dev_data_path=args.dev_data_path,
        test_data_path=args.test_data_path
    )

    goal2id = load_binary_file(os.path.join(args.policy_model_path, 'rtcp_goal2id.pkl'))
    topic2id = load_binary_file(os.path.join(args.policy_model_path, 'rtcp_topic2id.pkl'))

    # load the policy model
    policy_model_name = "policy.pth"
    context_encoder = AutoModel.from_pretrained(args.plm_policy_model)
    path_encoder = AutoModel.from_pretrained(args.plm_policy_model)
    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer)

    policy_tokenizer.add_special_tokens(special_tokens_dict)
    context_encoder.resize_token_embeddings(len(policy_tokenizer))
    path_encoder.resize_token_embeddings(len(policy_tokenizer))

    policy_model = PolicyModel(
        context_encoder=context_encoder,
        path_encoder=path_encoder,
        knowledge_encoder=None,  # do not have knowledge,
        n_layers=args.n_layers,
        n_topics=len(dataset.topics),
        n_goals=len(dataset.goals),
        n_heads=args.n_heads,
        lm_hidden_size=args.lm_size,
        ffn_size=args.ffn_size,
        fc_hidden_size=args.fc_size
    )
    policy_model = load_model(policy_model, os.path.join(args.policy_model_path, policy_model_name))
    policy_model.to(device)

    if not os.path.exists(args.target_set_path):
        os.mkdir(args.target_set_path)

    if os.path.exists(os.path.join(args.target_set_path, "target.pkl")):
        target_set = load_binary_file(os.path.join(args.target_set_path, "target.pkl"))
    else:
        # create the target item set.
        target_set = create_target_set(dataset.train_convs, dataset.test_instances, num_items=args.num_items)
        save_binary_file(target_set, os.path.join(args.target_set_path, "target.pkl"))

    terminal_act = "Say goodbye"
    rtcp_online_eval = RTCPLLamaOnlineEval(
        target_set=target_set,
        terminal_act=terminal_act,
        use_llm_score=args.use_llm_score,
        epsilon=args.epsilon,
        n=args.n,
        k=args.k,
        use_demonstration=args.use_demonstration,
        policy_model=policy_model,
        policy_tokenizer=policy_tokenizer,
        horizon=args.horizon,
        goal2id=goal2id,
        topic2id=topic2id,
        device=device,
        max_sequence_length=args.max_sequence_length,
        max_gen_length=args.max_gen_length,
    )

    # policy model / target_set_id / text_generation_model/ generated_conversations.txt
    saved_file_path = os.path.join(args.policy_model_path, f"target_set_{args.seed}", "llama")
    if not os.path.exists(saved_file_path):
        os.mkdir(saved_file_path)
    saved_file_path = os.path.join(saved_file_path, "generated_conversations.txt")
    # compute online evaluation metrics
    srk, sr, avg_turn = rtcp_online_eval.eval(saved_file_path=saved_file_path)

    print(f"success rate @ {args.k}: ", srk)
    print("Success rate:", sr)
    print("Avg turn: ", avg_turn)
