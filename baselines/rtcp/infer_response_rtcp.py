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
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, GPT2LMHeadModel, GPT2Config

from dyna_gym.models.policy import save_model, load_model
from baselines.rtcp.prefix_tuning import PrefixTuningTemplate
from baselines.rtcp.gen_model import PromptGPT2
from dataset.datasets import RTCPTorchDataset
from dataset.durecdial import DuRecdial
from eval.eval_generation import GenerationEvaluator
from config.config import special_tokens_dict, PAD_TOKEN
from baselines.rtcp.utils import convert_example_to_feature_for_rtcp_response_generation, sample_sequence
from dataset.data_utils import load_binary_file, load_policy_results, merge_predictions, merge_topic_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--goal_outpath", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
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

    # model
    parser.add_argument("--plm_model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--num_tokens", default=50, type=int)
    parser.add_argument("--n_goal_toks", default=2, type=int)
    parser.add_argument("--n_topic_toks", default=2, type=int)
    parser.add_argument("--use_goal_topic", action="store_true", help="whether to use wandb")
    parser.add_argument("--freeze_plm", action="store_true", help="whether to use wandb")

    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
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

    goal2id = load_binary_file(os.path.join(args.output_dir, 'rtcp_goal2id.pkl'))
    topic2id = load_binary_file(os.path.join(args.output_dir, 'rtcp_topic2id.pkl'))

    # load goal predictions
    dev_pred_goals = load_binary_file(os.path.join(args.goal_outpath, "dev_goal.pkl"))
    test_pred_goals = load_binary_file(os.path.join(args.goal_outpath, "test_goal.pkl"))

    dev_pred_topics = load_binary_file(os.path.join(args.goal_outpath, "dev_topic.pkl"))
    test_pred_topics = load_binary_file(os.path.join(args.goal_outpath, "test_topic.pkl"))

    # merge predictions
    dataset.dev_instances = merge_predictions(dataset.dev_instances, dev_pred_goals)
    dataset.test_instances = merge_predictions(dataset.test_instances, test_pred_goals)

    # merge predictions
    dataset.dev_instances = merge_topic_predictions(dataset.dev_instances, dev_pred_topics)
    dataset.test_instances = merge_topic_predictions(dataset.test_instances, test_pred_topics)

    # pad token for GPT2 and DialogGPT
    special_tokens_dict['pad_token'] = PAD_TOKEN
    model_name = "response_generation.pth"

    # t5 as the response generation model
    plm = GPT2LMHeadModel.from_pretrained(args.plm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)
    plm.resize_token_embeddings(len(tokenizer))

    plm_config = GPT2Config.from_pretrained(args.plm_model)
    prefix_model = PrefixTuningTemplate(
        config=plm_config,
        num_token=args.num_tokens,
        n_action_toks=args.n_goal_toks,
        n_topic_toks=args.n_topic_toks,
        use_goal_topic=args.use_goal_topic
    )

    model = PromptGPT2(plm=plm, prefix_model=prefix_model, freeze_plm=args.freeze_plm)
    model = load_model(model, os.path.join(args.output_dir, model_name))
    model.to(device)

    # optim & amp
    modules = [model]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # data

    dev_torch_dataset = RTCPTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.dev_instances,
        goal2id=goal2id,
        topic2id=topic2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_rtcp_response_generation,
        is_test=True,
        is_gen=True,
        max_target_length=args.max_target_length
    )
    test_torch_dataset = RTCPTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.test_instances,
        goal2id=goal2id,
        topic2id=topic2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_rtcp_response_generation,
        is_test=True,
        is_gen=True,
        max_target_length=args.max_target_length
    )

    valid_dataloader = DataLoader(
        dev_torch_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_torch_dataset.collate_fn,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_torch_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=test_torch_dataset.collate_fn,
        shuffle=False
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    logger.info("***** Running training *****")
    logger.info(f"  Num Dev examples = {len(dev_torch_dataset)}")
    logger.info(f"  Num Test examples = {len(test_torch_dataset)}")
    # Only show the progress bar once on each machine.

    gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
    evaluator = GenerationEvaluator(tokenizer, log_file_path=gen_file_path)

    # dev
    valid_loss = []
    model.eval()
    for instance in tqdm(dev_torch_dataset.instances, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            history = instance["input_ids"]
            action_id = instance["goal_id"]
            topic_id = instance["topic_id"]
            label_ids = instance["label"]

            # using RTCP's official decoding method
            gen_resp_ids = sample_sequence(model, history, action_id, topic_id, tokenizer, device=device,
                                           max_dec_len=args.max_gen_length, top_k=args.top_k, top_p=args.top_p,
                                           temperature=args.temperature)

            # evaluating the RTCP generation performance.
            evaluator.evaluate([gen_resp_ids], [label_ids], log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    report, valid_decoded_preds, valid_decoded_labels = evaluator.report()
    valid_report = {}
    for k, v in report.items():
        valid_report[f'valid/{k}'] = v
    valid_loss = np.mean(valid_loss)
    valid_report['valid/loss'] = valid_loss
    logger.info(valid_report)
    if run:
        run.log(valid_report)
    evaluator.reset_metric()

    # test
    test_loss = []
    model.eval()
    for instance in tqdm(test_torch_dataset.instances, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            history = instance["input_ids"]
            action_id = instance["goal_id"]
            topic_id = instance["topic_id"]
            label_ids = instance["label"]

            # using RTCP's official decoding method
            gen_resp_ids = sample_sequence(model, history, action_id, topic_id, tokenizer, device=device,
                                           max_dec_len=args.max_gen_length, top_k=args.top_k, top_p=args.top_p,
                                           temperature=args.temperature)

            # evaluating the RTCP generation performance.
            evaluator.evaluate([gen_resp_ids], [label_ids], log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    report, test_decoded_preds, test_decoded_labels = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'test/{k}'] = v
    test_loss = np.mean(test_loss)
    test_report['test/loss'] = test_loss
    logger.info(test_report)
    if run:
        run.log(test_report)
    evaluator.reset_metric()
