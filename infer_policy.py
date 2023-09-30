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
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel

from dyna_gym.models.policy import PolicyModel, load_model
from dataset.base import BaseTorchDataset
from dataset.durecdial import DuRecdial
from eval.eval_policy import PolicyEvaluator
from config.config import special_tokens_dict
from dataset.data_utils import convert_example_to_feature_for_goal_prediction, save_policy_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--train_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--max_sequence_length', type=int, help="max length of both encoder and decoder input.")
    # model
    parser.add_argument("--plm_model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--lm_size", type=int)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")

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
    goal2id = {k: v for v, k in enumerate(dataset.goals)}
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
    model = load_model(model, os.path.join(args.output_dir, 'policy.pth'))
    model.to(device)

    # data
    dev_torch_dataset = BaseTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.dev_instances,
        goal2id=goal2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_goal_prediction
    )
    test_torch_dataset = BaseTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.test_instances,
        goal2id=goal2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_goal_prediction
    )

    valid_dataloader = DataLoader(
        dev_torch_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_torch_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_torch_dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=test_torch_dataset.collate_fn,
    )

    model, valid_dataloader, test_dataloader = accelerator.prepare(model, valid_dataloader, test_dataloader)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num dev examples = {len(dev_torch_dataset)}")
    logger.info(f"  Num test examples = {len(dev_torch_dataset)}")
    # Only show the progress bar once on each machine.
    evaluator = PolicyEvaluator()

    # dev
    valid_preds = []
    model.eval()
    for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            logits = model(batch['context'])
            evaluator.evaluate(logits, batch['labels'])
            pred_classes = logits.argmax(dim=-1)
            pred_classes = pred_classes.detach().cpu().numpy().tolist()
            valid_preds.extend(pred_classes)

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
    valid_report = {}
    for k, v in report.items():
        valid_report[f'valid/{k}'] = v
    logger.info(valid_report)
    if run:
        run.log(valid_report)

    evaluator.reset_metric()
    # test
    test_preds = []
    model.eval()
    for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            logits = model(batch['context'])
            evaluator.evaluate(logits, batch['labels'])
            pred_classes = logits.argmax(dim=-1)
            pred_classes = pred_classes.detach().cpu().numpy().tolist()
            test_preds.extend(pred_classes)

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'test/{k}'] = v
    logger.info(test_report)
    if run:
        run.log(test_report)
    evaluator.reset_metric()

    # save results
    save_policy_results(valid_preds, os.path.join(args.output_dir, "dev_policy.txt"))
    save_policy_results(test_preds, os.path.join(args.output_dir, "test_policy.txt"))
    logger.info('Save predictions successfully')
