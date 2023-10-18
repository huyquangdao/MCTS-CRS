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
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BartForConditionalGeneration, \
    T5ForConditionalGeneration

from dyna_gym.models.policy import load_model
from dataset.datasets import UnimindTorchDataset
from dataset.durecdial import DuRecdial
from eval.eval_generation import GenerationEvaluator
from eval.eval_policy import PolicyEvaluator
from config.config import special_tokens_dict
from dataset.data_utils import save_knowledge_results, load_binary_file
from baselines.unimind.utils import convert_example_to_feature_for_unimind_goal_prediction, \
    convert_example_to_feature_for_unimind_topic_prediction, convert_example_to_feature_for_unimind_response_generation


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
    parser.add_argument('--max_target_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_gen_length', default=50, type=int, help="max length of both encoder and decoder input.")
    # model
    parser.add_argument("--plm_model", type=str)
    parser.add_argument("--tokenizer", type=str)
    # optim
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
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

    task = "goal"

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
    # goal2id = {k: v for v, k in enumerate(dataset.goals)}
    # UNIMIND utilizes BART as the response generation model
    model = BartForConditionalGeneration.from_pretrained(args.plm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model = load_model(model, os.path.join(args.output_dir, task, "unimind.pth"))
    model.to(device)

    input_transformation_dict = {
        "goal": convert_example_to_feature_for_unimind_goal_prediction,
        "topic": convert_example_to_feature_for_unimind_topic_prediction,
        "response": convert_example_to_feature_for_unimind_response_generation,
    }

    # data for goal prediction
    dev_torch_dataset = UnimindTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.dev_instances,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=input_transformation_dict['goal'],
        is_test=True,
        is_gen=True,
        max_target_length=args.max_target_length
    )
    test_torch_dataset = UnimindTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.test_instances,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=input_transformation_dict['goal'],
        is_test=True,
        is_gen=True,
        max_target_length=args.max_target_length
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
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num dev examples = {len(dev_torch_dataset)}")
    logger.info(f"  Num test examples = {len(test_torch_dataset)}")

    # Only show the progress bar once on each machine.
    gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
    evaluator = GenerationEvaluator(tokenizer, log_file_path=gen_file_path)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # dev
    valid_loss = []
    model.eval()
    for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch['context'], labels=batch['labels'], return_dict=True)
            loss = outputs['loss']
            logits = outputs['logits']
            valid_loss.append(float(loss))

        gen_seqs = accelerator.unwrap_model(model).generate(
            **batch['context'],
            max_new_tokens=args.max_gen_length,
            no_repeat_ngram_size=3
        )
        gen_resp_ids = []
        for gen_seq in gen_seqs:
            gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
            gen_resp_ids.append(gen_seq)

        label_resp_ids = []
        for label_seq in batch['labels']:
            label_seq = [token_id for token_id in label_seq if token_id != -100]
            label_resp_ids.append(label_seq)
        evaluator.evaluate(gen_resp_ids, label_resp_ids, log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    _, valid_preds, valid_labels = evaluator.report()

    # post processing labels
    valid_labels = [x.split(":")[-1].strip() for x in valid_labels]
    valid_preds = [x.split(":")[-1].strip() for x in valid_preds]
    goal_acc = PolicyEvaluator.compute_categorical_acc(valid_preds, valid_labels)
    logger.info({"goal_accuracy": goal_acc})
    if run:
        run.log({"goal_accuracy": goal_acc})
    evaluator.reset_metric()

    # test
    test_loss = []
    test_preds = []
    model.eval()
    for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(**batch['context'], labels=batch['labels'], return_dict=True)
            loss = outputs['loss']
            logits = outputs['logits']
            test_loss.append(float(loss))

        gen_seqs = accelerator.unwrap_model(model).generate(
            **batch['context'],
            max_new_tokens=args.max_gen_length,
            no_repeat_ngram_size=3
        )
        gen_resp_ids = []
        for gen_seq in gen_seqs:
            gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
            gen_resp_ids.append(gen_seq)

        label_resp_ids = []
        for label_seq in batch['labels']:
            label_seq = [token_id for token_id in label_seq if token_id != -100]
            label_resp_ids.append(label_seq)
        evaluator.evaluate(gen_resp_ids, label_resp_ids, log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    _, test_preds, test_labels = evaluator.report()
    test_labels = [x.split(":")[-1].strip() for x in test_labels]
    test_preds = [x.split(":")[-1].strip() for x in test_preds]
    goal_acc = PolicyEvaluator.compute_categorical_acc(test_preds, test_labels)
    logger.info({"goal_accuracy": goal_acc})
    if run:
        run.log({"goal_accuracy": goal_acc})
    evaluator.reset_metric()

    # save the predictions
    save_knowledge_results(valid_preds, os.path.join(args.output_dir, task, "dev_goal.txt"))
    save_knowledge_results(test_preds, os.path.join(args.output_dir, task, "test_goal.txt"))
