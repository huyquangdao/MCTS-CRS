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
from dataset.base import BaseTorchDataset
from dataset.durecdial import DuRecdial
from eval.eval_generation import GenerationEvaluator
from config.config import special_tokens_dict
from dataset.data_utils import convert_example_to_feature_for_response_generation, load_policy_results, \
    merge_predictions, load_binary_file, load_knowledge_results, merge_topic_predictions, merge_know_predictions, save_knowledge_results


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
    parser.add_argument('--goal_outpath', type=str, help="max length of both encoder and decoder input.")
    parser.add_argument('--know_outpath', type=str, help="max length of both encoder and decoder input.")
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
    goal2id = load_binary_file(os.path.join(args.goal_outpath, "goal2id.pkl"))

    # load goal predictions
    dev_pred_goals = load_policy_results(os.path.join(args.goal_outpath, "dev_goal.txt"))
    test_pred_goals = load_policy_results(os.path.join(args.goal_outpath, "test_goal.txt"))

    # load topic predictions
    dev_pred_topics = load_policy_results(os.path.join(args.goal_outpath, "dev_topic.txt"))
    test_pred_topics = load_policy_results(os.path.join(args.goal_outpath, "test_topic.txt"))

    # load knowledge predictions
    dev_pred_know = load_knowledge_results(os.path.join(args.know_outpath, "dev_knowledge.txt"))
    test_pred_know = load_knowledge_results(os.path.join(args.know_outpath, "test_knowledge.txt"))

    # merge goal predictions
    dataset.dev_instances = merge_predictions(dataset.dev_instances, dev_pred_goals)
    dataset.test_instances = merge_predictions(dataset.test_instances, test_pred_goals)

    # merge topic predictions
    dataset.dev_instances = merge_topic_predictions(dataset.dev_instances, dev_pred_topics)
    dataset.test_instances = merge_topic_predictions(dataset.test_instances, test_pred_topics)

    # merge know predictions
    dataset.dev_instances = merge_know_predictions(dataset.dev_instances, dev_pred_know)
    dataset.test_instances = merge_know_predictions(dataset.test_instances, test_pred_know)

    # bart as the response generation model
    model = BartForConditionalGeneration.from_pretrained(args.plm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model= load_model(model, os.path.join(args.output_dir, "response_generation.pth"))
    model.to(device)

    # data
    dev_torch_dataset = BaseTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.dev_instances,
        goal2id=goal2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_response_generation,
        is_test=True,
        is_gen=True,
        max_target_length=args.max_target_length
    )
    test_torch_dataset = BaseTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.test_instances,
        goal2id=goal2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_response_generation,
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
    report,test_decoded_preds, test_decoded_labels = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'test/{k}'] = v
    test_loss = np.mean(test_loss)
    test_report['test/loss'] = test_loss
    logger.info(test_report)
    if run:
        run.log(test_report)
    evaluator.reset_metric()

    # save test generated texts and labels
    save_knowledge_results(test_decoded_preds, os.path.join(args.output_dir, "test_gen.txt"))
    save_knowledge_results(test_decoded_labels, os.path.join(args.output_dir, "test_label.txt"))
