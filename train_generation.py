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

from dyna_gym.models.policy import PolicyModel, save_model
from dataset.base import BaseTorchDataset
from dataset.durecdial import DuRecdial
from eval.eval_generation import GenerationEvaluator
from config.config import special_tokens_dict
from dataset.data_utils import convert_example_to_feature_for_response_generation, load_policy_results, merge_predictions


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
    parser.add_argument('--goal_outpath', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_gen_length', default=50, type=int, help="max length of both encoder and decoder input.")
    # model
    parser.add_argument("--plm_model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--lm_size", type=int)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
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
    goal2id = {k: v for v, k in enumerate(dataset.goals)}

    # load goal predictions
    dev_pred_goals = load_policy_results(os.path.join(args.goal_outpath, "dev_policy.txt"))
    test_pred_goals = load_policy_results(os.path.join(args.goal_outpath, "test_policy.txt"))

    # merge predictions
    dataset.dev_instances = merge_predictions(dataset.dev_instances, dev_pred_goals)
    dataset.test_instances = merge_predictions(dataset.test_instances, test_pred_goals)

    # t5 as the response generation model
    model = T5ForConditionalGeneration.from_pretrained(args.plm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
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
    train_torch_dataset = BaseTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.train_instances,
        goal2id=goal2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_response_generation,
        is_test=False,
        is_gen=True,
        max_target_length=args.max_target_length
    )
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

    train_dataloader = DataLoader(
        train_torch_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_torch_dataset.collate_fn,
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

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_torch_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

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

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        model.train()
        for step, batch in enumerate(train_dataloader):
            loss = model(**batch['context'], labels=batch['labels'], return_dict=True)['loss']
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))
            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        del train_loss, batch

        # dev
        valid_loss = []
        model.eval()
        for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                outputs = model(**batch['context'], labels=batch['labels'], return_dict=True)
                loss = outputs['loss']
                logits = outputs['logits']
                valid_loss.append(float(loss))
                evaluator.evaluate(logits, batch['labels'])

            gen_seqs = accelerator.unwrap_model(model).generate(
                **batch['context'],
                max_new_tokens=args.max_gen_length,
                no_repeat_ngram_size=3
            )
            gen_resp_ids = []
            for gen_seq in gen_seqs:
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq)
            evaluator.evaluate(gen_resp_ids, batch['labels'], log=accelerator.is_local_main_process)

        # metric
        accelerator.wait_for_everyone()
        report = evaluator.report()
        valid_report = {}
        for k, v in report.items():
            valid_report[f'valid/{k}'] = v
        valid_loss = np.mean(valid_loss)
        valid_report['valid/loss'] = valid_loss
        valid_report['epoch'] = epoch
        logger.info(valid_report)
        if run:
            run.log(valid_report)

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')
            save_model(model, output_dir=os.path.join(args.output_dir, 'policy.pth'))

        evaluator.reset_metric()
        # test
        test_loss = []
        model.eval()
        for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                outputs = model(**batch['context'], labels = batch['labels'], return_dict = True)
                loss = outputs['loss']
                logits = outputs['logits']
                test_loss.append(float(loss))
                evaluator.evaluate(logits, batch['labels'])

            gen_seqs = accelerator.unwrap_model(model).generate(
                **batch['context'],
                max_new_tokens=args.max_gen_length,
                no_repeat_ngram_size=3
            )
            gen_resp_ids = []
            for gen_seq in gen_seqs:
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq)
            evaluator.evaluate(gen_resp_ids, batch['labels'], log=accelerator.is_local_main_process)

        # metric
        accelerator.wait_for_everyone()
        report = evaluator.report()
        test_report = {}
        for k, v in report.items():
            test_report[f'test/{k}'] = v
        test_loss = np.mean(test_loss)
        test_report['test/loss'] = test_loss
        test_report['epoch'] = epoch
        logger.info(test_report)
        if run:
            run.log(test_report)
        evaluator.reset_metric()
