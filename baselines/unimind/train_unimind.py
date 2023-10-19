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

from dyna_gym.models.policy import save_model, load_model
from dataset.durecdial import DuRecdial
from eval.eval_generation import GenerationEvaluator
from eval.eval_policy import PolicyEvaluator
from config.config import special_tokens_dict
from baselines.unimind.utils import convert_example_to_feature_for_unimind_goal_prediction, \
    convert_example_to_feature_for_unimind_topic_prediction, convert_example_to_feature_for_unimind_response_generation

from baselines.unimind.utils import train_unimind, evaluate_unimind, construct_task_torchdatasets


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
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--num_finetune_epochs", type=int, default=5,
                        help="Total number of training epochs to perform.")

    #
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--max_finetune_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')

    parser.add_argument("--do_train", action="store_true", help="whether to use wandb")
    parser.add_argument("--do_finetune", action="store_true", help="whether to use wandb")

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
        train_data_path=
        args.train_data_path,
        dev_data_path=args.dev_data_path,
        test_data_path=args.test_data_path
    )
    goal2id = {k: v for v, k in enumerate(dataset.goals)}

    # UNIMIND utilizes bart as the pretrained backbone
    model = BartForConditionalGeneration.from_pretrained(args.plm_model)
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

    input_transformation_dict = {
        "goal": convert_example_to_feature_for_unimind_goal_prediction,
        "topic": convert_example_to_feature_for_unimind_topic_prediction,
        "response": convert_example_to_feature_for_unimind_response_generation,
    }

    gen_file_path = os.path.join('log', f'gen_{local_time}.jsonl')
    evaluator = GenerationEvaluator(tokenizer, log_file_path=gen_file_path)

    # train loop
    if args.do_train:
        # construct datasets by using full goal, topic and response instances.
        train_torch_dataset, dev_torch_dataset, test_torch_dataset = construct_task_torchdatasets(
            args=args,
            tokenizer=tokenizer,
            dataset=dataset,
            input_transformation_dict=input_transformation_dict,
            goal2id=goal2id,
            device=device,
            task=None,
            is_test=False,
            is_gen=True
        )

        # dataloader
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

        # save model with best metric
        metric, mode = 'loss', -1
        assert mode in (-1, 1)
        if mode == 1:
            best_metric = 0
        else:
            best_metric = float('inf')
        best_metric_dir = os.path.join(args.output_dir, 'best')
        os.makedirs(best_metric_dir, exist_ok=True)

        for epoch in range(args.num_train_epochs):
            train_loss = train_unimind(
                args=args,
                model=model,
                accelerator=accelerator,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                lr_scheduler=lr_scheduler,
                run=run,
                progress_bar=progress_bar,
                completed_steps=completed_steps

            )
            logger.info(f'epoch {epoch} train loss {train_loss}')

            valid_loss, _, _ = evaluate_unimind(
                args=args, accelerator=accelerator, model=model, tokenizer=tokenizer, valid_dataloader=valid_dataloader,
                evaluator=evaluator
            )
            evaluator.reset_metric()

            # save the model with the best valid loss
            if valid_loss < best_metric:
                save_model(model, output_dir=os.path.join(args.output_dir, 'unimind.pth'))
                best_metric = valid_loss

    # fine tuning stage.
    if args.do_finetune:
        tasks = ["goal", "topic", "response"]
        # loop overall tasks
        for task in tasks:
            # construct datasets by using full goal, topic and response instances.
            train_torch_dataset, dev_torch_dataset, test_torch_dataset = construct_task_torchdatasets(
                args=args,
                tokenizer=tokenizer,
                dataset=dataset,
                input_transformation_dict=input_transformation_dict[task],
                goal2id=goal2id,
                device=device,
                task=task,
                is_gen=True,
                is_test=False
            )

            # dataloader
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
            # creating a model checkpoint for each task
            if not os.path.exists(os.path.join(args.output_dir, task)):
                os.mkdir(os.path.join(args.output_dir, task))

            # load model from checkpoint
            model = load_model(model, os.path.join(args.output_dir, 'unimind.pth'))
            model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
            # step, epoch, batch size
            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            if args.max_finetune_steps is None:
                args.max_finetune_steps = args.num_finetune_epochs * num_update_steps_per_epoch
            else:
                args.num_finetune_epochs = math.ceil(args.max_finetune_steps / num_update_steps_per_epoch)

            total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
            completed_steps = 0
            # lr_scheduler
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
            lr_scheduler = accelerator.prepare(lr_scheduler)
            local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

            progress_bar = tqdm(range(args.max_finetune_steps), disable=not accelerator.is_local_main_process)

            logger.info("***** Running Tuning Stage *****")
            logger.info(f"  Task = {task} generation")
            logger.info(f"  Num examples = {len(train_torch_dataset)}")
            logger.info(f"  Num Epochs = {args.num_finetune_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

            # reset best metric
            best_metric = float('inf')

            # finetune main loop
            for epoch in range(args.num_finetune_epochs):
                train_loss = []
                model.train()

                train_loss = train_unimind(
                    args=args,
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    lr_scheduler=lr_scheduler,
                    run=run,
                    progress_bar=progress_bar,
                    completed_steps=completed_steps

                )
                logger.info(f'epoch {epoch} train loss {train_loss}')
                valid_loss, valid_preds, valid_labels = evaluate_unimind(
                    args=args,
                    model=model,
                    accelerator=accelerator,
                    tokenizer=tokenizer,
                    valid_dataloader=valid_dataloader,
                    evaluator=evaluator
                )

                # save the model with the best valid loss
                if valid_loss < best_metric:
                    save_model(model, output_dir=os.path.join(args.output_dir, task, 'unimind.pth'))
                    best_metric = train_loss
