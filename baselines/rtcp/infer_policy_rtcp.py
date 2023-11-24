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
import itertools

# from dyna_gym.models.policy import PolicyModel, save_model
from dyna_gym.models.policy import load_model
# from dataset.base import BaseTorchDataset
from dataset.datasets import RTCPTorchDataset
from baselines.rtcp.policy import PolicyModel

from dataset.durecdial import DuRecdial
from eval.eval_policy import PolicyEvaluator
from config.config import special_tokens_dict, DURECDIALGOALS

from baselines.rtcp.utils import convert_example_to_feature_for_rtcp_goal_topic_prediction
from dataset.data_utils import save_binary_file, load_binary_file, save_knowledge_results


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
    parser.add_argument("--ffn_size", type=int)
    parser.add_argument("--fc_size", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)

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

    # goal2id = {k: v for v, k in enumerate(dataset.goals)}
    # topic2id = {k: v for v, k in enumerate(dataset.topics)}

    goal2id = load_binary_file(os.path.join(args.output_dir, 'rtcp_goal2id.pkl'))
    topic2id = load_binary_file(os.path.join(args.output_dir, 'rtcp_topic2id.pkl'))

    # switch from predicting a goal to predicting a pair of a goal and a topic
    # goal2id = itertools.product(dataset.goals, dataset.topics)
    # goal2id = {k: v for v, k in enumerate(goal2id)}

    context_encoder = AutoModel.from_pretrained(args.plm_model)
    path_encoder = AutoModel.from_pretrained(args.plm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    tokenizer.add_special_tokens(special_tokens_dict)
    context_encoder.resize_token_embeddings(len(tokenizer))
    path_encoder.resize_token_embeddings(len(tokenizer))

    model = PolicyModel(
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
    model.to(device)
    model = load_model(model, os.path.join(args.output_dir, "policy.pth"))

    # optim & amp

    dev_torch_dataset = RTCPTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.dev_instances,
        goal2id=goal2id,
        topic2id=topic2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_rtcp_goal_topic_prediction
    )
    test_torch_dataset = RTCPTorchDataset(
        tokenizer=tokenizer,
        instances=dataset.test_instances,
        goal2id=goal2id,
        topic2id=topic2id,
        max_sequence_length=args.max_sequence_length,
        device=device,
        convert_example_to_feature=convert_example_to_feature_for_rtcp_goal_topic_prediction
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

    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num dev examples = {len(dev_torch_dataset)}")
    logger.info(f"  Num test examples = {len(test_torch_dataset)}")
    # Only show the progress bar once on each machine.

    evaluator = PolicyEvaluator()
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
    valid_goal_preds = []
    valid_topic_preds = []
    model.eval()
    for batch in tqdm(valid_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(batch)
            loss = outputs['loss']
            valid_loss.append(float(loss))
            # only compute the topic accuracy
            evaluator.evaluate(outputs['topic_logits'], batch['labels_topic'])
            goal_pred_classes = outputs['goal_logits'].argmax(dim=-1)
            topic_pred_classes = outputs['topic_logits'].argmax(dim=-1)
            goal_pred_classes = goal_pred_classes.detach().cpu().numpy().tolist()
            topic_pred_classes = topic_pred_classes.detach().cpu().numpy().tolist()
            valid_goal_preds.extend(goal_pred_classes)
            valid_topic_preds.extend(topic_pred_classes)

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
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
    test_goal_preds = []
    test_topic_preds = []
    test_goal_labels = []
    test_topic_labels = []
    model.eval()
    for batch in tqdm(test_dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            outputs = model(batch)
            loss = outputs['loss']
            test_loss.append(float(loss))

            # preds
            goal_pred_classes = outputs['goal_logits'].argmax(dim=-1)
            topic_pred_classes = outputs['topic_logits'].argmax(dim=-1)
            goal_pred_classes = goal_pred_classes.detach().cpu().numpy().tolist()
            topic_pred_classes = topic_pred_classes.detach().cpu().numpy().tolist()
            test_goal_preds.extend(goal_pred_classes)
            test_topic_preds.extend(topic_pred_classes)

            # label
            test_goal_labels.extend(batch['labels_goal'].detach().cpu().numpy().tolist())
            test_topic_labels.extend(batch['labels_topic'].detach().cpu().numpy().tolist())

    goal_p, goal_r, goal_f = PolicyEvaluator.compute_precision_recall_f1_metrics(test_goal_preds, test_goal_labels)
    topic_p, topic_r, topic_f = PolicyEvaluator.compute_precision_recall_f1_metrics(test_topic_preds, test_topic_labels)

    id2goal = {v: k for k, v in goal2id.items()}
    id2topic = {v: k for k, v in topic2id.items()}

    # convert goal, topic indices to categories
    valid_goal_preds = [id2goal[x] for x in valid_goal_preds]
    test_goal_preds = [id2goal[x] for x in test_goal_preds]
    valid_topic_preds = [id2topic[x] for x in valid_topic_preds]
    test_topic_preds = [id2topic[x] for x in test_topic_preds]

    # # save policy results
    # save_knowledge_results(valid_goal_preds, os.path.join(args.output_dir, "dev_goal.txt"))
    # save_knowledge_results(test_goal_preds, os.path.join(args.output_dir, "test_goal.txt"))
    #
    # save_knowledge_results(valid_topic_preds, os.path.join(args.output_dir, "dev_topic.txt"))
    # save_knowledge_results(test_topic_preds, os.path.join(args.output_dir, "test_topic.txt"))

    save_binary_file(valid_goal_preds, os.path.join(args.output_dir, 'dev_goal.pkl'))
    save_binary_file(test_goal_preds, os.path.join(args.output_dir, 'test_goal.pkl'))
    save_binary_file(valid_topic_preds, os.path.join(args.output_dir, 'dev_topic.pkl'))
    save_binary_file(test_topic_preds, os.path.join(args.output_dir, 'test_topic.pkl'))

    logger.info('Save predictions successfully')
    logger.info(f'Task: [Goal], precision: {goal_p}, recall: {goal_r}, f1: {goal_f}')
    logger.info(f'Task: [Topic], precision: {topic_p}, recall: {topic_r}, f1: {topic_f}')
