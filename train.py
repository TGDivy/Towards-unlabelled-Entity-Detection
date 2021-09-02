import logging
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import Dataset

logger = logging.getLogger(__name__)


def to_device(args, batch):
    batch["inputs"] = tuple(
        input_tensor.to(args.device) for input_tensor in batch["inputs"]
    )
    batch["labels"] = batch["labels"].to(args.device)

    return batch


def run_batch_selection_train(args, model, batch):
    inputs = to_device(args, batch)
    model_outputs = model(inputs)
    loss = model_outputs[0]
    return loss


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, tokenizer, model):
    train_dataset = Dataset(args, tokenizer=tokenizer, split_type="train")
    if args.local_rank in [-1, 0]:
        log_dir = (
            os.path.join("Outputs/Runs/Final_Run/", args.exp_name)
            if args.exp_name
            else None
        )
    #         tb_writer = SummaryWriter(log_dir)
    #         args.output_dir = log_dir

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    t_total = (
        len(train_dataloader)
        // args.gradient_accumulation_steps
        * args.num_train_epochs
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    global_step = 0
    model.zero_grad()
    train_iterator = trange(
        0 + 1,
        int(args.num_train_epochs) + 1,
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # for reproducibility
    maxr1 = 0
    maximize_attraction = 0
    for epoch in train_iterator:
        local_steps = 0
        tr_loss = 0.0

        epoch_iterator = tqdm(
            train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        )
        for step, batch in enumerate(epoch_iterator):

            model.train()
            loss = run_batch_selection_train(args, model, batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                local_steps += 1
                global_step += 1

                epoch_iterator.set_postfix(Loss=tr_loss / local_steps)

        train_dataset = Dataset(args, tokenizer=tokenizer, split_type="train")
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            collate_fn=train_dataset.collate_fn,
        )
    return model
