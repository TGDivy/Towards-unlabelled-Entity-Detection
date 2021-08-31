if True: #Group imports
    import logging
    import os
    import random
    import shutil
    import json
    import sys
    import pickle

    import numpy as np
    import sklearn
    import torch

    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    from torch.utils.data.distributed import DistributedSampler
    import torch.nn.functional as F

    from tqdm.notebook import tqdm, trange

    from display import Display
#     from display import get_spans, render_entities
#     from torch.utils.tensorboard import SummaryWriter

    logger = logging.getLogger(__name__)

def to_device(args, batch):
    batch["inputs"] = tuple(input_tensor.to(args.device) for input_tensor in batch["inputs"])
    batch["labels"] = batch["labels"].to(args.device)

    return batch

def run_batch_selection_eval(args, model, batch, tokenizer, display):
    batch = to_device(args, batch)    
    loss, logits = model(batch, train=False)
    prediction = (logits>0.5)*1 + (logits>1.5)*1  + (logits>2.5)*1
    
    total = torch.sum(batch["inputs"][1])
        
    total_correct = torch.sum((prediction== batch["labels"])[0][:total], dtype=torch.float)
    
    if total_correct<total:
        text, spans = display.get_spans(
            tokenizer,
            batch["data_info"]["tokenized_sentences"][0],
            batch["data_info"]["subtoken_maps"][0],
            prediction[0],
            logits[0],
            batch["labels"][0],
        )
        display.render_entities(text, spans)
    return loss, total_correct, total

def evaluate(args, eval_dataset, model, tokenizer, desc="", epoch=0):
    print("Start Evaluation!")
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)
    
    model.eval()
    
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (args.task != "selection" or eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    
    accuracy_dic = {
        "total":0,
        "R1": 0, 
        }
    
    prediction_dic = {}
    display = Display()
    
    epoch_iterator = tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0], ncols=90)
    for batch in epoch_iterator:
        with torch.no_grad():
            loss, total_correct, total = run_batch_selection_eval(args, model, batch, tokenizer, display)
            accuracy_dic["total"] +=total
            accuracy_dic["R1"] +=total_correct
            eval_loss+=loss
            epoch_iterator.set_postfix(Accuracy=(accuracy_dic["R1"]/accuracy_dic["total"]))
        
        nb_eval_steps += 1

    return accuracy_dic