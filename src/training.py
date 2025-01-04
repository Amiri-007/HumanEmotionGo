# src/training.py

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def loss_fn(outputs, targets, alpha=0.05):
    """
    Label smoothing: 
      smoothed = targets*(1-alpha)+(alpha/num_classes)
    """
    num_classes = targets.size(1)
    smooth_targets = targets*(1 - alpha) + (alpha / num_classes)
    return nn.BCEWithLogitsLoss()(outputs, smooth_targets)

def train_one_epoch(
    model, 
    train_loader, 
    optimizer, 
    scheduler, 
    device, 
    scaler, 
    epoch_idx, 
    epochs, 
    grad_accum=2, 
    is_main_process=True,
    precision="fp16",
    train_sampler=None
):
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch_idx)

    total_loss = 0.0
    steps = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{epochs}", disable=not is_main_process)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(progress_bar):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        # Mixed-precision context
        cast_enabled = (precision in ["fp16","bf16"])
        with torch.cuda.amp.autocast(enabled=cast_enabled):
            logits, _, _, _, _ = model(ids, mask)
            loss = loss_fn(logits, targets, alpha=0.05)
            loss = loss / grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum
        steps += 1

        if is_main_process and batch_idx % 10 == 0:
            progress_bar.set_postfix({'avg_loss': f'{total_loss/steps:.4f}'})
    
    return total_loss / steps

def evaluate(model, loader, device, scaler, precision="fp16"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    cast_enabled = (precision in ["fp16","bf16"])
    with torch.no_grad():
        for batch in loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)

            with torch.cuda.amp.autocast(enabled=cast_enabled):
                logits, _, _, _, _ = model(ids, mask)
                loss = loss_fn(logits, targets, alpha=0.05)

            total_loss += loss.item()
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    bin_preds = (all_preds > 0.5).astype(int)

    precision = precision_score(all_targets, bin_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, bin_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, bin_preds, average='macro', zero_division=0)

    return total_loss / len(loader), precision, recall, f1, all_preds, all_targets
