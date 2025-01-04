###############################################################################
# File: training.py
# Author: Michael R. Amiri
# Date: 2025-01-04
#
# Description:
#  This module contains:
#   - A label-smoothing loss function
#   - A train_one_epoch function
#   - An evaluate function
#  It uses PyTorch's distributed features for multi-GPU training via DDP,
#  and AMP (fp16/bf16) for efficient CUDA usage. 
###############################################################################

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

def loss_fn(outputs, targets, alpha=0.05):
    """
    Label smoothing:
      smoothed_targets = targets * (1 - alpha) + (alpha / num_classes)
    """
    num_classes = targets.size(1)
    smoothed_targets = targets * (1 - alpha) + (alpha / num_classes)
    return nn.BCEWithLogitsLoss()(outputs, smoothed_targets)

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
    """
    Conducts a single epoch of training under optional AMP and DDP.

    Args:
      model (nn.Module): The model (DDP-wrapped if multi-GPU).
      train_loader (DataLoader): Training data loader.
      optimizer (torch.optim.Optimizer): Optimizer instance.
      scheduler (Scheduler): Learning rate scheduler.
      device (torch.device): GPU device on which to train.
      scaler (GradScaler or None): AMP gradient scaler if using fp16/bf16.
      epoch_idx (int): The current epoch index.
      epochs (int): Total number of epochs.
      grad_accum (int): Gradient accumulation steps.
      is_main_process (bool): If True, we log progress. 
      precision (str): "fp16", "bf16", or "fp32".
      train_sampler (DistributedSampler or None): If distributed, set epoch seeds.

    Returns:
      float: The average training loss for the epoch.
    """
    model.train()
    if train_sampler is not None:
        train_sampler.set_epoch(epoch_idx)

    total_loss = 0.0
    steps = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_idx+1}/{epochs}", disable=not is_main_process)
    optimizer.zero_grad(set_to_none=True)

    cast_enabled = (precision in ["fp16", "bf16"])
    for batch_idx, batch in enumerate(progress_bar):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)

        # Forward under AMP
        with torch.cuda.amp.autocast(enabled=cast_enabled):
            logits, _, _, _, _ = model(ids, mask)
            loss = loss_fn(logits, targets, alpha=0.05)
            loss = loss / grad_accum

        # Backprop
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every grad_accum steps
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
    """
    Evaluates the model on a validation/test set.

    Args:
      model (nn.Module): The model in eval mode.
      loader (DataLoader): Data loader for validation/test.
      device (torch.device): GPU device.
      scaler (GradScaler or None): AMP scaler if used.
      precision (str): "fp16", "bf16", or "fp32".
    
    Returns:
      tuple: (val_loss, precision, recall, f1, all_preds, all_targets)
    """
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

    precision_val = precision_score(all_targets, bin_preds, average='macro', zero_division=0)
    recall_val = recall_score(all_targets, bin_preds, average='macro', zero_division=0)
    f1_val = f1_score(all_targets, bin_preds, average='macro', zero_division=0)

    return total_loss / len(loader), precision_val, recall_val, f1_val, all_preds, all_targets
