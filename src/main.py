###############################################################################
# File: main.py
# Author: Michael R. Amiri
# Date: 2025-01-04
#
# Description:
#  Main entry point for the multi-label emotion classification using RoBERTa.
#  Supports:
#    - Single GPU or multi-GPU via Distributed Data Parallel (DDP).
#    - Mixed precision (fp16/bf16/fp32).
#    - Early stopping with patience.
#    - Extended visualizations: t-SNE across layers, hierarchical clustering,
#      final confusion matrix, and cluster analysis of penultimate features.
#  Recommends batch sizes that are multiples of 32 to align with CUDA warp size.
###############################################################################

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging
import warnings
warnings.filterwarnings('ignore')

# Visualization headless
import matplotlib
matplotlib.use('Agg')

# HuggingFace / TQDM / training imports
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from tqdm.auto import tqdm

# Local modules
from dataset import BERTDataset
from model import CustomRobertaModel
from training import train_one_epoch, evaluate
from visualization import extract_epoch_features, plot_tsne_evolution, plot_hc_evolution
from analysis import plot_confusion_analysis, plot_representation_analysis

def main():
    """
    Parses command-line arguments, sets up environment (DDP if multi-GPU),
    trains, and runs final analysis.
    """
    parser = argparse.ArgumentParser(description="Emotion Classification with Extended Visualizations")

    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to testing CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU (multiple of 32 is ideal)")
    parser.add_argument("--lr", type=float, default=5e-6, help="Base learning rate")
    parser.add_argument("--max_len", type=int, default=128, help="Max text length")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16","bf16","fp32"], help="Mixed precision choice")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save outputs")
    parser.add_argument("--visualize", action='store_true', help="Produce extended t-SNE/hierarchical clustering analysis.")
    args = parser.parse_args()

    # Make output dirs
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    save_dir2 = os.path.join(args.save_dir, "analysis_plots")
    os.makedirs(save_dir2, exist_ok=True)

    # Logging
    logging.basicConfig(
        level=logging.INFO if ('RANK' not in os.environ or int(os.environ['RANK']) == 0) else logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, "training.log")),
            logging.StreamHandler()
        ]
    )

    # Distributed / device setup
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main_process = (local_rank == 0)

    # Set seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Check CSVs
    if not os.path.exists(args.train_csv):
        raise FileNotFoundError(f"Train file not found: {args.train_csv}")
    if not os.path.exists(args.test_csv):
        raise FileNotFoundError(f"Test file not found: {args.test_csv}")

    df_train = pd.read_csv(args.train_csv)
    df_test = pd.read_csv(args.test_csv)

    df_train_data = df_train.drop(columns=['ID'], errors='ignore')
    df_test_data = df_test.drop(columns=['ID'], errors='ignore')
    target_cols = [c for c in df_train_data.columns if c not in ['Text','ID']]

    # Tokenizer / Datasets
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    train_dataset = BERTDataset(df_train_data, tokenizer, args.max_len, target_cols)
    test_dataset = BERTDataset(df_test_data, tokenizer, args.max_len, target_cols)

    # Distributed Samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size>1 else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size>1 else None

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    num_classes = len(target_cols)
    model = CustomRobertaModel(num_classes=num_classes).to(device)

    # AMP
    if args.precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    elif args.precision == "bf16":
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        torch.set_default_dtype(torch.bfloat16)
    else:
        scaler = None

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            'params': [p for n,p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('roberta' in n)],
            'lr': args.lr,
            'weight_decay': 0.01
        },
        {
            'params': [p for n,p in param_optimizer if (any(nd in n for nd in no_decay)) and ('roberta' in n)],
            'lr': args.lr,
            'weight_decay': 0.0
        },
        {
            'params': [p for n,p in param_optimizer if (not any(nd in n for nd in no_decay)) and ('roberta' not in n)],
            'lr': args.lr*5,
            'weight_decay': 0.01
        },
        {
            'params': [p for n,p in param_optimizer if (any(nd in n for nd in no_decay)) and ('roberta' not in n)],
            'lr': args.lr*5,
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters)

    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*num_training_steps),
        num_training_steps=num_training_steps
    )

    # If multi-GPU, wrap model in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Training
    train_losses = []
    val_losses = []
    train_accs = []
    val_f1s = []
    best_loss = float('inf')
    patience = 3
    patience_counter = 0

    # For storing embeddings across epochs for TSNE/HC
    all_epochs_features = {
        '1':[],
        '12':[],
        '24':[],
        'mlp':[],
        'att':[],
        'clf':[]
    }
    emotion_label_array = None

    for epoch in range(args.epochs):
        tr_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            epoch_idx=epoch,
            epochs=args.epochs,
            grad_accum=2,
            is_main_process=is_main_process,
            precision=args.precision,
            train_sampler=train_sampler
        )
        val_loss, prec, rec, f1, _, _ = evaluate(model, test_loader, device, scaler, args.precision)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(0.0)  # placeholder
        val_f1s.append(f1)

        if is_main_process:
            logging.info(f"Epoch {epoch+1}/{args.epochs} - "
                         f"Train Loss={tr_loss:.4f}, Val Loss={val_loss:.4f}, "
                         f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(args.save_dir,"best_emotion_model.pt"))
                logging.info(f"New best model with val_loss={best_loss:.4f} saved.")
            else:
                patience_counter += 1
                logging.info(f"No improvement, patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logging.info("Early stopping triggered.")
                    break

            # Visualization each epoch if requested
            if args.visualize:
                # Reload the best model for consistent feature extraction
                model_eval = CustomRobertaModel(num_classes=num_classes).to(device)
                model_eval.load_state_dict(torch.load(os.path.join(args.save_dir,"best_emotion_model.pt"), map_location=device))
                if world_size > 1:
                    model_eval = DDP(model_eval, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

                feats_dict, lbls = extract_epoch_features(model_eval, test_loader, device=device, sample_size=2000)
                for k in feats_dict:
                    all_epochs_features[k].append(feats_dict[k])

                if emotion_label_array is None:
                    emotion_label_array = lbls.copy()

    # Final Evaluation & Analysis
    if is_main_process and os.path.exists(os.path.join(args.save_dir,"best_emotion_model.pt")):
        final_model = CustomRobertaModel(num_classes=num_classes).to(device)
        final_model.load_state_dict(torch.load(os.path.join(args.save_dir,"best_emotion_model.pt"), map_location=device))
        if world_size > 1:
            final_model = DDP(final_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        val_loss, precision_val, recall_val, f1_val, all_preds, all_targets = evaluate(final_model, test_loader, device, scaler, args.precision)
        logging.info(f"Final evaluation: ValLoss={val_loss:.4f}, Precision={precision_val:.4f}, Recall={recall_val:.4f}, F1={f1_val:.4f}")

        # emotion-wise stats
        bin_preds = (all_preds > 0.5).astype(int)
        emotions = target_cols
        n_emo = len(emotions)
        per_emotion_stats = []
        for i, emo in enumerate(emotions):
            y_true = all_targets[:,i]
            y_pred = bin_preds[:,i]
            tp = np.sum((y_pred==1) & (y_true==1))
            fp = np.sum((y_pred==1) & (y_true==0))
            fn = np.sum((y_pred==0) & (y_true==1))
            prec_i = tp/(tp+fp) if (tp+fp)>0 else 0.0
            rec_i = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1_i = (2*prec_i*rec_i/(prec_i+rec_i)) if (prec_i+rec_i)>0 else 0.0
            support_i = int(np.sum(y_true))
            per_emotion_stats.append((emo, prec_i, rec_i, f1_i, support_i))

        macro_p = np.mean([x[1] for x in per_emotion_stats])
        macro_r = np.mean([x[2] for x in per_emotion_stats])
        macro_f = np.mean([x[3] for x in per_emotion_stats])
        total_sup = sum([x[4] for x in per_emotion_stats])
        weighted_p = np.average([x[1] for x in per_emotion_stats], weights=[x[4] for x in per_emotion_stats])
        weighted_r = np.average([x[2] for x in per_emotion_stats], weights=[x[4] for x in per_emotion_stats])
        weighted_f = np.average([x[3] for x in per_emotion_stats], weights=[x[4] for x in per_emotion_stats])

        logging.info("{:<20} {:<12} {:<12} {:<12} {:<12}".format("Emotion","Precision","Recall","F1","Support"))
        logging.info("-"*70)
        for row in per_emotion_stats:
            emo,p_i,r_i,f_i,s_i = row
            logging.info("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12d}".format(emo,p_i,r_i,f_i,s_i))
        logging.info("-"*70)
        logging.info(f"Macro Avg       P={macro_p:.4f} R={macro_r:.4f} F1={macro_f:.4f}")
        logging.info(f"Weighted Avg    P={weighted_p:.4f} R={weighted_r:.4f} F1={weighted_f:.4f} total={total_sup}")

        # Confusion & Misclassification
        cooccurrence = np.zeros((n_emo,n_emo))
        for i in range(n_emo):
            cooccurrence[i,i] = np.sum(all_targets[:,i]==1)
        pred_argmax = np.argmax(all_preds,axis=1)
        gold_argmax = np.argmax(all_targets,axis=1)
        misclass = np.zeros((n_emo,n_emo))
        for i in range(len(pred_argmax)):
            if pred_argmax[i] != gold_argmax[i]:
                misclass[pred_argmax[i], gold_argmax[i]] += 1

        plot_confusion_analysis(cooccurrence, misclass, emotions, os.path.join(save_dir2,"confusion_analysis.png"))

        # Final representation analysis (penultimate layer)
        final_feats_dict, lbls = extract_epoch_features(final_model, test_loader, device=device, sample_size=2000)
        feats_clf = final_feats_dict['clf']
        lbl_emotions = np.array(emotions)[lbls]
        plot_representation_analysis(feats_clf, lbl_emotions, emotions, os.path.join(save_dir2,"final_analysis.png"))

        # TSNE/HC evolution across epochs
        if args.visualize and len(all_epochs_features['clf'])>0 and emotion_label_array is not None:
            plot_tsne_evolution(all_epochs_features, emotion_label_array, os.path.join(save_dir2,"tsne_evolution.png"))
            plot_hc_evolution(all_epochs_features, emotion_label_array, os.path.join(save_dir2,"hc_evolution.png"))

    logging.info("Training and evaluation completed.")

if __name__ == "__main__":
    main()
