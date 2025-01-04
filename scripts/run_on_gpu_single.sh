#!/bin/bash
###############################################################################
# File: run_on_greene_single.sh
# Author: Michael R. Amiri
# Date: 2025-01-04
# 
# Description:
#  This script is an example Slurm batch file for launching the emotion
#  classification code on a single GPU node of the Greene HPC cluster.
#
# Usage:
#   sbatch run_on_greene_single.sh
###############################################################################

#SBATCH --job-name=emotion_single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/emotion_single_%j.out

module purge
module load cuda/11.3
module load python/3.8

# Activate your environment
source activate my_emotion_env

# Move to the root directory of the project
cd /path/to/my_emotion_classification

# Run the training script
python src/main.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-6 \
    --max_len 128 \
    --precision fp16 \
    --save_dir checkpoints \
    --visualize
