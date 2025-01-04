# Enhanced Fine-Grained Emotion Classification (CUDA/Parallelized)

**Authors**: Michael R. Amiri (ma7422@nyu.edu), Yuqi Hang (yh2072@nyu.edu), Ziyu Qi (zq2127@nyu.edu), Yuchen Gao (yg3113@nyu.edu)

## Overview

This repository demonstrates a **multi-label emotion classification** pipeline using **RoBERTa-large**. It has been carefully structured to leverage **CUDA** capabilities on NVIDIA GPUs:

- **Batch size** settable to multiples of **32** for improved warp efficiency.
- **Single-GPU** or **multi-GPU** parallel training via **Distributed Data Parallel**.
- **Label smoothing**, **mixed precision** (fp16/bf16/fp32), and **early stopping**.

## Key Features

1. **Automatic Mixed Precision (AMP)**: Minimizes GPU memory usage and speeds up training.  
2. **Parallelization**: If `WORLD_SIZE>1`, the code initializes PyTorch distributed processes (DDP).  
3. **Visualization**: Extended t-SNE (layers [1,12,24, MLP, Attention, Classifier]), hierarchical clustering of the penultimate layer, confusion/misclassification matrix, final cluster analysis.  
4. **HPC Compatibility**: Two example Slurm scripts for HPC usage:
   - `run_on_greene_single.sh` (single GPU)
   - `run_on_greene_multi.sh` (multi-GPU distributed)
