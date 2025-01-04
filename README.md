# Enhanced Fine-Grained Emotion Classification (CUDA/Parallelized)

**Authors**: 
Michael R. Amiri (ma7422@nyu.edu), 
Yuqi Hang (yh2072@nyu.edu), 
Ziyu Qi (zq2127@nyu.edu), 
Yuchen Gao (yg3113@nyu.edu), 

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

Install dependencies:
bash
Copy code
pip install -r requirements.txt
or set up a Conda environment and install accordingly.
Usage
Local Single-GPU (or CPU) example:

bash
Copy code
python src/main.py \
    --train_csv data/train.csv \
    --test_csv data/test.csv \
    --epochs 10 \
    --batch_size 32 \
    --precision fp16 \
    --save_dir checkpoints \
    --visualize
HPC - Single GPU:

bash
Copy code
sbatch scripts/run_on_greene_single.sh
HPC - Multi GPU:

bash
Copy code
sbatch scripts/run_on_greene_multi.sh
Directory Layout
css
Copy code
my_emotion_classification/
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_on_greene_single.sh
│   └── run_on_greene_multi.sh
└── src/
    ├── __init__.py
    ├── main.py
    ├── dataset.py
    ├── model.py
    ├── training.py
    ├── visualization.py
    ├── analysis.py
src/main.py: Entry point for the training + analysis pipeline.
src/dataset.py: Data loading code.
src/model.py: Definition of CustomRobertaModel with MLP, attention, and multi-layer classifier.
src/training.py: Training/evaluation loops with mixed precision + DDP logic.
src/visualization.py: TSNE/HC evolution extraction and plotting.
src/analysis.py: Confusion matrix, final cluster analysis, representation plots.
scripts/: Slurm job scripts for Greene HPC (single or multi-GPU).
Citation
If you use or extend this code, please cite or mention the repository.
Happy training and exploring multi-label emotions with CUDA acceleration!

yaml
Copy code

---

## 2. `requirements.txt`

```txt
# Professional-tier list of pinned dependencies for the HPC environment
torch==2.0.1
transformers==4.28.1
pandas==1.5.3
numpy==1.23.5
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
tqdm==4.65.0
