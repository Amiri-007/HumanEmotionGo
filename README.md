# **Computational Cognitive Modeling of Human Emotion**

**Fine-Grained Sentiment Classification with Representational Analysis**

---

## **Overview**

This repository implements a **fine-grained sentiment classification** pipeline using **RoBERTa-large** on the **GoEmotions dataset** (27 emotion categories). It provides:

1. **Multi-GPU Parallel Training** with PyTorch **Distributed Data Parallel** (DDP), optimized for HPC clusters (e.g., Greene at NYU).  
2. **Mixed-Precision** (fp16/bf16/fp32) to reduce memory usage and improve throughput on NVIDIA GPUs.  
3. **Comprehensive Visualizations**—including:
   - **t-SNE** evolution of embeddings across epochs and layers (e.g., BERT layer 1, 12, 24, MLP, Attention, Classifier),
   - **Hierarchical Clustering** snapshots at various epochs/layers,
   - **Confusion/Misclassification** analysis and correlation heatmaps.
4. **Label Smoothing**, **Early Stopping**, and **Flexible** arguments for `epochs`, `learning rate`, `batch size`, etc.
5. **Representation Analysis** that mirrors **cognitive science** theories of hierarchical emotion processing, comparing how embeddings evolve from lower layers (basic lexical patterns) to higher layers (complex emotional semantics).

This code aims to **bridge computational and cognitive** perspectives, showing how **Transformer-based** architectures approximate the **hierarchical** nature of human emotional understanding—while also highlighting **challenges** with nuanced or less frequent emotions (e.g., _grief_, _relief_, _disappointment_).

---

## **Key Improvements Over Previous Versions**

1. **Enhanced Readability**: The pipeline now includes more intuitive variable names and code comments.
2. **Updated HPC Scripts**: We provide two sample Slurm scripts—one for **single GPU** usage, one for **multi-GPU**—to exploit the Greene HPC’s A100/H100 GPUs effectively.
3. **Model Analysis Hooks**: Additional logging and an **EmotionRepresentationTracker** module (in code) enable deeper inspection of how the model learns, layer by layer.
4. **Comprehensive Visuals**: Inspired by the t-SNE/hierarchical clustering analysis from the paper, we included more thorough plotting routines for *all* layers and epochs.

---

## **Repository Structure**

```
my_emotion_classification/
├── README.md                  # You are here
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── run_on_gpu_single.sh          # HPC script for single-GPU usage
│   └── run_on_gpu_cluster_multi.sh   # HPC script for multi-GPU usage (DDP)
└── src/
    ├── __init__.py
    ├── main.py                 # Main entry point for training + analysis
    ├── dataset.py              # BERTDataset definition
    ├── model.py                # CustomRoBERTaModel with MLP + attention + classifier
    ├── training.py             # Training loop, evaluation loop, label-smoothing loss
    ├── visualization.py        # TSNE + hierarchical clustering extraction & plotting
    ├── analysis.py             # Confusion matrix, final representation analysis, heatmaps
```

### **Highlights**
- **`main.py`**: Orchestrates data loading, model setup, HPC auto-detection, and final analysis.  
- **`training.py`**: Houses `train_one_epoch` and `evaluate` functions, each supporting mixed-precision and multi-GPU.  
- **`visualization.py`**: Tools for capturing layerwise embeddings, performing **t-SNE** across epochs, and hierarchical clustering.  
- **`analysis.py`**: Functions to generate confusion matrices, cluster heatmaps, etc.

---

## **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Amiri-007/HumanEmotionGo.git
   cd HumanEmotionGo
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or use a Conda environment if preferred (e.g., `conda create -n emotion python=3.9 && conda activate emotion`).

---

## **Usage**

### **Local Single GPU (or CPU)**

Run training on your local machine (uses 1 GPU if available):
```bash
python src/main.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --epochs 10 \
  --batch_size 32 \
  --lr 5e-6 \
  --precision fp16 \
  --max_len 128 \
  --save_dir checkpoints \
  --visualize
```

- **`--visualize`** will enable t-SNE, hierarchical clustering, confusion matrix plots, etc.
- **`--precision`** can be `fp16`, `bf16`, or `fp32` depending on your hardware and preference.
- **`--batch_size`** ideally a multiple of **32** to align with **NVIDIA GPU warp** size.

### **HPC: Single GPU** 
Submit the **single-GPU** Slurm script:
```bash
sbatch scripts/run_on_gpu_single.sh
```
That script might contain:
```bash
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
# ... etc ...

module purge
module load cuda/11.3
module load python/3.8
source activate my_emotion_env

cd /path/to/HumanEmotionGo
python src/main.py --train_csv data/train.csv ...
```

### **HPC: Multi-GPU (Distributed Data Parallel)** 
Submit the **multi-GPU** Slurm script:
```bash
sbatch scripts/run_on_gpu_cluster_multi.sh
```
Inside you might have:
```bash
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
# ... etc ...

module purge
module load cuda/11.3
module load python/3.8
source activate my_emotion_env

cd /path/to/HumanEmotionGo
torchrun --nproc_per_node=4 src/main.py --train_csv data/train.csv ...
```
This spawns **4 processes** across **4 GPUs**, each training on a portion of data, synchronizing gradients with **DDP**.

---

## **Example Findings**

1. **Emotion Performance Variation**  
   - **High F1**: _Gratitude_ (0.91), _Amusement_ (0.85), _Love_ (0.84)  
   - **Low F1**: _Grief_ (0.00), _Relief_ (0.29), _Realization_ (0.29)  
   - Macro-average F1 ~ **0.55**, indicating moderate performance overall.
2. **Layer-wise Representation Analysis**:  
   - Early layers (1, 12) show limited separation of emotion embeddings.  
   - Later layers (24, MLP, Attention, Classifier) yield more distinct clusters in **t-SNE**.  
3. **Hierarchical Clustering** reveals consistent grouping of **positive** vs. **negative** emotions in deeper layers, reflecting a cognitively plausible separation of sentiment categories.
4. **Model-Generated vs. Human-Annotated** correlation heatmaps: The model is fairly aligned with broad emotion categories, but struggles with subtle or rare categories. This underscores the complexity of human emotional nuance.

---

## **Insights and Recommendations**

- **Future Directions**:  
  1. **Focus on Rare Emotions**: Additional data or advanced sampling may boost performance for low-frequency categories (like _grief_).  
  2. **Cross-Cultural Extensions**: Expand beyond English/Reddit to see how BERT-like models handle broader emotional expression.  
  3. **Multimodal Approaches**: Incorporate audio/visual data for a richer reflection of genuine human emotional cues.

- **Practical Tips**:  
  1. **Batch Size** = 32 (or 64, 128 if memory allows) to match GPU warp size.  
  2. **Label Smoothing** helps reduce overconfidence in classification.  
  3. **Early Stopping** is crucial to avoid overfitting on HPC resources.

---

## **Citation**

If you reference this work in your research, please cite both the **GoEmotions** dataset paper:

> Demszky, D., Movshovitz-Attias, D., Ko, J., et al. (2020). *GoEmotions: A dataset of fine-grained emotions*. arXiv preprint arXiv:2005.00547.

and our approach, which parallels **cognitive theories** of hierarchical emotion processing in large language models.

---

## **Acknowledgments**

We gratefully acknowledge the **Greene HPC Center at New York University** for providing computational resources (A100’s, H100) that made training these large RoBERTa models feasible. This project draws on cognitive science literature to interpret how Transformer layers evolve emotional representations.

**Contact**:  
- **Michael R. Amiri** (<ma7422@nyu.edu>)
- **Yuqi Hang** (<yh2072@nyu.edu>)  
- **Ziyu Qi** (<zq2127@nyu.edu>)  
- **Yuchen Gao** (<yg3113@nyu.edu>)  

---

_This README is based on our extended paper “Computational Cognitive Modeling of Human Emotion: A Fine-Grained Sentiment Classification with Representational Analysis,” where we share both the **quantitative** performance metrics and the **qualitative** layerwise insights into how the model clusters emotions._
