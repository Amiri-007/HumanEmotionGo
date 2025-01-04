###############################################################################
# File: visualization.py
# Author: Michael R. Amiri
#
# Description:
#  Provides:
#   - extract_epoch_features: collects layer-wise features for TSNE & clustering
#   - plot_tsne_evolution: plots TSNE across epochs for multiple layers
#   - plot_hc_evolution: does hierarchical clustering plots across epochs
###############################################################################

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

def extract_epoch_features(model_eval, loader, device, sample_size=2000):
    """
    Gathers hidden representations from multiple layers:
      - BERT layer 1
      - BERT layer 12
      - BERT layer 24
      - MLP output
      - Attention (weighted_out)
      - Classifier penultimate feats

    Returns:
      dict of feats: { '1':..., '12':..., '24':..., 'mlp':..., 'att':..., 'clf':... }
      label array: np.array of dominant label indices
    """
    model_eval.eval()
    feats_1, feats_12, feats_24 = [], [], []
    feats_mlp, feats_att, feats_clf = [], [], []
    all_labels = []
    count = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets'].cpu().numpy()

            logits, all_hs, seq_out, weighted_out, penult = model_eval(ids, mask)
            layer_1 = all_hs[1][:,0,:].cpu().numpy()
            layer_12 = all_hs[12][:,0,:].cpu().numpy()
            layer_24 = all_hs[24][:,0,:].cpu().numpy()

            mlp_out = seq_out[:,0,:].cpu().numpy()
            att_out = weighted_out.cpu().numpy()
            clf_out = penult.cpu().numpy()

            feats_1.append(layer_1)
            feats_12.append(layer_12)
            feats_24.append(layer_24)
            feats_mlp.append(mlp_out)
            feats_att.append(att_out)
            feats_clf.append(clf_out)

            dom_label = np.argmax(targets, axis=1)
            all_labels.extend(dom_label.tolist())

            count += len(ids)
            if count >= sample_size:
                break

    feats_1 = np.vstack(feats_1)
    feats_12 = np.vstack(feats_12)
    feats_24 = np.vstack(feats_24)
    feats_mlp = np.vstack(feats_mlp)
    feats_att = np.vstack(feats_att)
    feats_clf = np.vstack(feats_clf)

    return {
        '1': feats_1,
        '12': feats_12,
        '24': feats_24,
        'mlp': feats_mlp,
        'att': feats_att,
        'clf': feats_clf
    }, np.array(all_labels)

def plot_tsne_evolution(all_ep_feats, emotion_labels, save_path):
    """
    TSNE across epochs for keys: [1,12,24,mlp,att,clf].
    Each row = epoch, each column = representation layer.
    """
    keys = ['1','12','24','mlp','att','clf']
    epochs = len(all_ep_feats['1'])
    fig, axes = plt.subplots(nrows=epochs, ncols=len(keys), figsize=(5*len(keys), 5*epochs), facecolor='white')
    fig.suptitle('t-SNE Visualization Evolution Across Layers/Epochs', fontsize=16, y=0.95)

    unique_emotions = np.unique(emotion_labels)
    colors = plt.cm.tab20(np.linspace(0,1,len(unique_emotions)))
    color_dict = dict(zip(unique_emotions, colors))

    layer_titles = {
        '1':'BERT Layer 1',
        '12':'BERT Layer 12',
        '24':'BERT Layer 24',
        'mlp':'MLP',
        'att':'Attention',
        'clf':'Classifier(256)'
    }

    for e in range(epochs):
        for li, key in enumerate(keys):
            if epochs > 1:
                ax = axes[e, li]
            else:
                ax = axes[li]
            feats = all_ep_feats[key][e]
            if feats.shape[0] > 5 and feats.shape[1] > 2:
                pca = PCA(n_components=min(50, feats.shape[1]))
                feats_pca = pca.fit_transform(feats)
                tsne = TSNE(n_components=2, random_state=42, init='random')
                feats_tsne = tsne.fit_transform(feats_pca)

                for emo in unique_emotions:
                    mask = (emotion_labels == emo)
                    ax.scatter(feats_tsne[mask,0], feats_tsne[mask,1], c=[color_dict[emo]], alpha=0.5, s=10)
                if e == 0:
                    ax.set_title(layer_titles[key])
                if li == 0:
                    ax.set_ylabel(f"Epoch {e+1}")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_hc_evolution(all_ep_feats, emotion_labels, save_path):
    """
    Hierarchical clustering evolution across epochs for penultimate classifier feats.
    """
    feats_key = 'clf'
    epochs = len(all_ep_feats[feats_key])
    fig, axes = plt.subplots(nrows=epochs, ncols=1, figsize=(8,4*epochs), facecolor='white')
    fig.suptitle('Hierarchical Clustering Evolution (Classifier)', fontsize=16, y=0.95)

    if epochs == 1:
        axes = [axes]

    for e in range(epochs):
        ax = axes[e]
        feats = all_ep_feats[feats_key][e]
        unique_emotions = np.unique(emotion_labels)

        emotion_centroids = []
        emotion_names = []
        for emo in unique_emotions:
            mask = (emotion_labels == emo)
            if np.sum(mask) > 0:
                centroid = np.mean(feats[mask], axis=0)
                emotion_centroids.append(centroid)
                emotion_names.append(emo)

        emotion_centroids = np.array(emotion_centroids)
        if emotion_centroids.shape[0] > 1:
            dist_mat = pdist(emotion_centroids, metric='euclidean')
            Z = linkage(dist_mat, method='ward')
            dendrogram(Z, labels=emotion_names, orientation='right', leaf_font_size=8, ax=ax)
            ax.set_title(f"Epoch {e+1}")
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center')
        ax.set_xlabel('Distance')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
