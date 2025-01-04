###############################################################################
# File: analysis.py
# Author: Michael R. Amiri
# Date: 2025-01-04
#
# Description:
#  Plotting functions for confusion/misclassification analysis and final
#  penultimate-layer representation analysis (PCA, t-SNE, hierarchical 
#  clustering, KMeans distribution).
###############################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

def plot_confusion_analysis(cooc, miscl, emo_names, savename):
    """
    Creates a 1x2 figure:
      Left: cooccurrence matrix (distribution)
      Right: misclassification matrix
    """
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,8))
    sns.heatmap(cooc, xticklabels=emo_names, yticklabels=emo_names, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax1)
    ax1.set_title('Emotion Distribution\n(# of samples per emotion)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.heatmap(miscl, xticklabels=emo_names, yticklabels=emo_names, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax2)
    ax2.set_title('Emotion Misclassification Matrix\n(pred row vs. gold col)')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def plot_representation_analysis(features, labels, emo_names, savename):
    """
    Plots:
      - PCA explained variance
      - t-SNE scatter
      - hierarchical clustering of centroids
      - KMeans cluster distribution
    for final penultimate-layer features.
    """
    if features.shape[0] < 5:
        return

    n_components_pca = min(50, features.shape[1], features.shape[0] - 1)
    pca = PCA(n_components=n_components_pca)
    pca_result = pca.fit_transform(features)
    explained_var = np.sum(pca.explained_variance_ratio_) * 100

    tsne_out = None
    if pca_result.shape[1] >= 2:
        tsne = TSNE(n_components=2, random_state=42, init='random')
        tsne_out = tsne.fit_transform(pca_result)

    unique_labs = np.unique(labels)
    emotion_centroids = []
    emotion_names = []
    for lb in unique_labs:
        mask = (labels == lb)
        if np.sum(mask) > 0:
            centroid = np.mean(pca_result[mask], axis=0)
            emotion_centroids.append(centroid)
            emotion_names.append(lb)
    emotion_centroids = np.array(emotion_centroids)
    if emotion_centroids.shape[0] > 1:
        Z = linkage(emotion_centroids, 'ward')

    plt.figure(figsize=(18,6))
    gs = plt.GridSpec(2,3, height_ratios=[1.5,1.5], wspace=0.3, hspace=0.8)
    plt.suptitle("Final Representation (Classifier Layer)", fontsize=16)

    # PCA explained
    ax1 = plt.subplot(gs[0,0])
    ax1.plot(np.cumsum(pca.explained_variance_ratio_)*100, 'b-')
    ax1.set_title(f"PCA Explained (Total: {explained_var:.1f}%)")
    ax1.set_xlabel('Components')
    ax1.set_ylabel('Cumulative %')

    # t-SNE
    ax2 = plt.subplot(gs[0,1])
    if tsne_out is not None:
        color_map = plt.cm.tab20(np.linspace(0,1,len(unique_labs)))
        color_dict = dict(zip(unique_labs, color_map))
        for emo in unique_labs:
            m = (labels == emo)
            ax2.scatter(tsne_out[m,0], tsne_out[m,1], c=[color_dict[emo]], alpha=0.7, s=30)
        ax2.legend(fontsize='x-small', ncol=3)
        ax2.set_title("t-SNE")

    # Hierarchical clustering
    ax3 = plt.subplot(gs[0,2])
    if emotion_centroids.shape[0] > 1:
        dendrogram(Z, labels=emotion_names, orientation='right', leaf_font_size=8, ax=ax3)
        ax3.set_title("Hierarchical Clustering")

    # KMeans cluster distribution
    if features.shape[0] > 5:
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_result)
        cluster_stats = np.zeros((n_clusters, len(unique_labs)))
        for i in range(n_clusters):
            cm = (cluster_labels == i)
            for j, em in enumerate(unique_labs):
                em_mask = (labels == em)
                if np.sum(cm) > 0:
                    cluster_stats[i,j] = np.sum(cm & em_mask) / np.sum(cm)
                else:
                    cluster_stats[i,j] = 0.0

        ax4 = plt.subplot(gs[1,:])
        descs = []
        for i in range(n_clusters):
            top_emos = np.argsort(cluster_stats[i])[-3:][::-1]
            mm = [f"{unique_labs[k]}:{cluster_stats[i][k]:.2f}" for k in top_emos if cluster_stats[i][k]>=0.1]
            descs.append(f"Cluster {i}\n({','.join(mm)})")
        sns.heatmap(cluster_stats, cmap='YlOrRd', xticklabels=unique_labs, yticklabels=descs, annot=True, fmt='.2f', ax=ax4)
        ax4.set_title("Emotion Distribution in Clusters")

    plt.savefig(savename)
    plt.close()
