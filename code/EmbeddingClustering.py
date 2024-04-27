import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from Configure import model_configs
from sklearn.metrics import accuracy_score

def compare_embeddings(image_embeddings, audio_embeddings, labels):

    reduced_image_embeddings = apply_tsne(image_embeddings)
    reduced_audio_embeddings = apply_tsne(audio_embeddings)

    visualize_embeddings(reduced_image_embeddings, labels, 'Image Embeddings', 'ImageEmbeddingsTNSE.png')
    visualize_embeddings(reduced_audio_embeddings, labels, 'Audio Embeddings', 'AudioEmbeddingsTNSE.png')

    image_cluster_labels = apply_kmeans(image_embeddings)
    audio_cluster_labels = apply_kmeans(audio_embeddings)

    compare_clusters(image_cluster_labels, audio_cluster_labels, labels)

def apply_tsne(embeddings):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def visualize_embeddings(reduced_embeddings, labels, title, savefig):
    plt.figure(figsize=(10, 8))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label)
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig(model_configs.result_dir + savefig)

def apply_kmeans(embeddings, k=10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def compare_clusters(image_cluster_labels, audio_cluster_labels, labels):
    
    image_accuracy = accuracy_score(labels, image_cluster_labels)
    audio_accuracy = accuracy_score(labels, audio_cluster_labels)
    
    print("Image Cluster Accuracy:", image_accuracy)
    print("Audio Cluster Accuracy:", audio_accuracy)


