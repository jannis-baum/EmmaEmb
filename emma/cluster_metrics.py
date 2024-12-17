from typing import Optional

import numpy as np

from sklearn.metrics import (
    silhouette_score, silhouette_samples, 
    davies_bouldin_score, calinski_harabasz_score
    )


def evaluate_clusters(
    evaluation_method: str,
    embeddings: np.array, 
    labels: list,
    distance_metric: str = None) -> int:
    
    if evaluation_method == 'silhouette_score':
        score = silhouette_score(embeddings, labels, metric=distance_metric)
        
    elif evaluation_method == 'davies_bouldin_index':
        if distance_metric != "euclidean":
            raise ValueError("davies_bouldin_index only \
                implemented with the euclidean distance.")
        score = davies_bouldin_score(embeddings, labels)
    
    elif evaluation_method == 'calinski_harabasz_score':
        if distance_metric:
            raise ValueError("davies_bouldin_index is\
                not available for different distance metrics.")
        score = calinski_harabasz_score(embeddings, labels)
    else: 
        raise ValueError(f"{evaluation_method} is not a \
            support evaluation methods.")
        
    return score


def evaluate_clusters_per_cluster(
    evaluation_method: str,
    embeddings: np.array, 
    labels: list,
    distance_metric: str) -> dict:
    
    if evaluation_method == 'silhouette_score':
        silhouette_vals = silhouette_samples(embeddings, labels)
        unique_labels = np.unique(labels)
        per_cluster_scores = {label: np.mean(silhouette_vals[labels == label]) 
                              for label in unique_labels}
        
    else: 
        raise ValueError(f"{evaluation_method} is not a \
            support evaluation methods.")
    
    return per_cluster_scores