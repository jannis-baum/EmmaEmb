from typing import Optional

import numpy as np
from scipy.stats import entropy

from sklearn.metrics import (
    silhouette_score, silhouette_samples, 
    davies_bouldin_score, calinski_harabasz_score
    )

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


def evaluate_clusters(
    evaluation_method: str,
    embeddings: np.array, 
    labels: list,
    distance_metric: str = None) -> int:
    
    if evaluation_method == 'silhouette_score':
        if sum(labels) < 0: #TODO fix this error
            return None
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

def purity_score(labels_true, labels_pred):
    """
    Compute the purity score for a clustering compared to true labels.
    
    Purity is the proportion of the total number of points that are assigned
    to the most frequent class within each cluster.
    
    Parameters:
    - labels_true (np.ndarray): True class labels.
    - labels_pred (np.ndarray): Predicted cluster labels.
    
    Returns:
    - float: Purity score.
    """
    total_correct = 0
    
    # iterate over each cluster
    for cluster in np.unique(labels_pred):
        # find indices of the points in the current cluster
        cluster_indices = np.where=(labels_pred == cluster)[0]
        
        # get the true labels of the points in the cluster
        true_labels_in_cluster = labels_true[cluster_indices]
        
        # find most frequent true label in this cluster
        most_frequent_class = np.bincount(true_labels_in_cluster).argmax()
        
        # add the number of points in this cluster that match the most frequent class
        total_correct += np.sum(true_labels_in_cluster == most_frequent_class)
    
    return total_correct/ len(labels_true)

def entropy_score(labels_true, labels_pred):
    """
    Compute the entropy score for a clustering compared to the true labels.
    
    Entropy measures the disorder in the distribution of true labels within 
    each cluster.
    
    Parameters:
    - labels_true (np.array): True class labels
    - labels_pred (np.array): Predicted cluster labels
    
    Returns:
    - float: Entropy score
    """
    
    entropy_scores = []
    
    # iterate over each cluster
    for cluster in np.unique(labels_pred):
        # find the indices of the points in the current cluster
        cluster_indices = np.where(labels_pred == cluster)[0]
        
        # get the true labels of the points in the cluster
        true_labels_in_cluster = labels_true[cluster_indices]
        
        # calculate the entropy of the distribution of true labels in this cluster
        class_probabilities = np.bincount(true_labels_in_cluster) / len(true_labels_in_cluster)
        cluster_entropy = entropy(class_probabilities)
        entropy_scores.append(cluster_entropy)
    
    # return the mean entropy across all clusters
    return np.mean(entropy_scores)


def compare_clusterings(
    labels_1: np.ndarray,
    labels_2: np.ndarray,
    score_method: str = "ARI",
    **kwargs
    ) -> float:
    """
    Compare two clusterings. Different methods can be used.
    
    Parameters:
    - labels_1 (np.ndarray): First set of clustering labels 
        or ground truth labels.
    - labels_2 (np.ndarray): Second set of clustering labels
    - score_method (str): The scoring method to use. Options:
        - 'ARI': Adjusted Rand Index
        - 'NMI': Normalised Mutual Information
        - 'Purity': Purity score
        - 'Entropy': Entropy score.
    - kwargs: Additional parameters for specific score
        methods (if any are required).
    
    Returns:
    - float: The computed score based on the specified method.
    """
    
    # Ensure labels are numpy arrays
    labels_1 = np.asarray(labels_1, dtype=str)
    labels_2 = np.asarray(labels_2, dtype=str)
    
    # Select and compute the score
    if score_method == "ARI":
        score = adjusted_rand_score(labels_1, labels_2)
    elif score_method == "NMI":
        score = normalized_mutual_info_score(labels_1, 
                                             labels_2, 
                                             **kwargs)
    elif score_method == "Purity":
        score = purity_score(labels_1, labels_2)
    elif score_method == "Entropy":
        score = entropy_score(labels_1, labels_2)
    else:
        raise ValueError(f"Invalid score_method '{score_method}'. \
            Choose from 'ARI', 'NMI', 'Purity', or 'Entropy'.")

    return score