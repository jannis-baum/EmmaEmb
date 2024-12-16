from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from typing import Tuple, Union
import numpy as np

def calculate_clusters(
    embeddings: np.array,
    algorithm="kmeans", **kwargs
) -> Tuple[np.ndarray, 
           Union[KMeans, DBSCAN, AgglomerativeClustering, 
                 SpectralClustering, GaussianMixture, OPTICS, Birch, HDBSCAN]
           ]:
    """
    Perform unsupervised clustering on an embedding space using different algorithms.

    Parameters:
    - embeddings (ndarray): The input embedding space (n_samples x n_features).
    - algorithm (str): The clustering algorithm to use. Options are:
        - "kmeans"
        - "dbscan"
        - "agglomerative"
        - "spectral"
        - "gmm" (Gaussian Mixture Model)
        - "optics"
        - "birch"
        - "hdbscan"
    - kwargs: Additional parameters for the selected clustering algorithm.

    Returns:
    - labels (ndarray): Cluster labels for each sample.
    - model: The trained clustering model.
    """
     # Check if embeddings are valid
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings cannot be None or empty.")

    algorithm = algorithm.lower()
    
    # Select the clustering algorithm
    if algorithm == "kmeans":
        model = KMeans(**kwargs)
    elif algorithm == "dbscan":
        model = DBSCAN(**kwargs)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(**kwargs)
    elif algorithm == "spectral":
        model = SpectralClustering(**kwargs)
    elif algorithm == "gmm":
        model = GaussianMixture(**kwargs)
    elif algorithm == "optics":
        model = OPTICS(**kwargs)
    elif algorithm == "birch":
        model = Birch(**kwargs)
    elif algorithm == "hdbscan":
        model = HDBSCAN(**kwargs)
    else:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    # Fit the model and get cluster labels
    if algorithm in ["gmm", "hdbscan"]:
        # GaussianMixture and HDBSCAN require `fit_predict` to get labels
        labels = model.fit_predict(embeddings)
    else:
        labels = model.fit(embeddings).labels_

    return labels, model