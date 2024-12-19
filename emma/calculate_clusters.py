from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS, Birch, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from typing import Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def calculate_clusters(
    embeddings: np.array,
    algorithm="kmeans", **kwargs
) -> Tuple[np.ndarray, 
           Union[KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, 
                 SpectralClustering, GaussianMixture, OPTICS, Birch, HDBSCAN]
           ]:
    """
    Perform unsupervised clustering on an embedding space using different algorithms.

    Parameters:
    - embeddings (ndarray): The input embedding space (n_samples x n_features).
    - algorithm (str): The clustering algorithm to use. Options are:
        - "kmeans"
        - "minibatch_kmeans"
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
    elif algorithm == "minibatch_kmeans":
        model = MiniBatchKMeans(**kwargs)
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
    elif algorithm == "minibatch_kmeans":
        # MiniBatchKMeans requires `fit` and then `labels_`
        model.fit(embeddings)
        labels = model.labels_
    else:
        labels = model.fit(embeddings).labels_

    return labels, model



class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_layers, bottleneck_dim):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        decoder_layers = []

        # Build encoder
        current_dim = input_dim
        for h_dim in hidden_layers:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = h_dim
        encoder_layers.append(nn.Linear(current_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder
        current_dim = bottleneck_dim
        for h_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = h_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def apply_autoencoder(
    embeddings: np.ndarray, params: dict
)-> np.ndarray:
    """
    Reduce dimensionality using an Autoencoder.
    
    Parameters:
    - embeddings (np.ndarray): Input embeddings (n_samples x n_features).
    - params (dict): Autoencoder parameters:
        - hidden_layers: List of hidden layer sizes.
        - bottleneck_dim: Size of the bottleneck layer (reduced dimension).
        - epochs: Number of training epochs.
        - batch_size: Batch size for training.
        - learning_rate: Learning rate for optimization.
    
    Returns:
    - np.ndarray: Reduced embeddings from the bottleneck layer.
    """
    # Extract parameters
    hidden_layers = params.get('hidden_layers', [512, 256])
    bottleneck_dim = params.get('bottleneck_dim', 50)
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', 128)
    learning_rate = params.get('learning_rate', 1e-3)

    # Initialize Autoencoder
    input_dim = embeddings.shape[1]
    autoencoder = Autoencoder(input_dim, hidden_layers, bottleneck_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Convert embeddings to PyTorch tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train Autoencoder
    autoencoder.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            reconstructed = autoencoder(batch[0])
            loss = criterion(reconstructed, batch[0])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Extract reduced embeddings
    autoencoder.eval()
    with torch.no_grad():
        reduced_embeddings = autoencoder.encoder(embeddings_tensor).numpy()

    return reduced_embeddings