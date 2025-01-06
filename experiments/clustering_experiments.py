import itertools
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os 

from emma.ema import EmbeddingHandler

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv"
embedding_dir = "embeddings/"
models = {#"ESM2_small": "esm2_t6_8M_UR50D/layer_6/chopped_1022_overlap_300", 
          #"ESM2_medium": "esm2_t33_650M_UR50D/layer_33/chopped_1022_overlap_300",
          #"ProstT5": "Rostlab/ProstT5/layer_None/chopped_1022_overlap_300",
          "ESM2": "esm2_t36_3B_UR50D/layer_36/chopped_1022_overlap_300",
          "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"
          }

output_dir = "experiments/Pla2g2/clustering_results"
    
CLUSTERING_PARAMETERS = {
    # "kmeans": {
    #     "random_state": [42, 97],
    #     "n_clusters": [10, 100, 1000],
    #     "dim_reduction_method": [None, "PCA", "TSNE"]
    # },
    # "minibatch_kmeans": {
    #     "random_state": [42, 97],
    #     "n_clusters": [10, 100, 1000],
    #     "batch_size": [100, 500],
    #     "max_iter": [50, 200], 
    #     "random_state": [42, 97],
    #     "dim_reduction_method": [None, "PCA", "Autoencoder"],
    #     "normalise": [None]
    # },
    "minibatch_kmeans": {
        "random_state": [42, 97],
        "n_clusters": [10, 40, 100],
        "batch_size": [40, 100],
        "max_iter": [50, 200],
        "random_state": [42, 97],
        "dim_reduction_method": [None, "PCA", "Autoencoder"],
        "normalise": [None]
    },
    "hdbscan": {
        "min_cluster_size": [2, 10, 100],
        #"cluster_selection_epsilon": [],
        "metric": ["euclidean", "cityblock"], #manhattan, cosine
        "dim_reduction_method": [None, "PCA", "Autoencoder"],
    },
    "agglomerative": {
        "n_clusters": [10, 40, 100],
        'metric': ['euclidean', 'cityblock'], 
        'linkage': ['ward', 'complete', 'average', 'single'],
        "dim_reduction_method": [None, "PCA", "Autoencoder"],
    }
}

def k_distance_graph(
    data: np.ndarray,
    k: int,
):
    # Fit k-NN to the data
    k = 5  # Choose k based on min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)  # Replace `data` with your dataset
    distances, _ = nbrs.kneighbors(data)

    # Sort the distances of the kth neighbor
    distances = np.sort(distances[:, k - 1])
    plt.plot(distances)
    plt.xlabel("Samples")
    plt.ylabel(f"{k}-distance")
    plt.title("k-distance Graph")
    return plt

metadata = pd.read_csv(fp_metadata)

for model_alias, model_name in models.items(): 
    ema = EmbeddingHandler(sample_meta_data=metadata)
    ema.add_emb_space(embeddings_source=embedding_dir + model_name, 
                      emb_space_name=model_alias)
    
    for algorithm, params in CLUSTERING_PARAMETERS.items():
        
        param_grid = list(itertools.product(*params.values()))
        param_names = list(params.keys())
        
        for param_combination in param_grid:
            param_dict = dict(zip(param_names, param_combination))
            
            if (algorithm == "agglomerative") & (param_dict.get('linkage', "") == "ward") & (param_dict.get('metric', "") != "euclidean"):
                print(f"Skipping parameter combination {param_dict} \
                    as ward linkage can only be calculated with euclidean distance.")
                continue
            
            # Determine dimensionality reduction method
            dim_reduction_method = param_dict.get("dim_reduction_method", "no_dimensionality_reduction")
            if dim_reduction_method is None:
                dim_reduction_method = "no_dimensionality_reduction"
                
            # Construct directory path
            clustering_dir = os.path.join(
                output_dir,
                model_alias,
                algorithm,
                dim_reduction_method,
                "-".join([f"{k}={v}" for k, v in param_dict.items() if k != "dim_reduction_method"])
            )
            
            # File path for labels
            npy_path = os.path.join(clustering_dir, "labels.npy")
            
            # Check if the file already exists
            if os.path.exists(npy_path):
                print(f"Labels already exist for {model_alias}, {algorithm}, {param_dict}. Skipping calculation.")
                continue
            
            print(param_combination)
            
            ema.calculate_clusters(
                embedding_space=model_alias,
                algorithm=algorithm,
                **param_dict
            )
            
            clustering_params = {k: v for k, v in param_dict.items() if (k != "dim_reduction_method") and (k != "normalise")}
            dim_reduction_method = param_dict.get("dim_reduction_method", None)
            normalise = param_dict.get("normalise", None)
            
            clustering_result = ema.get_clustering_results(embedding_space=model_alias,
                algorithm=algorithm,
                clustering_params=clustering_params,
                dim_reduction_method=dim_reduction_method,
                normalise=normalise)
            
            if clustering_result:
                # Retrieve labels
                labels = clustering_result["labels"]
                
                # Create directories if they do not exist
                os.makedirs(clustering_dir, exist_ok=True)
                
                # Save labels as a .npy file
                np.save(npy_path, labels)
                print(f"Saved labels to: {npy_path}")
            else:
                print(f"No clustering result found for {model_alias}, {algorithm}, {param_dict}")
            
    # df = ema.summarise_clustering_performance(
    #     embedding_space=model_alias,
    #     evaluation_method="silhouette_score",
    #     evaluation_params={"distance_metric": "manhattan"},
    #     output="df"
    # )
    # print()