import itertools
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from emma.ema import EmbeddingHandler

fp_metadata = "examples/deeploc/data/deeploc_train_features.csv"
embedding_dir = "embeddings/"
models = {#"ESM2_small": "esm2_t6_8M_UR50D/layer_6/chopped_1022_overlap_300", 
          #"ESM2_medium": "esm2_t30_150M_UR50D/layer_30/chopped_1022_overlap_300",
          "ProstT5": "Rostlab/ProstT5/layer_None/chopped_1022_overlap_300",
          # "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"
          }


    
CLUSTERING_PARAMETERS = {
    # "kmeans": {
    #     "random_state": [42, 97],
    #     "n_clusters": [10, 50, 100, 500, 1000, 2000],
    #     "dim_reduction_method": [None] #"PCA", "TSNE"
    # },
    "minibatch_kmeans": {
        "n_clusters": [10, 20, 50, 100, 150, 200],
        "batch_size": [100, 500, 1000],
        "max_iter": [50, 100, 200], 
        "random_state": [42]#97
    }
    # "dbscan": {
    #     "random_state": [97, 98, 99],
        
    # }
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
            
            ema.calculate_clusters(
                embedding_space=model_alias,
                algorithm=algorithm,
                **param_dict
            )
    df = ema.summarise_clustering_performance(
        embedding_space=model_alias,
        evaluation_method="silhouette_score",
        evaluation_params={"distance_metric": "manhattan"},
        output="df"
    )
    print()