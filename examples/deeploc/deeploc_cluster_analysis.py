import pandas as pd

from emma.ema import EmbeddingHandler

fp_metadata = "examples/deeploc/data/deeploc_train_features.csv"
embedding_dir = "embeddings/"
models = {#"ESM2_small": "esm2_t6_8M_UR50D/layer_6/chopped_1022_overlap_300", 
          #"ESM2_medium": "esm2_t30_150M_UR50D/layer_30/chopped_1022_overlap_300",
          "ESM2_medium": "esm2_t33_650M_UR50D/layer_33/chopped_1022_overlap_300",
          #"ProstT5": "Rostlab/ProstT5/layer_None/chopped_1022_overlap_300",
          "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"
          }

def categorize_length(length):
    if length < 100:
        return '< 100'
    elif length < 500:
        return '100 - 499'
    elif length < 1000:
        return '500 - 9999'
    else:
        return '>= 1000'

metadata = pd.read_csv(fp_metadata)
metadata['length_categories'] = metadata['sequence_length'].apply(categorize_length)
ema = EmbeddingHandler(sample_meta_data=metadata)

for model_alias, model_name in models.items(): 
    ema.add_emb_space(embeddings_source=embedding_dir + model_name, 
                      emb_space_name=model_alias)

fig = ema.visualise_emb_pca(
    emb_space_name="ProtT5",
    colour="length_categories"
)

ema.calculate_clusters(
    embedding_space="ProstT5",
    algorithm="kmeans",
    n_clusters=80,
    dim_reduction_method="PCA"
)

# ema.store_clustering(
#         embedding_space="ProstT5",
#         clustering_params="kmeans",
#         n_clusters=80,
#         dim_reduction_method="PCA"
#     )

ema.calculate_clusters(
    embedding_space="ProstT5",
    algorithm="kmeans",
    n_clusters=15,
    dim_reduction_method="PCA"
)

ema.calculate_clusters(
    embedding_space="ProstT5",
    algorithm="kmeans",
    n_clusters=70,
)
ema.calculate_clusters(
    embedding_space="ProstT5",
    algorithm="spectral",
    n_clusters=70,
    dim_reduction_method="PCA"
)
ema.calculate_clusters(
    embedding_space="ProstT5",
    algorithm="DBSCAN",
    dim_reduction_method="TSNE"
)


ema.remove_clustering(embedding_space="ProstT5", clustering_algorithm="DBSCAN", dim_reduction_method="TSNE")

fig = ema.summarise_clustering_performance(
        embedding_space='ProstT5',
        evaluation_method="silhouette_score",
        output = "plot",
        evaluation_params={"distance_metric": "euclidean"}
    )
score = ema.calculate_clusters(
    embedding_space="esm2_t30_150M_UR50D",
    algorithm="kmeans",
    n_clusters=80,
    dim_reduction_method="PCA"
)





ema.get_clustering_summary()

score = ema.evaluate_clustering(
    embedding_space="esm2_t30_150M_UR50D",
    clustering_algorithm="kmeans",
    n_clusters=55,
    dim_reduction_method="PCA",
    evaluation_method="silhouette_score",
    distance_metric="euclidean"
)
    
score = ema.evaluate_clustering(
    embedding_space="esm2_t30_150M_UR50D",
    clustering="cluster",
    evaluation_method="silhouette_score",
    distance_metric="euclidean"
)

xf = ema.evaluate_clustering_per_cluster(
    embedding_space="esm2_t30_150M_UR50D",
    clustering="cluster",
    evaluation_method="silhouette_score",
    distance_metric="euclidean"
)
    
ema.plot_heatmap(x_axis="cluster_esm2_t6_8M_UR50D",
                 y_axis='cluster_esm2_t30_150M_UR50D')

ema.visualise_emb_tsne(
    emb_space_name='esm2_t30_150M_UR50D/layer_30/chopped_1022_overlap_300',
    colour="subcellular_location"
)
