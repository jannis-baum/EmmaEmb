import pandas as pd

from emma.ema import EmbeddingHandler

fp_metadata = "examples/deeploc/data/deeploc_train_subset_features.csv"
embedding_dir = "embeddings/"
models = ['esm2_t6_8M_UR50D/layer_6/chopped_1022_overlap_300', 
          'esm2_t30_150M_UR50D/layer_30/chopped_1022_overlap_300']


metadata = pd.read_csv(fp_metadata)
ema = EmbeddingHandler(sample_meta_data=metadata)

for model in models: 
    ema.add_emb_space(embeddings_source=embedding_dir + model, 
                      emb_space_name=model.split("/")[0])
    
ema.calculate_clusters(
    embedding_space="esm2_t30_150M_UR50D",
    algorithm="kmeans",
    n_clusters=80,
    dim_reduction_method="PCA"
)

ema.calculate_clusters(
    embedding_space="esm2_t30_150M_UR50D",
    algorithm="kmeans",
    n_clusters=15,
    dim_reduction_method="PCA"
)

ema.calculate_clusters(
    embedding_space="esm2_t30_150M_UR50D",
    algorithm="kmeans",
    n_clusters=70,
)
ema.calculate_clusters(
    embedding_space="esm2_t30_150M_UR50D",
    algorithm="spectral",
    n_clusters=70,
    dim_reduction_method="PCA"
)

fig = ema.summarise_clustering_performance(
        embedding_space='esm2_t30_150M_UR50D',
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
