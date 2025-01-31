import pandas as pd
from emmaemb.core import Emma
from emmaemb.vizualization import *
from emmaemb.functions import *

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv"
# fp_metadata = "examples/deeploc/data/deeploc_train_features.csv"
embedding_dir = "embeddings/"
models = {
    "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
    "ESMC": "esmc-300m-2024-12/layer_None/chopped_1022_overlap_300",
}

metadata = pd.read_csv(fp_metadata)
emma = Emma(feature_data=metadata)

for model_alias, model_name in models.items():
    emma.add_emb_space(
        emb_space_name=model_alias,
        embeddings_source=embedding_dir + model_name,
    )

# visualise reduced embedding space
fig_pca = plot_emb_space(emma, "ProtT5", color_by="enzyme_class")

pwd = emma.get_pairwise_distances("ProtT5", "cityblock")
pwd = emma.get_pairwise_distances("ESMC", "cityblock")

# HEATMAP OF PAIRWISE DISTANCES
fig_pwd_heatmap = plot_pairwise_distance_heatmap(
    emma, "ProtT5", metric="cityblock", group_by="enzyme_class"
)

# SCATTER PLOT OF PAIRWISE DISTANCES
fig_pwd_comparison = plot_pairwise_distance_comparison(
    emma,
    emb_space_y="ProtT5",
    emb_space_x="ESMC",
    metric="cityblock",
    group_by="species",
)

# KNN ALIGNMENT SCORES
fig_alignment_scores = plot_knn_alignment_across_embedding_spaces(
    emma, feature="enzyme_class", k=10, metric="cityblock"
)

fig_alignment_scores_class = plot_knn_alignment_across_classes(
    emma, feature="enzyme_class", k=100, metric="cityblock"
)

# KNN CLASS MIXING MATRIX
fig_class_mixing_matrix = plot_knn_class_mixing_matrix(
    emma,
    emb_space="ProtT5",
    feature="enzyme_class",
    k=100,
    metric="cityblock",
)

# LOW SIMILARITY CLASS DISTRIBUTION
fig_low_similarity_class_distribution = plot_low_similarity_distribution(
    emma,
    emb_space_1="ProtT5",
    emb_space_2="ESMC",
    feature="enzyme_class",
    k=10,
    metric="cityblock",
    similarity_threshold=0.3,
)

print()
