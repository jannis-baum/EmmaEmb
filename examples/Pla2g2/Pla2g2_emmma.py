import pandas as pd

from emma.ema import EmbeddingHandler

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv"
embedding_dir = "embeddings/"
models = {"ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
          "ESM2": "esm2_t36_3B_UR50D/layer_36/chopped_1022_overlap_300",
          }

metadata = pd.read_csv(fp_metadata)
ema = EmbeddingHandler(sample_meta_data=metadata)

for model_alias, model_name in models.items(): 
    ema.add_emb_space(embeddings_source=embedding_dir + model_name, 
                      emb_space_name=model_alias)
    
fig = ema.visualise_emb_pca(
    emb_space_name="ProtT5",
    colour="length_bin"
)

from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import numpy as np
import plotly.express as px
from scipy.sparse.csgraph import connected_components


knn1 = kneighbors_graph(ema.emb['ProtT5']['emb'], n_neighbors=30, mode='distance', metric="cosine")
knn2 = kneighbors_graph(ema.emb['ESM2']['emb'], n_neighbors=30, mode='distance',  metric="cosine")

# Check connectivity of the graphs
n_components_1, labels_1 = connected_components(knn1, directed=False)
n_components_2, labels_2 = connected_components(knn2, directed=False)

print(f"Number of components in knn1: {n_components_1}")
print(f"Number of components in knn2: {n_components_2}")

import leidenalg as la
import igraph as ig
import networkx as nx
from scipy.sparse import csr_matrix

# Step 2: Ensure symmetry (make the matrix square)
knn_graph = knn_graph.maximum(knn1.T)  # Ensure bidirectional edges

# Step 3: Convert to binary adjacency matrix
binary_graph = csr_matrix(knn_graph).astype(bool)

# Step 4: Create an igraph graph from the binary adjacency matrix
ig_graph = ig.Graph.Adjacency(binary_graph.toarray().tolist())


ig_graph = ig.Graph.Adjacency((graph > 0).tolist())

# Apply Leiden algorithm
partition = la.find_partition(ig_graph, la.RBConfigurationVertexPartition)
clusters = partition.membership


geo1 = shortest_path(knn1, directed=False, unweighted=False, method='auto')
geo2 = shortest_path(knn2, directed=False, unweighted=False, method='auto')

geodesic_diff = np.abs(geo1 - geo2)

indices = range(geodesic_diff.shape[0])
df = pd.DataFrame(geodesic_diff, index=indices, columns=indices)
df_long = df.reset_index().melt(id_vars="index", var_name="Point 2", value_name="Geodesic Difference")

from sklearn.manifold import TSNE
import numpy as np

# Use TSNE to reduce dimensions to 2D
proj1 = TSNE(n_components=2, random_state=42).fit_transform(ema.emb['ProtT5']['emb'])
proj2 = TSNE(n_components=2, random_state=42).fit_transform(ema.emb['ESM2']['emb'])

# Identify points with high divergence
pointwise_diff = geodesic_diff.mean(axis=1)  # Mean difference for each point
top_indices = np.argsort(-pointwise_diff)[:10]  # Top 10 divergent points

import plotly.express as px
import pandas as pd

# Prepare DataFrames for Plotly
df_proj1 = pd.DataFrame(proj1, columns=["x", "y"])
df_proj1["Embedding"] = "Embedding 1"
df_proj1["Divergence"] = pointwise_diff
df_proj1["Highlight"] = ["High" if i in top_indices else "Low" for i in range(len(proj1))]

df_proj2 = pd.DataFrame(proj2, columns=["x", "y"])
df_proj2["Embedding"] = "Embedding 2"
df_proj2["Divergence"] = pointwise_diff
df_proj2["Highlight"] = ["High" if i in top_indices else "Low" for i in range(len(proj2))]

# Combine DataFrames for both embeddings
df_combined = pd.concat([df_proj1, df_proj2])

df_combined['sample_name'] = ema.sample_names * 2
df_combined['group'] = ema.meta_data['group'] * 2

# Create interactive scatter plot
fig = px.scatter(
    df_combined,
    x="x",
    y="y",
    symbol="Highlight",
    color="group",
    facet_col="Embedding",
    hover_data=['sample_name'],
    size="Divergence",
    title="2D Projection with High Divergence Points Highlighted",
    labels={"x": "Projection X", "y": "Projection Y"},
)

fig.update_traces(marker=dict(opacity=0.7))
fig.update_layout(width=1000, height=500)
fig.show()



fig = ema.plot_emb_dis_scatter(
    emb_space_name_1="ProtT5",
    emb_space_name_2="ESM2",
    distance_metric="euclidean",
    colour_group="gene",
    colour_value_1="Pla2g2B"
)

fig = ema.plot_emb_dis_dif_heatmap(
    emb_space_name_1="ProtT5",
    emb_space_name_2="ESM2",
    distance_metric="sqeuclidean",
)

ema.calculate_clusters(
    embedding_space="ProtT5",
    n_clusters=10,
    dim_reduction_method="PCA",
    algorithm="kmeans",
)

ema.set_default_clustering(
    embedding_space="ProtT5",
    clustering_params={"n_clusters":10},
    dim_reduction_method="PCA",
    clustering_algorithm="kmeans",
)

fig = ema.plot_cluster_distribution(
    embedding_space="ProtT5",
    clustering_params={"n_clusters":10},
    dim_reduction_method="PCA",
    clustering_algorithm="kmeans",
)

print()





fig = ema.summarise_clustering_performance(
    embedding_space="ESM2",
    evaluation_method="silhouette_score",
    evaluation_params={
        "distance_metric": "manhattan"
    },
    output="plot"
)

ema.evaluate_clustering(
    embedding_space="ESM2",
    clustering_algorithm="kmeans",
    evaluation_method="silhouette_score",
    distance_metric="euclidean"
)