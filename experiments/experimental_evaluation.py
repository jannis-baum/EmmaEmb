
import os
import numpy as np
import pandas as pd
from emma.ema import EmbeddingHandler

from emma.cluster_metrics import compare_clusterings

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv" #"examples/deeploc/data/deeploc_train_features.csv"
embedding_dir = "embeddings/"
models = {#"ESM2_small": "esm2_t6_8M_UR50D/layer_6/chopped_1022_overlap_300", 
          #"ESM2_medium": "esm2_t33_650M_UR50D/layer_33/chopped_1022_overlap_300",
          #"ProstT5": "Rostlab/ProstT5/layer_None/chopped_1022_overlap_300",
          "ESM2": "esm2_t36_3B_UR50D/layer_36/chopped_1022_overlap_300",
          "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300"
          }

output_dir = "experiments/Pla2g2/clustering_results"

metadata = pd.read_csv(fp_metadata)


def find_all_experimental_results(output_dir, ema):
    fetched_clusterings = {}
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if "labels.npy" in files:
                relative_path = os.path.relpath(root, output_dir)
                path_parts = relative_path.split(os.sep)
                
                if len(path_parts) > 4:
                    print(f"Skipping invalid path {root}")
                    continue
                
                embedding_space = path_parts[0]
                algorithm = path_parts[1]
                dim_reduction_method = path_parts[2]
                param_string = path_parts[3]
                
                if dim_reduction_method == "no_dimensionality_reduction":
                    dim_reduction_method = None
                
                clustering_params = {}
                for param in param_string.split("-"):
                    try:
                        key, value = param.split("=")
                        if value.isdigit():
                            value = int(value)
                        else:
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        clustering_params[key] = value
                    except ValueError:
                        print(f"Skipping invalid parameter string: {param}")
                        continue
                
                labels_path = os.path.join(root, "labels.npy")
                labels = np.load(labels_path)
                
                normalise = clustering_params.get("normalise", None)
                if normalise == "None":
                    normalise = None
                clustering_params={k:v for k,v in clustering_params.items() if k != "normalise"}
                
                ema.add_clustering_result(
                    embedding_space=embedding_space,
                    algorithm=algorithm,
                    labels=labels,
                    dim_reduction_method=dim_reduction_method,
                    clustering_params=clustering_params,
                    normalise = normalise
                )
                fetched_clusterings[(embedding_space, 
                                    algorithm, 
                                    tuple(clustering_params.items()))] = labels

    return fetched_clusterings, ema
            
            
ema = EmbeddingHandler(sample_meta_data=metadata)         
for model_alias, model_name in models.items():    
    ema.add_emb_space(embeddings_source=embedding_dir + model_name, 
                      emb_space_name=model_alias)
    
fetched_clusterings, ema = find_all_experimental_results(
    output_dir=output_dir,
    ema=ema
)


ema.set_default_clustering(
    embedding_space="ProtT5",
    clustering_algorithm="agglomerative",
    clustering_params={'metric': "euclidean", 'linkage': 'average', 'n_clusters': 40}
)

ema.set_default_clustering(
    embedding_space="ESM2",
    clustering_algorithm="agglomerative",
    clustering_params={'metric': "euclidean", 'linkage': 'average', 'n_clusters': 40}
)

score = compare_clusterings(
    ema.meta_data['cluster_ESM2'], ema.meta_data['gene'], score_method = "NMI"
)

# Ensure both clusterings are numpy arrays
cluster_ESM2 = np.array(ema.meta_data['cluster_ESM2'], dtype=str)
cluster_ProtT5 = np.array(ema.meta_data['cluster_ProtT5'], dtype=str)

# Get unique cluster labels
unique_esm2 = np.unique(cluster_ESM2)
unique_prott5 = np.unique(cluster_ProtT5)

contingency_matrix = np.zeros((len(unique_esm2), len(unique_prott5)), dtype=int)

# Loop through each pair of cluster labels and count occurrences
for i, esm2_label in enumerate(unique_esm2):
    for j, prott5_label in enumerate(unique_prott5):
        contingency_matrix[i, j] = np.sum((cluster_ESM2 == esm2_label) & (cluster_ProtT5 == prott5_label))

df = pd.DataFrame(contingency_matrix, 
                  columns=['Cluster_ProtT5_' + str(i) for i in range(contingency_matrix.shape[1])],
                  index=['Cluster_ESM2_' + str(i) for i in range(contingency_matrix.shape[0])])
df_sorted = df.loc[df.sum(axis=1).sort_values().index, df.sum(axis=0).sort_values().index]

import plotly.express as px

fig = px.imshow(df_sorted, text_auto=True, color_continuous_scale='Viridis')
fig.update_layout(title="Contingency Matrix Heatmap", 
                  xaxis_title="Clustering 2", 
                  yaxis_title="Clustering 1")
fig.show()

df_percentage = df_sorted.div(df_sorted.sum(axis=0), axis=1) * 100

# Create a heatmap using plotly.express
fig = px.imshow(df_percentage, 
                labels={'x': 'Clustering 2', 'y': 'Clustering 1'},
                x=df_percentage.columns, 
                y=df_percentage.index,
                color_continuous_scale='YlGnBu', 
                title="Contingency Matrix Heatmap (Percentage of Column Sum)",
                color_continuous_midpoint=50)

# Show the plot
fig.show()

cluster_group= np.array(ema.meta_data['group'], dtype=str)
unique_group = np.unique(cluster_group)

contingency_matrix_group_esm2 = np.zeros((len(unique_esm2), len(unique_group)), dtype=int)
contingency_matrix_group_prott5 = np.zeros((len(unique_prott5), len(unique_group)), dtype=int)

for i, esm2_label in enumerate(unique_esm2):
    for j, group_label in enumerate(unique_group):
        contingency_matrix_group_esm2[i, j] = np.sum((cluster_ESM2 == esm2_label) & (cluster_group == group_label))
        

for i, prott5_label in enumerate(unique_prott5):
    for j, group_label in enumerate(unique_group):
        contingency_matrix_group_prott5[i, j] = np.sum((cluster_ProtT5 == prott5_label) & (cluster_group == group_label))
   


df = pd.DataFrame(contingency_matrix_group_esm2, 
                  columns=[unique_group[i] for i in range(contingency_matrix_group_esm2.shape[1])],
                  index=['Cluster_ESM2_' + str(i) for i in range(contingency_matrix_group_esm2.shape[0])])
df_sorted = df.loc[df.sum(axis=1).sort_values().index, df.sum(axis=0).sort_values().index]

import plotly.express as px

fig = px.imshow(df_sorted, text_auto=True, color_continuous_scale='Viridis')
fig.update_layout(title="Contingency Matrix Heatmap", 
                  xaxis_title="Clustering 2", 
                  yaxis_title="Clustering 1")
fig.show()

df_percentage = df_sorted.div(df_sorted.sum(axis=0), axis=1) * 100


# Create a heatmap using plotly.express
fig = px.imshow(df_percentage, 
                labels={'x': 'Clustering 2', 'y': 'Clustering 1'},
                x=df_percentage.columns, 
                y=df_percentage.index,
                color_continuous_scale='YlGnBu', 
                title="Contingency Matrix Heatmap (Percentage of Column Sum)",
                color_continuous_midpoint=50)

# Show the plot
fig.show()



df = pd.DataFrame(contingency_matrix_group_prott5, 
                  columns=[unique_group[i] for i in range(contingency_matrix_group_prott5.shape[1])],
                  index=['Cluster_ProtT5_' + str(i) for i in range(contingency_matrix_group_prott5.shape[0])])
df_sorted = df.loc[df.sum(axis=1).sort_values().index, df.sum(axis=0).sort_values().index]

import plotly.express as px

fig = px.imshow(df_sorted, text_auto=True, color_continuous_scale='Viridis')
fig.update_layout(title="Contingency Matrix Heatmap", 
                  xaxis_title="Clustering 2", 
                  yaxis_title="Clustering 1")
fig.show()

df_percentage = df_sorted.div(df_sorted.sum(axis=0), axis=1) * 100


# Create a heatmap using plotly.express
fig = px.imshow(df_percentage, 
                labels={'x': 'Clustering 2', 'y': 'Clustering 1'},
                x=df_percentage.columns, 
                y=df_percentage.index,
                color_continuous_scale='YlGnBu', 
                title="Contingency Matrix Heatmap (Percentage of Column Sum)",
                color_continuous_midpoint=50)

# Show the plot
fig.show()


fig = ema.summarise_clustering_performance(
        embedding_space="ESM2",
        evaluation_method="silhouette_score",
        evaluation_params={"distance_metric": "euclidean"},
        output="plot"
    )

fig = ema.summarise_clustering_performance(
        embedding_space="ESM2_medium",
        evaluation_method="calinski_harabasz_score",
        evaluation_params={"distance_metric": "euclidean"},
        output="plot"
    )

df = ema.get_clustering_summary(
)

ema.set_default_clustering(
    embedding_space="ESM2",
    clustering_algorithm="agglomerative",
    clustering_params={'metric': "euclidean", 'linkage': 'average', 'n_clusters': 40}
)



fig = ema.plot_sample_cluster_distribution(
    embedding_space="ESM2_medium",
    algorithm="minibatch_kmeans",
    clustering_params={'random_state': 42, 'n_clusters': 10, 'batch_size': 100, 'max_iter': 200},
    dim_reduction_method = None,
)




print()