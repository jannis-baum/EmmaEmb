import pandas as pd
import numpy as np

from sklearn.neighbors import kneighbors_graph
import plotly.express as px

from emma.ema import EmbeddingHandler
from emma.knn_analysis import plot_heatmap_feature_distribution, analyze_low_similarity_distribution


# parameter for this script
figures_to_be_plotted = [
    # 'Fig_A1',
    # 'Fig_A2',
    #'Fig_A3',
    #'Fig_B1',
    'Fig_B2',
    'Fig_B3'
]

output_dir = "figures/"
distance_metric = "cosine"
k_neighbors = 10

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv"
embedding_dir = "embeddings/"
models = {
        "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
        "ESMC": "esmc-300m-2024-12/layer_None/chopped_1022_overlap_300"
          }

metadata = pd.read_csv(fp_metadata)
ema = EmbeddingHandler(sample_meta_data=metadata)

for model_alias, model_name in models.items(): 
    ema.add_emb_space(embeddings_source=embedding_dir + model_name, 
                      emb_space_name=model_alias)


if 'Fig_A1' in figures_to_be_plotted:
    
    fig_A1 = ema.plot_emb_dis_scatter(
        emb_space_name_1 = "ESMC",
        emb_space_name_2 = "ProtT5",
        distance_metric=distance_metric,
    )
    
    fig_A1.update_layout(
        title = None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        
        xaxis_title = "Cosine distances ESMC",
        xaxis_title_font=dict(size=26), 
        
        yaxis_title = "Cosine distances ProtT5",
        yaxis_title_font=dict(size=26),
        
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        
        xaxis = dict(range=[0, 1]),
        
        yaxis = dict(range=[0, 1]),
        
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    fig_A1.update_traces(marker=dict(size=8,
                                     color="grey"))
    
    fig_A1.write_image(output_dir + "fig_4_A1.pdf", 
                      format="pdf",
                      width=600, 
                      height=600)

if "Fig_A2" in figures_to_be_plotted:
    
    fig_A2 = ema.plot_emb_dis_scatter(
        emb_space_name_1 = "ESMC",
        emb_space_name_2 = "ProtT5",
        distance_metric=distance_metric,
        colour_group="species",
        colour_value_1="birds",
        colour_value_2="birds",
    )
    
    fig_A2.update_layout(
        title = None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        
        xaxis_title = "Cosine distances ESMC",
        xaxis_title_font=dict(size=26), 
        
        yaxis_title = "Cosine distances ProtT5",
        yaxis_title_font=dict(size=26),
        
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        
        xaxis = dict(range=[0, 1]),
        yaxis = dict(range=[0, 1]),
        
        showlegend=False,
        
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    fig_A2.update_traces(marker=dict(size=8))
    fig_A2.data[1].marker.color = "darkred"
    
    fig_A2.write_image(output_dir + "fig_4_A2.pdf", 
                      format="pdf",
                      width=600, 
                      height=600)
    
if "Fig_A3" in figures_to_be_plotted:
    
    fig_A3 = ema.plot_emb_dis_scatter(
        emb_space_name_1 = "ESMC",
        emb_space_name_2 = "ProtT5",
        distance_metric=distance_metric,
        colour_group="species",
        colour_value_1="crocodile",
        colour_value_2="crocodile",
    )
    
    fig_A3.update_layout(
        title = None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        
        xaxis_title = "Cosine distances ESMC",
        xaxis_title_font=dict(size=26), 
        
        yaxis_title = "Cosine distances ProtT5",
        yaxis_title_font=dict(size=26),
        
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        
        xaxis = dict(range=[0, 1]),
        yaxis = dict(range=[0, 1]),
        
        showlegend=False,
        
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    fig_A3.update_traces(marker=dict(size=8))
    
    fig_A3.data[1].marker.color = "orange"
    
    fig_A3.write_image(output_dir + "fig_4_A3.pdf", 
                      format="pdf",
                      width=600, 
                      height=600)
    


if 'Fig_B1' in figures_to_be_plotted:
    
    fig_B1 = analyze_low_similarity_distribution(
        ema,
        "ESMC",
        "ProtT5",
        "species",
        similarity_threshold=0.3,
        k_neighbors=10,
        distance_metric="cosine"     
    )
    
    tick_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig_B1.update_layout(
        legend=dict(
            font=dict(
                size=18 
            ),
            title="Species",
        ),
        
        title = None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        
        xaxis_title = "% in dataset",
        xaxis_title_font=dict(size=26), 
        
        yaxis_title = "% in low similarity subset",
        yaxis_title_font=dict(size=26),
        
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        
        xaxis = dict(range=[-0.05, 0.7],
                     tickvals=tick_vals,  
                     tickformat=".0%",
                     dtick=0.2,
                     tickwidth=6,  
                     tickcolor="black"),
        yaxis = dict(range=[-0.05, 0.7],
                     tickformat=".0%",
                     dtick=0.2),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    fig_B1.update_traces(marker=dict(size=12))
    
    fig_B1.write_image(output_dir + "fig_4_B1.pdf", 
                      format="pdf",
                      width=600, 
                      height=400)
    
if 'Fig_B2' in figures_to_be_plotted:
    
    fig_B2 = analyze_low_similarity_distribution(
        ema,
        "ESMC",
        "ProtT5",
        "enzyme_class",
        similarity_threshold=0.3,
        k_neighbors=10,
        distance_metric="cosine"     
    )
    
    tick_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig_B2.update_layout(
        legend=dict(
            font=dict(
                size=18
            ),
            title="Enzyme class",
        ),
        
        title = None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        
        xaxis_title = "% in dataset",
        xaxis_title_font=dict(size=26), 
        
        yaxis_title = "% in low similarity subset",
        yaxis_title_font=dict(size=26),
        
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        
        xaxis = dict(range=[-0.05, 0.4],
                     tickvals=tick_vals,  
                     tickformat=".0%",
                     dtick=0.2,
                     tickwidth=6,  
                     tickcolor="black"),
        yaxis = dict(range=[-0.05, 0.4],
                     tickformat=".0%",
                     dtick=0.2),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    fig_B2.update_traces(marker=dict(size=12))
    
    fig_B2.write_image(output_dir + "fig_4_B2.pdf", 
                      format="pdf",
                      width=600, 
                      height=400)

if 'Fig_B3' in figures_to_be_plotted:
    
    fig_B3 = analyze_low_similarity_distribution(
        ema,
        "ESMC",
        "ProtT5",
        "length_bin",
        similarity_threshold=0.3,
        k_neighbors=10,
        distance_metric="cosine"     
    )
    
    tick_vals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    fig_B3.update_layout(
        legend=dict(
            font=dict(
                size=18
            ),
            title="Sequence length",
        ),
        
        title = None,
        # title="A. All pairwise distances between PLA2G2 embeddings<br>All embedding paris.",
        title_font=dict(size=26),
        
        xaxis_title = "% in dataset",
        xaxis_title_font=dict(size=26), 
        
        yaxis_title = "% in low similarity subset",
        yaxis_title_font=dict(size=26),
        
        xaxis_tickfont=dict(size=26),
        yaxis_tickfont=dict(size=26),
        
        xaxis = dict(range=[-0.05, 0.9],
                     tickvals=tick_vals,  
                     tickformat=".0%",
                     dtick=0.2,
                     tickwidth=6,  
                     tickcolor="black"),
        yaxis = dict(range=[-0.05, 0.9],
                     tickformat=".0%",
                     dtick=0.2),
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        )
    )
    fig_B3.update_traces(marker=dict(size=12))
    
    fig_B3.write_image(output_dir + "fig_4_B3.pdf", 
                      format="pdf",
                      width=600, 
                      height=400)
    
    print()






def compare_knn_fraction_across_embeddings(ema, feature_column, k_neighbors=10, title="Comparison of Fraction of Same-Class Neighbors"):
    """
    Compares the fraction of same-class neighbors across different embedding spaces using boxplots.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - feature_column: Column name in ema.meta_data for the feature to analyze.
    - k_neighbors: Number of neighbors to consider in KNN graph.
    - title: Title for the boxplot.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(embeddings, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i])
            fraction = same_class_count / k_neighbors
            fractions.append(fraction)

        # Collect results for this embedding
        df = pd.DataFrame({
            'Class': feature_classes,
            'Fraction': fractions,
            'Embedding': model_id  # Label the embedding space
        })
        all_results.append(df)

    # Combine results across all embeddings
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Plot comparison across embedding spaces
    fig = px.box(
        all_results_df,
        x='Embedding',
        y='Fraction',
        color='Embedding',
        title=title,
        labels={'Embedding': 'Embedding Space', 'Fraction': 'Fraction of k Nearest Neighbors in Same Class'},
        template='plotly_white'
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")  # Set font to Arial and black
    )
    
    # Display the boxplot
    fig.show()

compare_knn_fraction_across_embeddings(
    ema, 
    feature_column="group", 
    k_neighbors=10, 
    title="Comparison of Fraction of Same-Class Neighbors Across Embedding Spaces"
)



def plot_knn_fraction_heatmap(ema, feature_column, k_neighbors=10, title="KNN Fraction Heatmap"):
    """
    Generates a heatmap showing the mean fraction of same-class neighbors across embeddings.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - feature_column: Column name in ema.meta_data for the feature to analyze.
    - k_neighbors: Number of neighbors to consider in KNN graph.
    - title: Title for the heatmap.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(embeddings, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i])
            fraction = same_class_count / k_neighbors
            fractions.append(fraction)

        # Collect results for this embedding
        df = pd.DataFrame({
            'Class': feature_classes,
            'Fraction': fractions,
            'Embedding': model_id  # Label the embedding space
        })
        all_results.append(df)

    # Combine results across all embeddings
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Aggregate enrichment per embedding space
    heatmap_data = (
        all_results_df
        .groupby(['Class', 'Embedding'])['Fraction']
        .mean()  # Use mean or replace with median()
        .unstack()  # Reshape to have Classes as rows and Embeddings as columns
    )

    # Count samples per class for labels
    class_counts = all_results_df.groupby('Class').size()

    # Update heatmap index with sample counts
    heatmap_data.index = [
        f"{feature_class} (n = {count})" 
        for feature_class, count in zip(heatmap_data.index, class_counts[heatmap_data.index])
    ]

    # Generate heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Embedding Space", y="Feature Class (Samples)", color="Mean Fraction"),
        title=title,
        color_continuous_scale=[
            (0.0, "lightblue"),  # Light blue for the lowest value
            (1.0, "darkblue")    # Dark blue for the highest value
        ],
        text_auto=".2f"
    )

    # Update font settings for the heatmap
    fig.update_layout(
        font=dict(
            family="Arial",  # Set font to Arial
            #color="black"    # Set font color to black
        )
    )

    # Display heatmap
    fig.show()
# Example usage
plot_knn_fraction_heatmap(ema, feature_column="group", k_neighbors=10, 
                          title="Mean Fraction of Same-Class Neighbors")

from sklearn.neighbors import NearestNeighbors
import plotly.express as px

def compute_neighborhood_similarity(embeddings1, embeddings2, k):
    # Compute k-NN for embeddings1
    knn1 = NearestNeighbors(n_neighbors=k).fit(embeddings1)
    neighbors1 = knn1.kneighbors(return_distance=False)

    # Compute k-NN for embeddings2
    knn2 = NearestNeighbors(n_neighbors=k).fit(embeddings2)
    neighbors2 = knn2.kneighbors(return_distance=False)

    # Compute overlap for each point
    overlap_fraction = [
        len(set(neighbors1[i]).intersection(set(neighbors2[i]))) / k
        for i in range(len(embeddings1))
    ]
    return np.array(overlap_fraction)

def plot_boxplot_similarity_by_feature_class(embedding1_id, embedding2_id, feature, k):
    embeddings1 = ema.emb[embedding1_id]['emb']
    embeddings2 = ema.emb[embedding2_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Compute neighborhood similarity
    similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        "Feature Class": feature_classes,
        "Neighborhood Similarity": similarity_scores
    })
    
    # Plot boxplot
    fig = px.box(
        plot_data,
        x="Feature Class",
        y="Neighborhood Similarity",
        points="all",
        title=f"Neighborhood Similarity by Feature Class ({embedding1_id} vs {embedding2_id})",
        labels={"Neighborhood Similarity": "Fraction of Overlapping Neighbors"}
    )
    fig.update_layout(
        font=dict(family="Arial", color="black")  # Set font to Arial and black
    )
    fig.show()


# this is a great function!! 
def plot_heatmap_feature_distribution(
    embedding1_id, embedding2_id, feature, k, bins=5
):
    embeddings1 = ema.emb[embedding1_id]['emb']
    embeddings2 = ema.emb[embedding2_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Compute neighborhood similarity
    similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
    
    # Bin the similarity scores
    bin_edges = np.linspace(0, 1, bins + 1)  # Equal-width bins
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)]
    binned_scores = pd.cut(similarity_scores, bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        "Feature Class": feature_classes,
        "Similarity Bin": binned_scores
    })
    
    # Aggregate counts of feature classes in each bin
    distribution = (
        plot_data.groupby(["Similarity Bin", "Feature Class"])
        .size()
        .reset_index(name="Count")
    )
    
    # Pivot the data for heatmap
    heatmap_data = distribution.pivot(index="Feature Class", columns="Similarity Bin", values="Count")
    
    # Plot heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Similarity Bin", y="Feature Class", color="Count of Samples"),
        title=f"Feature Class Distribution Across Similarity Bins ({embedding1_id} vs {embedding2_id})",
        color_continuous_scale="Blues",  # Color scale from light blue (low) to dark blue (high)
        text_auto=".2f" # Display values in each cell
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()

plot_heatmap_feature_distribution(
    "ESMC", "ESM3", "group", k=10, bins=5
)


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

def compute_class_mixing_in_neighborhood(embeddings, feature_classes, k):
    """
    Calculate the overlap of neighbors' feature classes for each point.
    Returns a DataFrame with counts of each class in the k-nearest neighbors for each point.
    """
    # Fit the NearestNeighbors model to the embeddings
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    knn.fit(embeddings)
    
    # Get k-nearest neighbors for each point
    distances, indices = knn.kneighbors(embeddings)
    
    # Get the feature classes for each of the k-nearest neighbors
    neighbor_class_counts = np.zeros((len(embeddings), len(feature_classes.unique())))
    
    for i, idx in enumerate(indices):
        # Exclude the point itself (first in the neighbors list)
        neighbor_classes = feature_classes.iloc[idx[1:]]  # Skip the first neighbor (the point itself)
        
        # Count the occurrences of each feature class in the neighbors
        for class_label in neighbor_classes:
            class_index = np.where(feature_classes.unique() == class_label)[0][0]
            neighbor_class_counts[i, class_index] += 1
    
    return neighbor_class_counts, feature_classes.unique()

def plot_class_mixing_heatmap(embedding_id, feature, k):
    embeddings = ema.emb[embedding_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, k)

    # Convert counts to a DataFrame for easier plotting
    mixing_df = pd.DataFrame(mixing_counts, columns=class_labels)
    
    # Transpose to have feature classes of points on y-axis and neighbors' feature classes on x-axis
    mixing_df = mixing_df.T
    
    # Plot heatmap
    fig = px.imshow(
        mixing_df,  # We now transpose so feature classes are on the Y-axis
        labels=dict(x="Feature Class (Neighbor)", y="Feature Class (Sample)", color="Neighbor Count"),
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_id})",
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    
plot_class_mixing_heatmap("ESM3", "group", k=10)


# this is a great graphic!!:

def compute_class_mixing_in_neighborhood(embeddings, feature_classes, k):
    """
    Calculate the overlap of neighbors' feature classes for each point.
    Returns a DataFrame with counts of each class in the k-nearest neighbors for each point.
    """
    # Fit the NearestNeighbors model to the embeddings
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    knn.fit(embeddings)
    
    # Get k-nearest neighbors for each point
    distances, indices = knn.kneighbors(embeddings)
    
    # Initialize a matrix to store the counts of neighbors' feature classes
    neighbor_class_counts = np.zeros((len(feature_classes.unique()), len(feature_classes.unique())))
    
    # Loop over each sample and its neighbors
    for i, idx in enumerate(indices):
        # Exclude the point itself (the first in the neighbors list)
        neighbor_classes = feature_classes.iloc[idx[1:]]  # Skip the first neighbor (the point itself)
        
        # For each of the neighbors, increment the count for their feature class in the matrix
        sample_class = feature_classes.iloc[i]
        for neighbor_class in neighbor_classes:
            sample_class_index = np.where(feature_classes.unique() == sample_class)[0][0]
            neighbor_class_index = np.where(feature_classes.unique() == neighbor_class)[0][0]
            neighbor_class_counts[neighbor_class_index, sample_class_index] += 1
    
    return neighbor_class_counts, feature_classes.unique()

def plot_class_mixing_heatmap(embedding_id, feature, k):
    embeddings = ema.emb[embedding_id]["emb"]
    feature_classes = ema.meta_data[feature]
    
    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, k)

    # Convert counts to a DataFrame for easier plotting
    mixing_df = pd.DataFrame(mixing_counts, index=class_labels, columns=class_labels)
    
    # Plot heatmap
    fig = px.imshow(
        mixing_df,  # Feature classes on both axes
        labels=dict(x="Feature Class (Sample)", y="Feature Class (Neighbor)", color="Neighbor Count"),
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_id})",
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    
plot_class_mixing_heatmap("ESM3", "group", k=10)





from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path
import numpy as np
import plotly.express as px
from scipy.sparse.csgraph import connected_components


knn1 = kneighbors_graph(ema.emb['ESM3']['emb'], n_neighbors=5, mode='distance', metric="cosine")
knn2 = kneighbors_graph(ema.emb['ESMC']['emb'], n_neighbors=5, mode='distance',  metric="cosine")

# Check connectivity of the graphs
n_components_1, labels_1 = connected_components(knn1, directed=False)
n_components_2, labels_2 = connected_components(knn2, directed=False)

print(f"Number of components in knn1: {n_components_1}")
print(f"Number of components in knn2: {n_components_2}")

# 
k_neighbours = 10
feature = "group"
knn_graph = kneighbors_graph(ema.emb['ESM3']['emb'], 
                             n_neighbors=k_neighbours, mode='connectivity', include_self=False)
knn_matrix = knn_graph.toarray()

fractions = []
for i, neighbors in enumerate(knn_matrix):
    neighbor_indices = np.where(neighbors == 1)[0]
    same_class_count = np.sum(ema.meta_data["group"][neighbor_indices] == ema.meta_data["group"][i])
    fraction = same_class_count / k_neighbours
    fractions.append(fraction)

df = pd.DataFrame({
    'Class': ema.meta_data["group"],
    'Fraction': fractions
})

fig = px.box(
    df,
    x='Class',
    y='Fraction',
    title='Distribution of Fraction of Same-Class Neighbors',
    labels={'Class': 'Feature Class', 'Fraction': 'Fraction of k Nearest Neighbors in Same Class'},
    color='Class',  # Optional: Adds color differentiation by class
    template='plotly_white'
)

# Show the interactive plot
fig.show()



k_neighbors = 10

# Store results across all embeddings
all_results = []

# Iterate through each embedding space in ema
for model_id, emb_data in ema.emb.items():
    print(emb_data)
    embeddings = emb_data["emb"]  # Numpy array of embeddings
    feature_classes = ema.meta_data["group"]  # Assuming 'Class' column exists in meta_data
    
    # Create KNN graph for current embeddings
    knn_graph = kneighbors_graph(embeddings, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
    knn_matrix = knn_graph.toarray()
    
    # Calculate the fraction of same-class neighbors
    fractions = []
    for i, neighbors in enumerate(knn_matrix):
        neighbor_indices = np.where(neighbors == 1)[0]
        same_class_count = np.sum(feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i])
        fraction = same_class_count / k_neighbors
        fractions.append(fraction)

    # Collect results for this embedding
    df = pd.DataFrame({
        'Class': feature_classes,
        'Fraction': fractions,
        'Embedding': model_id  # Label the embedding space
    })
    all_results.append(df)

# Combine results across all embeddings
all_results_df = pd.concat(all_results, ignore_index=True)

# Aggregate enrichment per embedding space
summary_df = all_results_df.groupby('Embedding')['Fraction'].agg(['mean', 'median']).reset_index()

# Plot comparison across embedding spaces
fig = px.box(
    all_results_df,
    x='Embedding',
    y='Fraction',
    color='Embedding',
    title='Comparison of Fraction of Same-Class Neighbors Across Embedding Spaces',
    labels={'Embedding': 'Embedding Space', 'Fraction': 'Fraction of k Nearest Neighbors in Same Class'},
    template='plotly_white'
)
fig.show()

# Optional: Plot mean enrichment as a bar chart
fig2 = px.bar(
    summary_df,
    x='Embedding',
    y='mean',
    title='Mean Fraction of Same-Class Neighbors Across Embedding Spaces',
    labels={'Embedding': 'Embedding Space', 'mean': 'Mean Fraction of Same-Class Neighbors'},
    template='plotly_white',
    text='mean'
)
fig2.show()

### making a heatmap (embedding spaces x feature classes) x mean score of kmeans nearest neighbours with same class



heatmap_data = (
    all_results_df
    .groupby(['Class', 'Embedding'])['Fraction']
    .mean()  # Use mean or replace with median()
    .unstack()  # Reshape to have Classes as rows and Embeddings as columns
)

class_counts = all_results_df.groupby('Class').size()  # Count samples per class

heatmap_data.index = [
    f"{feature_class} (n = {count})" 
    for feature_class, count in zip(heatmap_data.index, class_counts[heatmap_data.index])
]

# Generate heatmap with updated y-axis labels and custom font
fig = px.imshow(
    heatmap_data,
    labels=dict(x="Embedding Space", y="Feature Class (Samples)", color="Mean Fraction"),
    title="Mean Fraction of Same-Class Neighbors (with Sample Counts)",
    color_continuous_scale=[
        (0.0, "lightblue"),  # Light blue for the lowest value
        (1.0, "darkblue")    # Dark blue for the highest value
    ]
)

# Update font settings for the heatmap
fig.update_layout(
    font=dict(
        family="Arial",  # Set font to Arial
        color="black"    # Set font color to black
    )
)

# Display heatmap
fig.show()

### heatmap with (n embedding spaces x features) x average score of k nearest neighbours with same class

import pandas as pd
import numpy as np
import plotly.express as px

# Initialize a dictionary to store average scores for each feature and embedding
average_scores = {}

columns = ['group', 'length_bin']

# Loop through features in ema.meta_data
for feature in columns:
    feature_scores = []

    # Loop through embedding spaces
    for embedding_id, embedding_obj in ema.emb.items():
        embeddings = embedding_obj['emb']

        # Compute KNN graph
        knn_graph = kneighbors_graph(embeddings, n_neighbors=k_neighbors, mode='connectivity', include_self=False)

        # Compute fraction of neighbors with the same class label for all points
        class_labels = ema.meta_data[feature].values
        same_class_fraction = [
            np.mean(class_labels[knn_graph.indices[knn_graph.indptr[i]:knn_graph.indptr[i + 1]]] == class_labels[i])
            for i in range(len(class_labels))
        ]

        # Average across all points regardless of their class
        feature_scores.append(np.mean(same_class_fraction))

    # Store scores for the feature
    average_scores[feature] = feature_scores

# Create a DataFrame for the heatmap
heatmap_data = pd.DataFrame(average_scores, index=ema.emb.keys()).T

# Plot the heatmap
fig = px.imshow(
    heatmap_data,
    labels=dict(x="Embedding Space", y="Feature", color="Average Fraction"),
    title="Average Fraction of Same-Class Neighbors per Feature and Embedding Space",
    color_continuous_scale=[
        (0.0, "lightblue"),  # Light blue for low values
        (1.0, "darkblue")    # Dark blue for high values
    ]
)

# Customize font
fig.update_layout(
    font=dict(
        family="Arial",  # Set font to Arial
        color="black"    # Set font color to black
    )
)

# Display the heatmap
fig.show()


import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import plotly.express as px

## k-NN feature enrichment scores

def compare_knn_fraction_across_embeddings(ema, feature_column, k_neighbors=10, title="Comparison of Fraction of Same-Class Neighbors"):
    """
    Compares the fraction of same-class neighbors across different embedding spaces using boxplots.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - feature_column: Column name in ema.meta_data for the feature to analyze.
    - k_neighbors: Number of neighbors to consider in KNN graph.
    - title: Title for the boxplot.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(embeddings, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i])
            fraction = same_class_count / k_neighbors
            fractions.append(fraction)

        # Collect results for this embedding
        df = pd.DataFrame({
            'Class': feature_classes,
            'Fraction': fractions,
            'Embedding': model_id  # Label the embedding space
        })
        all_results.append(df)

    # Combine results across all embeddings
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Plot comparison across embedding spaces
    fig = px.box(
        all_results_df,
        x='Embedding',
        y='Fraction',
        color='Embedding',
        title=title,
        labels={'Embedding': 'Embedding Space', 'Fraction': 'Fraction of k Nearest Neighbors in Same Class'},
        template='plotly_white'
    )
    
    # Display the boxplot
    fig.show()

# Example usage
compare_knn_fraction_across_embeddings(
    ema, 
    feature_column="group", 
    k_neighbors=10, 
    title="Comparison of Fraction of Same-Class Neighbors Across Embedding Spaces"
)


import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import plotly.express as px

def plot_knn_fraction_heatmap(ema, feature_column, k_neighbors=10, title="KNN Fraction Heatmap"):
    """
    Generates a heatmap showing the mean fraction of same-class neighbors across embeddings.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - feature_column: Column name in ema.meta_data for the feature to analyze.
    - k_neighbors: Number of neighbors to consider in KNN graph.
    - title: Title for the heatmap.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(embeddings, n_neighbors=k_neighbors, mode='connectivity', include_self=False)
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i])
            fraction = same_class_count / k_neighbors
            fractions.append(fraction)

        # Collect results for this embedding
        df = pd.DataFrame({
            'Class': feature_classes,
            'Fraction': fractions,
            'Embedding': model_id  # Label the embedding space
        })
        all_results.append(df)

    # Combine results across all embeddings
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Aggregate enrichment per embedding space
    heatmap_data = (
        all_results_df
        .groupby(['Class', 'Embedding'])['Fraction']
        .mean()  # Use mean or replace with median()
        .unstack()  # Reshape to have Classes as rows and Embeddings as columns
    )

    # Count samples per class for labels
    class_counts = all_results_df.groupby('Class').size()

    # Update heatmap index with sample counts
    heatmap_data.index = [
        f"{feature_class} (n = {count})" 
        for feature_class, count in zip(heatmap_data.index, class_counts[heatmap_data.index])
    ]

    # Generate heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Embedding Space", y="Feature Class (Samples)", color="Mean Fraction"),
        title=title,
        color_continuous_scale=[
            (0.0, "lightblue"),  # Light blue for the lowest value
            (1.0, "darkblue")    # Dark blue for the highest value
        ]
    )

    # Add text labels with cell values
    for i, row in enumerate(heatmap_data.index):
        for j, col in enumerate(heatmap_data.columns):
            value = heatmap_data.iloc[i, j]
            fig.add_annotation(
                x=col,
                y=row,
                text=f"{value:.2f}",
                showarrow=False,
                font=dict(color="black", size=10)
            )

    # Update font settings for the heatmap
    fig.update_layout(
        font=dict(
            family="Arial",  # Set font to Arial
            #color="black"    # Set font color to black
        )
    )

    # Display heatmap
    fig.show()

# Example usage
plot_knn_fraction_heatmap(ema, feature_column="group", k_neighbors=10, title="Mean Fraction of Same-Class Neighbors")



# Example usage
plot_knn_fraction_heatmap(ema, feature_column="group", k_neighbors=10, title="Mean Fraction of Same-Class Neighbors")



## NOW UNSUPERVISED APPROACHES

from sklearn.neighbors import NearestNeighbors

def compute_neighborhood_similarity(embeddings1, embeddings2, k):
    # Compute k-NN for embeddings1
    knn1 = NearestNeighbors(n_neighbors=k).fit(embeddings1)
    neighbors1 = knn1.kneighbors(return_distance=False)

    # Compute k-NN for embeddings2
    knn2 = NearestNeighbors(n_neighbors=k).fit(embeddings2)
    neighbors2 = knn2.kneighbors(return_distance=False)

    # Compute overlap for each point
    overlap_fraction = [
        len(set(neighbors1[i]).intersection(set(neighbors2[i]))) / k
        for i in range(len(embeddings1))
    ]
    return np.array(overlap_fraction)



def plot_boxplot_similarity_by_feature_class(embedding1_id, embedding2_id, feature, k):
    embeddings1 = ema.emb[embedding1_id]['emb']
    embeddings2 = ema.emb[embedding2_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Compute neighborhood similarity
    similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        "Feature Class": feature_classes,
        "Neighborhood Similarity": similarity_scores
    })
    
    # Plot boxplot
    fig = px.box(
        plot_data,
        x="Feature Class",
        y="Neighborhood Similarity",
        points="all",
        title=f"Neighborhood Similarity by Feature Class ({embedding1_id} vs {embedding2_id})",
        labels={"Neighborhood Similarity": "Fraction of Overlapping Neighbors"}
    )
    fig.update_layout(
        font=dict(family="Arial", color="black")  # Set font to Arial and black
    )
    fig.show()
    
def compute_similarity_heatmap(feature, k):
    # Prepare DataFrame to store mean similarities
    embedding_ids = list(ema.emb.keys())
    heatmap_data = pd.DataFrame(index=embedding_ids, columns=embedding_ids)
    feature_classes = ema.meta_data[feature].unique()
    
    # Compute similarities for each embedding pair
    for i, emb1 in enumerate(embedding_ids):
        for j, emb2 in enumerate(embedding_ids):
            if i <= j:  # Symmetric computation
                embeddings1 = ema.emb[emb1]['emb']
                embeddings2 = ema.emb[emb2]['emb']
                
                # Compute neighborhood similarity
                similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
                
                # Aggregate mean similarity per feature class
                mean_similarity = []
                for feature_class in feature_classes:
                    class_mask = ema.meta_data[feature] == feature_class
                    mean_similarity.append(similarity_scores[class_mask].mean())
                
                # Compute mean similarity across all feature classes
                heatmap_data.loc[emb1, emb2] = np.mean(mean_similarity)
                heatmap_data.loc[emb2, emb1] = heatmap_data.loc[emb1, emb2]  # Symmetry
    
    # Plot heatmap
    fig = px.imshow(
        heatmap_data.astype(float),
        labels=dict(x="Embedding Space", y="Embedding Space", color="Mean Similarity"),
        title=f"Neighborhood Similarity Heatmap (k = {k})",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    
    
# distribution of similarity between neighbourhoods of embedding spaces

def plot_similarity_distribution(embedding1_id, embedding2_id, k):
    embeddings1 = ema.emb[embedding1_id]['emb']
    embeddings2 = ema.emb[embedding2_id]['emb']
    
    # Compute neighborhood similarity
    similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        "Neighborhood Similarity": similarity_scores
    })
    
    # Plot distribution (KDE)
    fig = px.histogram(
        plot_data,
        x="Neighborhood Similarity",
        nbins=50,  # Number of bins for histogram
        title=f"Distribution of Neighborhood Similarity ({embedding1_id} vs {embedding2_id})",
        labels={"Neighborhood Similarity": "Fraction of Overlapping Neighbors"},
        marginal="box",  # Adds a box plot on top
        histnorm="density"  # Normalize histogram to density
    )
    
    # Update layout
    fig.update_layout(
        font=dict(family="Arial", color="black"),
        coloraxis_colorbar=dict(title="Density")
    )
    fig.show()
    
def plot_feature_distribution_across_similarity_bins_absolute(
    embedding1_id, embedding2_id, feature, k, bins=5
):
    embeddings1 = ema.emb[embedding1_id]['emb']
    embeddings2 = ema.emb[embedding2_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Compute neighborhood similarity
    similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
    
    # Bin the similarity scores
    bin_edges = np.linspace(0, 1, bins + 1)  # Equal-width bins
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)]
    binned_scores = pd.cut(similarity_scores, bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        "Feature Class": feature_classes,
        "Similarity Bin": binned_scores
    })
    
    # Aggregate absolute counts of feature classes in each bin
    distribution = (
        plot_data.groupby(["Similarity Bin", "Feature Class"])
        .size()
        .reset_index(name="Count")
    )
    
    # Plot stacked bar chart
    fig = px.bar(
        distribution,
        x="Similarity Bin",
        y="Count",
        color="Feature Class",
        title=f"Distribution of {feature} Across Similarity Bins ({embedding1_id} vs {embedding2_id})",
        labels={"Count": "Count of Feature Class", "Similarity Bin": "Neighborhood Similarity Bin"},
        text="Count"  # Add raw counts as text
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black"),
        barmode="stack"  # Stacked bar chart
    )
    fig.show()

plot_feature_distribution_across_similarity_bins_absolute(
    "ESM2", "ESM3", "group", k=10, bins=5
)

# this is a great function!! 
def plot_heatmap_feature_distribution(
    embedding1_id, embedding2_id, feature, k, bins=5
):
    embeddings1 = ema.emb[embedding1_id]['emb']
    embeddings2 = ema.emb[embedding2_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Compute neighborhood similarity
    similarity_scores = compute_neighborhood_similarity(embeddings1, embeddings2, k)
    
    # Bin the similarity scores
    bin_edges = np.linspace(0, 1, bins + 1)  # Equal-width bins
    bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)]
    binned_scores = pd.cut(similarity_scores, bins=bin_edges, labels=bin_labels, include_lowest=True)
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        "Feature Class": feature_classes,
        "Similarity Bin": binned_scores
    })
    
    # Aggregate counts of feature classes in each bin
    distribution = (
        plot_data.groupby(["Similarity Bin", "Feature Class"])
        .size()
        .reset_index(name="Count")
    )
    
    # Pivot the data for heatmap
    heatmap_data = distribution.pivot(index="Feature Class", columns="Similarity Bin", values="Count")
    
    # Plot heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Similarity Bin", y="Feature Class", color="Count of Samples"),
        title=f"Feature Class Distribution Across Similarity Bins ({embedding1_id} vs {embedding2_id})",
        color_continuous_scale="Blues",  # Color scale from light blue (low) to dark blue (high)
        text_auto=True  # Display values in each cell
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()

plot_heatmap_feature_distribution(
    "ESM2", "ESM3", "group", k=10, bins=5
)

# mixing between groups 

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

def compute_class_mixing_in_neighborhood(embeddings, feature_classes, k):
    """
    Calculate the overlap of neighbors' feature classes for each point.
    Returns a DataFrame with counts of each class in the k-nearest neighbors for each point.
    """
    # Fit the NearestNeighbors model to the embeddings
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    knn.fit(embeddings)
    
    # Get k-nearest neighbors for each point
    distances, indices = knn.kneighbors(embeddings)
    
    # Get the feature classes for each of the k-nearest neighbors
    neighbor_class_counts = np.zeros((len(embeddings), len(feature_classes.unique())))
    
    for i, idx in enumerate(indices):
        # Exclude the point itself (first in the neighbors list)
        neighbor_classes = feature_classes.iloc[idx[1:]]  # Skip the first neighbor (the point itself)
        
        # Count the occurrences of each feature class in the neighbors
        for class_label in neighbor_classes:
            class_index = np.where(feature_classes.unique() == class_label)[0][0]
            neighbor_class_counts[i, class_index] += 1
    
    return neighbor_class_counts, feature_classes.unique()

def plot_class_mixing_heatmap(embedding_id, feature, k):
    embeddings = ema.emb[embedding_id]['emb']
    feature_classes = ema.meta_data[feature]
    
    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, k)

    # Convert counts to a DataFrame for easier plotting
    mixing_df = pd.DataFrame(mixing_counts, columns=class_labels)
    
    # Transpose to have feature classes of points on y-axis and neighbors' feature classes on x-axis
    mixing_df = mixing_df.T
    
    # Plot heatmap
    fig = px.imshow(
        mixing_df,  # We now transpose so feature classes are on the Y-axis
        labels=dict(x="Feature Class (Neighbor)", y="Feature Class (Sample)", color="Neighbor Count"),
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_id})",
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    
plot_class_mixing_heatmap("ESM3", "group", k=10)

# this is a great graphic!!:

def compute_class_mixing_in_neighborhood(embeddings, feature_classes, k):
    """
    Calculate the overlap of neighbors' feature classes for each point.
    Returns a DataFrame with counts of each class in the k-nearest neighbors for each point.
    """
    # Fit the NearestNeighbors model to the embeddings
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    knn.fit(embeddings)
    
    # Get k-nearest neighbors for each point
    distances, indices = knn.kneighbors(embeddings)
    
    # Initialize a matrix to store the counts of neighbors' feature classes
    neighbor_class_counts = np.zeros((len(feature_classes.unique()), len(feature_classes.unique())))
    
    # Loop over each sample and its neighbors
    for i, idx in enumerate(indices):
        # Exclude the point itself (the first in the neighbors list)
        neighbor_classes = feature_classes.iloc[idx[1:]]  # Skip the first neighbor (the point itself)
        
        # For each of the neighbors, increment the count for their feature class in the matrix
        sample_class = feature_classes.iloc[i]
        for neighbor_class in neighbor_classes:
            sample_class_index = np.where(feature_classes.unique() == sample_class)[0][0]
            neighbor_class_index = np.where(feature_classes.unique() == neighbor_class)[0][0]
            neighbor_class_counts[neighbor_class_index, sample_class_index] += 1
    
    return neighbor_class_counts, feature_classes.unique()

def plot_class_mixing_heatmap(embedding_id, feature, k):
    embeddings = ema.emb[embedding_id]["emb"]
    feature_classes = ema.meta_data[feature]
    
    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, k)

    # Convert counts to a DataFrame for easier plotting
    mixing_df = pd.DataFrame(mixing_counts, index=class_labels, columns=class_labels)
    
    # Plot heatmap
    fig = px.imshow(
        mixing_df,  # Feature classes on both axes
        labels=dict(x="Feature Class (Sample)", y="Feature Class (Neighbor)", color="Neighbor Count"),
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_id})",
        color_continuous_scale="Blues",
        text_auto=True
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    
plot_class_mixing_heatmap("ESM3", "group", k=10)

def compute_class_mixing_in_neighborhood(embeddings, feature_classes, k):
    """
    Calculate the overlap of neighbors' feature classes for each point.
    Returns a DataFrame with counts of each class in the k-nearest neighbors for each point.
    """
    # Fit the NearestNeighbors model to the embeddings
    knn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
    knn.fit(embeddings)
    
    # Get k-nearest neighbors for each point
    distances, indices = knn.kneighbors(embeddings)
    
    # Initialize a matrix to store the counts of neighbors' feature classes
    neighbor_class_counts = np.zeros((len(feature_classes.unique()), len(feature_classes.unique())))
    
    # Loop over each sample and its neighbors
    for i, idx in enumerate(indices):
        # Exclude the point itself (the first in the neighbors list)
        neighbor_classes = feature_classes.iloc[idx[1:]]  # Skip the first neighbor (the point itself)
        
        # For each of the neighbors, increment the count for their feature class in the matrix
        sample_class = feature_classes.iloc[i]
        for neighbor_class in neighbor_classes:
            sample_class_index = np.where(feature_classes.unique() == sample_class)[0][0]
            neighbor_class_index = np.where(feature_classes.unique() == neighbor_class)[0][0]
            neighbor_class_counts[neighbor_class_index, sample_class_index] += 1
    
    return neighbor_class_counts, feature_classes.unique()

def plot_class_mixing_heatmap_diff(embedding_id_1, embedding_id_2, feature, k):
    embeddings_1 = ema.emb[embedding_id_1]["emb"]
    embeddings_2 = ema.emb[embedding_id_2]["emb"]
    feature_classes = ema.meta_data[feature]
    
    # Get class mixing counts for both embeddings
    mixing_counts_1, class_labels = compute_class_mixing_in_neighborhood(embeddings_1, feature_classes, k)
    mixing_counts_2, _ = compute_class_mixing_in_neighborhood(embeddings_2, feature_classes, k)

    # Compute the difference between the two matrices
    diff_counts = mixing_counts_2 - mixing_counts_1
    
    # Convert the difference to a DataFrame for easier plotting
    diff_df = pd.DataFrame(diff_counts, index=class_labels, columns=class_labels)
    
    # Plot the difference heatmap
    fig = px.imshow(
        diff_df,  # Difference matrix
        labels=dict(x="Feature Class (Sample)", y="Feature Class (Neighbor)", color="Difference in Neighbor Count"),
        title=f"Difference in Class Mixing Between Embedding Spaces ({embedding_id_1} vs {embedding_id_2})",
        color_continuous_scale="RdBu",  # Red to Blue to highlight differences
        text_auto=True
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    
plot_class_mixing_heatmap_diff("ESM2", "ESM3", "group", k=10)


import plotly.express as px
import numpy as np

# Convert the adjacency matrix to dense format if it's sparse
adj_matrix = knn1.toarray()

# Plot heatmap
fig_heatmap = px.imshow(adj_matrix, 
                        color_continuous_scale='Viridis',
                        labels=dict(x="Node Index", y="Node Index", color="Distance"),
                        title="Adjacency Matrix Heatmap")
fig_heatmap.show()

# Group labels
group_labels = ema.meta_data['group']

# Get unique group labels and assign discrete colors
unique_groups = sorted(set(group_labels))
color_map = {group: px.colors.qualitative.Plotly[i % 10] for i, group in enumerate(unique_groups)}

# Map each group to its assigned color
group_colors = [color_map[group] for group in group_labels]

import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Reduce embedding space to 2D for visualization
pca = PCA(n_components=2)
node_positions = pca.fit_transform(ema.emb['ESM3']['emb'])

# Extract edges from adjacency matrix
edges = np.array(np.nonzero(adj_matrix)).T  # Get non-zero indices as edges

# Create edge traces
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = node_positions[edge[0]]
    x1, y1 = node_positions[edge[1]]
    edge_x += [x0, x1, None]  # None separates individual lines
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=0.5, color='gray'),
                        hoverinfo='none')

# Create node traces
node_trace = go.Scatter(
    x=node_positions[:, 0], 
    y=node_positions[:, 1],
    mode='markers',
    marker=dict(
        size=10, 
        color=group_colors,  # Use group labels for colors
        colorscale='Viridis',  # Choose a colorscale (adjust as needed)
        colorbar=dict(title="Group")  # Add a colorbar for reference
    ),
    text=[f"Node {i}, Group: {group_labels[i]}" for i in range(len(node_positions))],
    hoverinfo='text',
    showlegend=False
)

# Add a legend for groups
legend_traces = []
for group, color in color_map.items():
    legend_traces.append(
        go.Scatter(
            x=[None], y=[None],  # Dummy points for legend
            mode='markers',
            marker=dict(size=10, color=color),
            name=group  # Group name in legend
        )
    )
# Combine all traces
fig_graph = go.Figure(
    data=[edge_trace, node_trace] + legend_traces,
    layout=go.Layout(
        title='Graph Visualization with Categorical Legend',
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        hovermode='closest'
    )
)

fig_graph.show()



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