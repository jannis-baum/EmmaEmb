
import pandas as pd
import numpy as np

from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors
import plotly.express as px


## k-NN feature enrichment scores

def compare_knn_fraction_across_embeddings(
    ema, 
    feature_column, 
    k_neighbors=10, 
    distance_metric='euclidean',
    embedding_order=None, 
    color='#303496',
):
    """
    Compares the fraction of same-class neighbors across \
        different embedding spaces using boxplots.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - feature_column: Column name in ema.meta_data for the \
        feature to analyze.
    - k_neighbors: Number of neighbors to consider in KNN graph.
    - distance_metric: Distance metric to use for KNN graph \
        (e.g., 'euclidean', 'cosine').
    - embedding_order: Optional list specifying the order of embedding spaces on the x-axis.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(
            embeddings, 
            n_neighbors=k_neighbors, 
            mode='connectivity', 
            metric=distance_metric,
            include_self=False
        )
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(
                feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i]
            )
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
        title=f"KNN feature enrichment scores for {feature_column}<br>k = {k_neighbors}, {distance_metric}",
        labels={'Embedding': 'Embedding Space', 'Fraction': 'KNN feature enrichment scores'},
        template='plotly_white',
        color_discrete_sequence=[color]
    )

    # Apply embedding order if provided
    if embedding_order:
        fig.update_xaxes(categoryorder='array', categoryarray=embedding_order)
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    
    # Display the boxplot
    fig.show()
    return fig

def compare_knn_statistics_across_embeddings(
    ema, 
    feature_column, 
    k_neighbors=10
):
    """
    Compares the mean and std of same-class neighbors across \
        different embedding spaces using a bar plot.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - feature_column: Column name in ema.meta_data for the \
        feature to analyze.
    - k_neighbors: Number of neighbors to consider in KNN graph.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(embeddings, 
                                     n_neighbors=k_neighbors, 
                                     mode='connectivity', 
                                     include_self=False)
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(
                feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i]
            )
            fraction = same_class_count / k_neighbors
            fractions.append(fraction)

        # Calculate mean and std for this embedding
        mean_fraction = np.mean(fractions)
        std_fraction = np.std(fractions)

        # Store the results
        all_results.append({
            'Embedding': model_id,
            'Mean Fraction': mean_fraction,
            'Std Fraction': std_fraction
        })

    # Convert results into a DataFrame
    results_df = pd.DataFrame(all_results)

    # Plot mean and std across embedding spaces using a bar plot
    fig = px.bar(
        results_df,
        x='Embedding',
        y=['Mean Fraction', 'Std Fraction'],
        title=f"Mean and Std of Same-Class Neighbors for {feature_column} with k = {k_neighbors}",
        labels={'Embedding': 'Embedding Space', 'value': 'Fraction of k Nearest Neighbors in Same Class'},
        barmode='group',
        template='plotly_white'
    )

    # Update layout for better appearance
    fig.update_layout(
        font=dict(family="Arial", color="black"),
        yaxis_title='Fraction',
        xaxis_title='Embedding Space'
    )

    # Display the plot
    fig.show()
    
    return fig

def plot_knn_fraction_heatmap(
    ema, 
    feature_column, 
    k_neighbors=10,
    distance_metric='euclidean',
    embedding_order=None,
    color_value='darkblue'
):
    """
    Generates a heatmap showing the mean fraction \
        of same-class neighbors across embeddings.

    Parameters:
    - ema: Handler object containing embeddings \
        and metadata.
    - feature_column: Column name in ema.meta_data \
        for the feature to analyze.
    - k_neighbors: Number of neighbors to consider \
        in KNN graph.
    - embedding_order: Optional list specifying the \
        order of embedding spaces on the x-axis.
    - distance_metric: Distance metric to use for KNN graph \
        (e.g., 'euclidean', 'cosine').
    - color_value: Hex code for the color to use at the maximum of the heatmap.
    """
    # Store results across all embeddings
    all_results = []

    # Iterate through each embedding space in ema
    for model_id, emb_data in ema.emb.items():
        embeddings = emb_data["emb"]  # Numpy array of embeddings
        feature_classes = ema.meta_data[feature_column]  # Extract feature classes

        # Create KNN graph for current embeddings
        knn_graph = kneighbors_graph(
            embeddings, 
            n_neighbors=k_neighbors, 
            mode='connectivity', 
            metric=distance_metric,
            include_self=False
        )
        knn_matrix = knn_graph.toarray()

        # Calculate the fraction of same-class neighbors
        fractions = []
        for i, neighbors in enumerate(knn_matrix):
            neighbor_indices = np.where(neighbors == 1)[0]
            same_class_count = np.sum(
                feature_classes.iloc[neighbor_indices].values == feature_classes.iloc[i]
            )
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

    # Reorder columns based on the custom order, if provided
    if embedding_order:
        heatmap_data = heatmap_data.reindex(columns=embedding_order)
        
    # Count samples per class for labels
    class_counts = all_results_df.groupby('Class').size()

    # Update heatmap index with sample counts
    heatmap_data.index = [
        f"{feature_class} (n = {int(count / len(ema.emb.items()))})" 
        for feature_class, count in zip(heatmap_data.index, class_counts[heatmap_data.index])
    ]

    # Generate heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(
            x="Embedding Space", 
            y="Feature Class (Samples)", 
            color="Mean Fraction"
            ),
        title=f"KNN Fraction Heatmap for {feature_column}<br>with k = {k_neighbors} ({distance_metric} metric)",
        color_continuous_scale=[
            (0.0, "lightblue"),  # White for the lowest value
            (1.0, "darkblue")    # Custom color for the highest value
        ],
        text_auto=".2f",
        aspect='auto'
    )

    # Update font settings for the heatmap
    fig.update_layout(
        font=dict(
            family="Arial",
        )
    )

    # Display heatmap
    fig.show()
    return fig
    
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from collections import Counter

def compute_class_mixing_in_neighborhood(embeddings, feature_classes, k_neighbors, distance_metric):
    """
    Calculate the overlap of neighbors' feature classes for each point.
    Returns a matrix with counts of each class in the k-nearest neighbors for each point.
    """
    # Encode feature classes to numerical labels
    le = LabelEncoder()
    encoded_classes = le.fit_transform(feature_classes)
    unique_classes = le.classes_
    num_classes = len(unique_classes)
    
    # Fit the NearestNeighbors model to the embeddings
    knn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric=distance_metric)  # +1 to include the point itself
    knn.fit(embeddings)
    
    # Get k-nearest neighbors for each point
    _, indices = knn.kneighbors(embeddings)
    
    # Initialize a matrix to store the counts of neighbors' feature classes
    neighbor_class_counts = np.zeros((num_classes, num_classes), dtype=int)
    
    # Loop over each sample and its neighbors
    for i, neighbors in enumerate(indices):
        sample_class_idx = encoded_classes[i]
        neighbor_class_indices = encoded_classes[neighbors[1:]]  # Exclude the point itself
        
        # Count occurrences of each class in the neighbors
        class_counts = Counter(neighbor_class_indices)
        
        # Update the neighbor class counts matrix
        for neighbor_class_idx, count in class_counts.items():
            neighbor_class_counts[neighbor_class_idx, sample_class_idx] += count
    
    return neighbor_class_counts, unique_classes

# cross-class neighbourhodd analysis
# def compute_class_mixing_in_neighborhood(embeddings, feature_classes, 
#                                          k_neighbors, distance_metric):
#     """
#     Calculate the overlap of neighbors' feature classes for each point.
#     Returns a DataFrame with counts of each class in the k-nearest neighbors for each point.
#     """
#     # Fit the NearestNeighbors model to the embeddings
#     knn = NearestNeighbors(n_neighbors=k_neighbors+1, metric=distance_metric)  # +1 because the point itself is included
#     knn.fit(embeddings)
    
#     # Get k-nearest neighbors for each point
#     distances, indices = knn.kneighbors(embeddings)
    
#     # Initialize a matrix to store the counts of neighbors' feature classes
#     neighbor_class_counts = np.zeros((len(feature_classes.unique()), len(feature_classes.unique())))
    
#     # Loop over each sample and its neighbors
#     for i, idx in enumerate(indices):
#         # Exclude the point itself (the first in the neighbors list)
#         neighbor_classes = feature_classes.iloc[idx[1:]]  # Skip the first neighbor (the point itself)
        
#         # For each of the neighbors, increment the count for their feature class in the matrix
#         sample_class = feature_classes.iloc[i]
#         for neighbor_class in neighbor_classes:
#             sample_class_index = np.where(feature_classes.unique() == sample_class)[0][0]
#             neighbor_class_index = np.where(feature_classes.unique() == neighbor_class)[0][0]
#             neighbor_class_counts[neighbor_class_index, sample_class_index] += 1
    
#     return neighbor_class_counts, feature_classes.unique()

def plot_class_mixing_heatmap(ema, embedding_space, feature, 
                              k_neighbors, 
                              distance_metric,
                              color = ""):
    embeddings = ema.emb[embedding_space]["emb"]
    feature_classes = ema.meta_data[feature]
    
    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, 
                                                                       k_neighbors,
                                                                       distance_metric)

    # Convert counts to a DataFrame for easier plotting
    mixing_df = pd.DataFrame(mixing_counts, index=class_labels, columns=class_labels)
    
    # Plot heatmap
    fig = px.imshow(
        mixing_df,  # Feature classes on both axes
        labels=dict(x="Feature Class (Sample)", y="Feature Class (Neighbor)", color="Neighbor Count"),
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_space})",
        color_continuous_scale="Blues",
        text_auto=True,
        aspect='auto'
    )
    
    fig.update_layout(
        font=dict(family="Arial", color="black")
    )
    fig.show()
    return fig

# cross-space neighbourhood similarity

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


def plot_heatmap_feature_distribution(ema,
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



def plot_class_mixing_stacked_bar(ema, embedding_space, feature, k):
    """
    Plot a stacked bar chart showing the proportions of neighbor classes for each class.
    """
    embeddings = ema.emb[embedding_space]["emb"]
    feature_classes = ema.meta_data[feature]

    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, k)

    # Convert counts to proportions
    mixing_proportions = (mixing_counts.T / mixing_counts.sum(axis=1)).T  # Normalize rows
    mixing_df = pd.DataFrame(mixing_proportions, index=class_labels, columns=class_labels)

    # Melt DataFrame for stacked bar plot
    melted_df = mixing_df.reset_index().melt(
        id_vars="index", var_name="Neighbor Class", value_name="Proportion"
    )
    melted_df.rename(columns={"index": "Sample Class"}, inplace=True)

    # Plot stacked bar chart
    fig = px.bar(
        melted_df,
        x="Sample Class",
        y="Proportion",
        color="Neighbor Class",
        barmode="stack",
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_space})",
        labels={"Proportion": "Proportion of Neighbors"},
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    fig.update_layout(
        font=dict(family="Arial", color="black"),
        xaxis_title="Feature Class (Sample)",
        yaxis_title="Proportion of Neighbors",
    )

    fig.show()

def plot_class_mixing_stacked_bar_absolute(ema, embedding_space, feature, k):
    """
    Plot a stacked bar chart showing the absolute counts of neighbor classes for each sample class.
    """
    embeddings = ema.emb[embedding_space]["emb"]
    feature_classes = ema.meta_data[feature]

    # Get class mixing counts
    mixing_counts, class_labels = compute_class_mixing_in_neighborhood(embeddings, feature_classes, k)

    # Convert counts to a DataFrame
    mixing_df = pd.DataFrame(mixing_counts, index=class_labels, columns=class_labels)

    # Melt DataFrame for stacked bar plot
    melted_df = mixing_df.reset_index().melt(
        id_vars="index", var_name="Neighbor Class", value_name="Count"
    )
    melted_df.rename(columns={"index": "Sample Class"}, inplace=True)

    # Plot stacked bar chart
    fig = px.bar(
        melted_df,
        x="Sample Class",
        y="Count",
        color="Neighbor Class",
        barmode="stack",
        title=f"Class Mixing in Neighborhoods (Embedding: {embedding_space})",
        labels={"Count": "Neighbor Count"},
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    # Add total count annotations on top of each bar
    total_counts = mixing_df.sum(axis=1)
    for i, total in enumerate(total_counts):
        fig.add_annotation(
            x=class_labels[i],
            y=total,
            text=str(int(total)),
            showarrow=False,
            font=dict(size=10, color="black", family="Arial"),
            xanchor="center",
            yanchor="bottom",
        )

    fig.update_layout(
        font=dict(family="Arial", color="black"),
        xaxis_title="Feature Class (Sample)",
        yaxis_title="Neighbor Count",
    )

    fig.show()
    
def analyze_low_similarity_distributions(
    ema,
    embedding_space_1,
    embedding_space_2,
    features,
    similarity_threshold=0.2,
    k_neighbors=10
):
    """
    Analyze the distribution of feature classes for samples with low neighborhood similarity.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - embedding_space_1: Name of the first embedding space.
    - embedding_space_2: Name of the second embedding space.
    - features: List of feature column names from ema.meta_data to analyze.
    - similarity_threshold: Threshold below which similarity is considered "low" (default: 0.2).
    - k_neighbors: Number of neighbors to consider in KNN graph (default: 10).
    """
    def compute_neighborhood_similarity(emb1, emb2, k):
        """
        Helper function to compute neighborhood similarity between two embeddings.
        """
        knn_1 = kneighbors_graph(emb1, n_neighbors=k, mode="connectivity").toarray()
        knn_2 = kneighbors_graph(emb2, n_neighbors=k, mode="connectivity").toarray()
        similarity = np.sum(knn_1 * knn_2, axis=1) / k  # Fraction of shared neighbors
        return similarity

    # Retrieve embeddings
    emb1 = ema.emb[embedding_space_1]["emb"]
    emb2 = ema.emb[embedding_space_2]["emb"]

    # Compute neighborhood similarity
    similarities = compute_neighborhood_similarity(emb1, emb2, k_neighbors)

    # Identify samples with low similarity
    low_similarity_indices = np.where(similarities < similarity_threshold)[0]
    low_similarity_samples = ema.meta_data.iloc[low_similarity_indices]

    # Analyze feature distributions for low-similarity samples
    feature_distributions = {}
    for feature in features:
        feature_classes = low_similarity_samples[feature]
        distribution = feature_classes.value_counts(normalize=True)  # Normalize for proportion
        feature_distributions[feature] = distribution

    # Plot distributions
    for feature, distribution in feature_distributions.items():
        fig = px.bar(
            distribution,
            x=distribution.index,
            y=distribution.values,
            title=f"Distribution of {feature} Classes (Similarity < {similarity_threshold})",
            labels={"x": feature, "y": "Proportion of Samples"},
            template="plotly_white"
        )
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')  # Add text labels
        fig.show()

    return fig

def analyze_low_similarity_distribution(
    ema,
    embedding_space_1,
    embedding_space_2,
    feature,
    similarity_threshold=0.2,
    k_neighbors=10,
    distance_metric="euclidean"
):
    """
    Analyze and visualize the distribution of a single feature for samples with low neighborhood similarity.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - embedding_space_1: Name of the first embedding space.
    - embedding_space_2: Name of the second embedding space.
    - feature: Feature column name from ema.meta_data to analyze.
    - similarity_threshold: Threshold below which similarity is considered "low" (default: 0.2).
    - k_neighbors: Number of neighbors to consider in KNN graph (default: 10).
    - distance_metric: Metric to use for computing KNN graph (default: "euclidean").
      Examples include "euclidean", "cosine", "manhattan", etc.
    """
    def compute_neighborhood_similarity(emb1, emb2, k, metric):
        """
        Helper function to compute neighborhood similarity between two embeddings.
        """
        knn_1 = kneighbors_graph(emb1, n_neighbors=k, mode="connectivity", metric=metric).toarray()
        knn_2 = kneighbors_graph(emb2, n_neighbors=k, mode="connectivity", metric=metric).toarray()
        similarity = np.sum(knn_1 * knn_2, axis=1) / k  # Fraction of shared neighbors
        return similarity

    # Retrieve embeddings
    emb1 = ema.emb[embedding_space_1]["emb"]
    emb2 = ema.emb[embedding_space_2]["emb"]

    # Compute neighborhood similarity
    similarities = compute_neighborhood_similarity(emb1, emb2, k_neighbors, distance_metric)

    # Identify samples with low similarity
    low_similarity_indices = np.where(similarities < similarity_threshold)[0]
    low_similarity_samples = ema.meta_data.iloc[low_similarity_indices]

    # Compute distributions
    total_distribution = ema.meta_data[feature].value_counts(normalize=True)  # Entire dataset
    low_similarity_distribution = low_similarity_samples[feature].value_counts(normalize=True)  # Low-similarity subset

    # Align indices to ensure consistent comparison
    aligned_distributions = total_distribution.align(low_similarity_distribution, fill_value=0)

    # Prepare scatter plot data
    fractions_in_dataset = aligned_distributions[0]  # Fraction in entire dataset
    fractions_in_subset = aligned_distributions[1]   # Fraction in low-similarity subset
    class_labels = aligned_distributions[0].index  # Class labels for legend

    # Create scatter plot
    fig = px.scatter(
        x=fractions_in_dataset,
        y=fractions_in_subset,
        color=class_labels,
        labels={"x": "Fraction in Dataset", "y": "Fraction in Subsample", "color": feature},
        title=f"Comparison of {feature} Fractions (Similarity < {similarity_threshold} between {embedding_space_1} and {embedding_space_2}, Metric: {distance_metric})",
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=10, line=None))  # Remove rim around dots
    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=1, y1=1,
        line=dict(color="LightGrey", dash="dash")
    )  # Add diagonal reference line
    
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", size=12, color="black"),
    )
    # show line at y=0 and x=0
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2)
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2)
    # hide gridlines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Update layout for font and legend
    fig.update_layout(
        font=dict(family="Arial", color="black"),
        # legend=dict(orientation="h", yanchor="bottom", y=-0.2),  # Move legend below plot
        showlegend=True
    )

    fig.show()

    return fractions_in_dataset, fractions_in_subset






def plot_low_similarity_class_representation(
    ema, 
    embedding_space_1, 
    embedding_space_2, 
    feature_columns, 
    similarity_threshold=0.2, 
    k_neighbors=10, 
    title_prefix="Class Representation in Low-Similarity Samples"
):
    """
    Plots class representation in low-similarity samples between two embedding spaces 
    across multiple features.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - embedding_space_1: Key of the first embedding space in `ema.emb`.
    - embedding_space_2: Key of the second embedding space in `ema.emb`.
    - feature_columns: List of feature columns in `ema.meta_data` to analyze.
    - similarity_threshold: Threshold for identifying low-similarity samples.
    - k_neighbors: Number of neighbors to consider in the neighborhood comparison.
    - title_prefix: Prefix for the scatter plot titles.
    """
    # Compute neighborhood similarity between the two embedding spaces
    sim_matrix = compute_neighborhood_similarity(
        ema.emb[embedding_space_1]["emb"],
        ema.emb[embedding_space_2]["emb"],
        k=k_neighbors
    )
    # Identify low-similarity samples
    low_similarity_indices = np.where(sim_matrix < similarity_threshold)[0]

    # Loop over each feature column to generate plots
    for feature_column in feature_columns:
        # Extract feature classes
        feature_classes = ema.meta_data[feature_column]
        
        # Overall and low-similarity proportions
        total_counts = feature_classes.value_counts(normalize=True)  # Proportion in dataset
        low_similarity_classes = feature_classes.iloc[low_similarity_indices]
        low_similarity_counts = low_similarity_classes.value_counts(normalize=True)  # Proportion in low-similarity

        # Combine results into a DataFrame
        representation_df = pd.DataFrame({
            "Overall Proportion": total_counts,
            "Low-Similarity Proportion": low_similarity_counts,
        }).fillna(0)  # Fill missing values for classes not in low-similarity samples
        representation_df["Total Count"] = feature_classes.value_counts()

        # Plot scatter for the current feature
        fig = px.scatter(
            representation_df,
            x="Overall Proportion",
            y="Low-Similarity Proportion",
            size="Total Count",
            hover_name=representation_df.index,
            labels={
                "Overall Proportion": "Proportion in Dataset",
                "Low-Similarity Proportion": f"Proportion in Low-Similarity (Sim < {similarity_threshold})"
            },
            title=f"{title_prefix}: {feature_column}",
            template="plotly_white",
        )

        # Add diagonal reference line
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color="LightGrey", dash="dash"),
        )

        # Display scatter plot
        fig.show()


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_low_similarity_class_proportions(
    ema,
    embedding_space_1,
    embedding_space_2,
    feature_columns,
    similarity_threshold=0.2,
    k_neighbors=10,
    title="Class Proportions in Low-Similarity Samples"
):
    """
    Plots proportions of class distributions for low-similarity samples across multiple features.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - embedding_space_1: Key of the first embedding space in `ema.emb`.
    - embedding_space_2: Key of the second embedding space in `ema.emb`.
    - feature_columns: List of feature columns in `ema.meta_data` to analyze.
    - similarity_threshold: Threshold for identifying low-similarity samples.
    - k_neighbors: Number of neighbors to consider in the neighborhood comparison.
    - title: Title for the figure.
    """
    # Compute neighborhood similarity between the two embedding spaces
    sim_matrix = compute_neighborhood_similarity(
        ema.emb[embedding_space_1]["emb"],
        ema.emb[embedding_space_2]["emb"],
        k=k_neighbors
    )
    # Identify low-similarity samples
    low_similarity_indices = np.where(sim_matrix < similarity_threshold)[0]

    # Determine subplot layout (2 columns)
    n_features = len(feature_columns)
    n_rows = int(np.ceil(n_features / 2))

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=feature_columns,
        horizontal_spacing=0.15,
        vertical_spacing=0.2,
    )

    # Loop over each feature column
    for idx, feature_column in enumerate(feature_columns):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        # Extract feature classes
        feature_classes = ema.meta_data[feature_column]
        
        # Compute proportions for dataset and low-similarity samples
        total_counts = feature_classes.value_counts(normalize=True)  # Proportion in the dataset
        low_similarity_classes = feature_classes.iloc[low_similarity_indices]
        low_similarity_proportions = low_similarity_classes.value_counts(normalize=True)

        # Create DataFrame for plotting
        proportions_df = pd.DataFrame({
            "Dataset Proportion": total_counts,
            "Low-Similarity Proportion": low_similarity_proportions,
        }).fillna(0)  # Fill missing values for classes not in low-similarity samples

        # Add scatter plot to the current subplot
        scatter = go.Scatter(
            x=proportions_df["Dataset Proportion"],
            y=proportions_df["Low-Similarity Proportion"],
            mode="markers",
            marker=dict(
                size=12,
                color=px.colors.qualitative.Set1[:len(proportions_df)],  # Distinct colors for classes
                showscale=False,
            ),
            text=proportions_df.index,  # Class names as hover text
            name=f"{feature_column}",
        )
        fig.add_trace(scatter, row=row, col=col)

        # Add diagonal reference line
        diagonal_line = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
        fig.add_trace(diagonal_line, row=row, col=col)

        # Update axes
        fig.update_xaxes(title_text="Dataset Proportion", row=row, col=col)
        fig.update_yaxes(title_text="Low-Similarity Proportion", row=row, col=col)

    # Update layout with legends and font
    fig.update_layout(
        title=title,
        height=300 * n_rows,  # Adjust height based on number of subplots
        template="plotly_white",
        font=dict(
            family="Arial",  # Set font to Arial
            color="black"    # Set font color to black
        ),
        legend=dict(
            title="Feature Classes",
            font=dict(family="Arial", color="black"),
        )
    )

    # Show figure
    fig.show()
    return fig

def plot_high_similarity_class_proportions(
    ema,
    embedding_space_1,
    embedding_space_2,
    feature_columns,
    similarity_threshold=0.8,
    k_neighbors=10,
    title="Class Proportions in High-Similarity Samples"
):
    """
    Plots proportions of class distributions for high-similarity samples across multiple features.

    Parameters:
    - ema: Handler object containing embeddings and metadata.
    - embedding_space_1: Key of the first embedding space in `ema.emb`.
    - embedding_space_2: Key of the second embedding space in `ema.emb`.
    - feature_columns: List of feature columns in `ema.meta_data` to analyze.
    - similarity_threshold: Threshold for identifying high-similarity samples.
    - k_neighbors: Number of neighbors to consider in the neighborhood comparison.
    - title: Title for the figure.
    """
    # Compute neighborhood similarity between the two embedding spaces
    sim_matrix = compute_neighborhood_similarity(
        ema.emb[embedding_space_1]["emb"],
        ema.emb[embedding_space_2]["emb"],
        k=k_neighbors
    )
    # Identify high-similarity samples (above threshold)
    high_similarity_indices = np.where(sim_matrix >= similarity_threshold)[0]

    # Determine subplot layout (2 columns)
    n_features = len(feature_columns)
    n_rows = int(np.ceil(n_features / 2))

    # Create subplots
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=feature_columns,
        horizontal_spacing=0.15,
        vertical_spacing=0.2,
    )

    # Loop over each feature column
    for idx, feature_column in enumerate(feature_columns):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        # Extract feature classes
        feature_classes = ema.meta_data[feature_column]
        
        # Compute proportions for dataset and high-similarity samples
        total_counts = feature_classes.value_counts(normalize=True)  # Proportion in the dataset
        high_similarity_classes = feature_classes.iloc[high_similarity_indices]
        high_similarity_proportions = high_similarity_classes.value_counts(normalize=True)

        # Create DataFrame for plotting
        proportions_df = pd.DataFrame({
            "Dataset Proportion": total_counts,
            "High-Similarity Proportion": high_similarity_proportions,
        }).fillna(0)  # Fill missing values for classes not in high-similarity samples

        # Add scatter plot to the current subplot
        scatter = go.Scatter(
            x=proportions_df["Dataset Proportion"],
            y=proportions_df["High-Similarity Proportion"],
            mode="markers",
            marker=dict(
                size=12,
                color=px.colors.qualitative.Set1[:len(proportions_df)],  # Distinct colors for classes
            ),
            text=proportions_df.index,  # Class names as hover text
            name=f"{feature_column}",
        )
        fig.add_trace(scatter, row=row, col=col)

        # Add diagonal reference line
        diagonal_line = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            showlegend=False,
        )
        fig.add_trace(diagonal_line, row=row, col=col)

        # Update axes
        fig.update_xaxes(title_text="Dataset Proportion", row=row, col=col)
        fig.update_yaxes(title_text="High-Similarity Proportion", row=row, col=col)

    # Update layout with legends and font
    fig.update_layout(
        title=title,
        height=300 * n_rows,  # Adjust height based on number of subplots
        template="plotly_white",
        font=dict(
            family="Arial",  # Set font to Arial
            color="black"    # Set font color to black
        )
    )

    # Show figure
    fig.show()
    