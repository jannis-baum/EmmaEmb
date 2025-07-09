import pandas as pd
import numpy as np

from collections import Counter
from sklearn.preprocessing import LabelEncoder

from emmaemb.core import Emma


# get knn alignment scores
def get_knn_alignment_scores(
    emma: Emma,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Function to calculate the alignment scores of k-nearest neighbors \
        across different embedding spaces.

    Args:
        emma (Emma): Emma object
        feature (str): Column name in the metadata DataFrame of \
            the Emma object.
        k (int, optional): Number of nearest neighbors to consider. \
            Defaults to 10.
        metric (str, optional): Distance metric to use. \
            Defaults to "euclidean".

    Returns:
        pd.DataFrame: DataFrame containing the alignment scores of \
            k-nearest neighbors across different embedding spaces.\
            Columns: Sample, Class (feature class name), \
                Fraction (KNN feature alignment score), \
                    Embedding (embedding space name)
    """

    # validate input
    embedding_spaces = emma.emb.keys()
    if embedding_spaces is None:
        raise ValueError("No embeddings found in Emma object")
    emma._check_column_is_categorical(feature)

    all_results = []
    feature_classes = emma.metadata[feature]

    for emb_space in embedding_spaces:
        rank_matrix = emma.get_knn(emb_space, k, metric)

        fractions = []
        for i in range(len(rank_matrix)):
            # Get the indices of the k-nearest neighbors (ranked by distance)
            neighbor_indices = rank_matrix[i]

            # Count how many of the k-nearest neighbors belong to
            # the same class
            same_class_count = np.sum(
                feature_classes.iloc[neighbor_indices].values
                == feature_classes.iloc[i]
            )
            fraction = same_class_count / k
            fractions.append(fraction)

        # Prepare results in a DataFrame for the current embedding space
        df = pd.DataFrame(
            {
                # "Sample": emma.sample_names,
                "Class": feature_classes,
                "Fraction": fractions,
                "Embedding": emb_space,
            }
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


def get_class_mixing_in_neighborhood(
    emma: Emma,
    emb_space: str,
    feature: str,
    k: int = 10,
    metric: str = "euclidean",
):
    # validate input
    emma._check_for_emb_space(emb_space)
    emma._check_column_is_categorical(feature)

    le = LabelEncoder()
    encoded_classes = le.fit_transform(emma.metadata[feature])
    unique_classes = le.classes_
    num_classes = len(unique_classes)

    neighbor_class_counts = np.zeros((num_classes, num_classes), dtype=int)
    neighboring_indices = emma.get_knn(emb_space, k, metric)

    for i, neighbors in enumerate(neighboring_indices):
        sample_class_idx = encoded_classes[i]
        neighbor_class_indices = encoded_classes[neighbors]

        class_counts = Counter(neighbor_class_indices)

        for neighbor_class_idx, count in class_counts.items():
            neighbor_class_counts[
                neighbor_class_idx, sample_class_idx
            ] += count

    return neighbor_class_counts, unique_classes


def get_neighbourhood_similarity(
    emma: Emma,
    emb_space_1: str,
    emb_space_2: str,
    k: int = 10,
    metric: str = "euclidean",
):
    knn_1 = emma.get_knn(emb_space_1, k, metric)
    knn_2 = emma.get_knn(emb_space_2, k, metric)

    similarity = np.zeros(len(knn_1))

    for i, (neighbors_1, neighbors_2) in enumerate(zip(knn_1, knn_2)):
        similarity[i] = len(set(neighbors_1).intersection(neighbors_2)) / k

    return similarity
