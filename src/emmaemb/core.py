import pandas as pd
import plotly.express as px
import numpy as np
import os

from scipy.spatial.distance import pdist, squareform

from src.emmaemb.config import EMB_SPACE_COLORS, DISTANCE_METRIC_ALIASES


class Emma:
    def __init__(self, feature_data: pd.DataFrame):

        # Metadata
        self.metadata = feature_data
        self.metadata_numeric_columns = self._get_numeric_columns()
        self.metadata_categorical_columns = self._get_categorical_columns()
        self.sample_names = self.metadata.iloc[:, 0].tolist()
        self.color_map = self._get_color_map_for_features()

        # Embedding spaces
        self.emb = dict()

        print(f"{len(self.sample_names)} samples loaded.")
        print(f"Categories in meta data: {self.metadata_categorical_columns}")
        print(
            f"Numerical columns in meta data: {self.metadata_numeric_columns}"
        )

    # Metadata

    def _get_numeric_columns(self) -> list:
        """Identify numeric columns in the metadata.

        Returns:
        list: List of column names that are numeric.
        """
        numerical_columns = (
            self.metadata.iloc[:, 1:]
            .select_dtypes(include=["int64", "float64"])
            .columns.tolist()
        )

        return numerical_columns

    def _get_categorical_columns(self) -> list:
        """Identify categorical columns in the metadata.

        Returns:
        list: List of column names that are categorical.
        """
        categorical_columns = [
            col
            for col in self.metadata.columns[1:]
            if col not in self.metadata_numeric_columns
        ]

        return categorical_columns

    def _check_column_in_metadata(self, column: str):
        """Check if a column is in the metadata.

        Args:
        column (str): Column name.
        """
        if column not in self.metadata.columns:
            raise ValueError(f"Column {column} not found in metadata.")
        else:
            return True

    def _check_column_is_categorical(self, column: str):
        """Check if a column is categorical.

        Args:
        column (str): Column name.
        """
        if column not in self.metadata_categorical_columns:
            raise ValueError(f"Column {column} is not categorical.")
        else:
            return True

    def _get_color_map_for_features(self) -> dict:
        """Generate a color map for categorical features
        in the metadata. The color map is used for plotting.
        The color map is generated based on the unique values
        in the categorical columns. Only defined for columns
        with less than 50 unique values."""

        if len(self.metadata_categorical_columns) == 0:
            print("No categorical columns found in metadata.")
            return {}

        color_map = {}

        for column in self.metadata_categorical_columns:
            column_values = self.metadata[column].unique()
            if len(column_values) > 50:
                print(
                    f"Skipping {column} as it has more than \
                        50 unique values."
                )
                continue

            colors = (
                px.colors.qualitative.Set2
                + px.colors.qualitative.Set2
                * (len(column_values) - len(px.colors.qualitative.Set2))
            )

            colors = colors[: len(column_values)]  # shorten if necessary
            color_map[column] = dict(zip(column_values, colors))

            # check for specifial values
            if "True" in column_values:
                color_map[column]["True"] = "steelblue"

            if "False" in column_values:
                color_map[column]["True"] = "darkred"

        return color_map

    # Embeddings

    def _load_embeddings_from_dir(self, dir_path: str, file_extension: str):
        """Load embeddings from individual files in a directory.

        Args:
        dir_path (str): Path to the directory containing the individual files.
        file_extension (str): Extension of the embedding files. Default 'npy'.
        """

        embeddings = []
        for sample in self.sample_names:
            emb_file = os.path.join(dir_path, f"{sample}.{file_extension}")
            if not os.path.isfile(emb_file):
                raise ValueError(f"Embedding file '{emb_file}' not found.")
            embeddings.append(np.load(emb_file))
        return np.stack(embeddings)

    def _assign_colour_to_embedding_space(self, num_emb_spaces: int) -> str:
        """Assign a colour to the embedding space."""
        if num_emb_spaces > len(EMB_SPACE_COLORS):
            return EMB_SPACE_COLORS[num_emb_spaces]
        # If there are more embedding spaces than predefined colors,
        # cycle through the list
        return EMB_SPACE_COLORS[
            (num_emb_spaces - len(EMB_SPACE_COLORS)) % len(EMB_SPACE_COLORS)
        ]

    def add_emb_space(
        self,
        emb_space_name: str,
        embeddings_source: str,
        file_extension: str = "npy",
    ):
        """Add an embedding space to the Emma object.
        
        Args:
        embeddings_source (str): Path to either a .npy file or a \
            directory containing .npy files for each embedding.
        emb_space_name (str): Name of the embedding space. Must be unique.
        ext (str): Extension of the embedding files (default 'npy').
        
        If embeddings_source is a .npy file, it is loaded directly assuming \
            it contains all embeddings for the provided meta data in \
                respective order.
        If embedding_source is a directory, embeddings are loaded from files \
            in the directory corresponding to self.sample_names.
        """

        # Validate the embedding space name
        if not emb_space_name:
            raise ValueError("Embedding space name must be provided.")
        if emb_space_name in self.emb:
            raise ValueError(
                f"Embedding space '{emb_space_name}' already \
                exists."
            )

        # Load embeddings
        embeddings = None
        if embeddings_source.endswith(f".{file_extension}"):
            # Single .npy file
            if not os.path.isfile(embeddings_source):
                raise ValueError(
                    f"Embedding file '{embeddings_source}' not found."
                )
            embeddings = np.load(embeddings_source)
        elif os.path.isdir(embeddings_source):
            # Directory with .npy files
            embeddings = self._load_embeddings_from_dir(
                embeddings_source, file_extension
            )
        else:
            raise ValueError(
                (
                    "'embeddings_source' must be a .npy file or \
                        a directory path."
                )
            )

        # Validate the number of embeddings
        if embeddings.shape[0] != len(self.sample_names):
            raise ValueError(
                (
                    "Number of embeddings does not match the number \
                        of samples in the metadata."
                )
            )

        # Add the embedding space
        self.emb[emb_space_name] = {
            "emb": embeddings,
            "colour": self._assign_colour_to_embedding_space(len(self.emb)),
        }

        print(f"Embedding space '{emb_space_name}' added successfully.")
        print(f"Embeddings have {embeddings.shape[1]} features each.")

    def _check_for_emb_space(self, emb_space_name: str):
        """Check if an embedding space is available.

        Args:
        emb_space_name (str): Name of the embedding space.
        """
        if emb_space_name not in self.emb:
            raise ValueError(f"Embedding space {emb_space_name} not found.")

    def remove_emb_space(self, emb_space_name: str):
        """Remove an embedding space from the Emma object.

        Args:
        emb_space_name (str): Name of the embedding space.
        """
        self._check_for_emb_space(emb_space_name)
        del self.emb[emb_space_name]
        print(f"Embedding space '{emb_space_name}' removed.")

    # Pairwise distances
    def __compute_pairwise_distances(
        self, emb_space: str, metric: str, embeddings: np.ndarray
    ):
        """Calculate pairwise distances between samples in an embedding space.

        Args:
        emb_space (str): Name of the embedding space.
        metric (str): Distance metric to use.
        """
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        if metric == "sqeuclidean_normalised":
            # divide each row by its norm
            emb = self.emb[emb_space]["emb"]
            emb_norm = np.linalg.norm(emb, axis=1)
            emb = emb / emb_norm[:, None]  # divide each row by its norm
            emb_pwd = squareform(pdist(emb, metric="sqeuclidean"))
            return emb_pwd

        elif metric == "euclidean_normalised":

            # divide each row of the emb by its norm
            emb = self.emb[emb_space]["emb"]
            emb_norm = np.linalg.norm(emb, axis=1)
            emb = emb / emb_norm[:, None]  # divide each row by its norm
            emb_pwd = squareform(pdist(emb, metric="euclidean"))
            return emb_pwd

        elif metric == "cityblock_normalised":
            emb_pwd = squareform(
                pdist(self.emb[emb_space]["emb"], metric="cityblock")
            )
            emb_pwd = emb_pwd / len(self.emb[emb_space]["emb"][1])
            return emb_pwd

        elif metric == "adjusted_cosine":
            # substract the mean of each column from each value
            emb = self.emb[emb_space]["emb"]
            emb = emb - np.median(emb, axis=0)  # emb.median(axis=0)
            emb_pwd = squareform(pdist(emb, metric="cosine"))
            return emb_pwd

        emb_pwd = squareform(pdist(embeddings, metric=metric))
        return emb_pwd

    def calculate_pairwise_distances(
        self, emb_space: str, metric: str = "euclidean"
    ):
        """Calculate pairwise distances between samples in an embedding space.\
            Will store the distances in the Emma object.
            Will also calculate and store the ranks based on the distances.

        Args:
        emb_space (str): Name of the embedding space.
        metric (str): Distance metric to use. Default 'euclidean'.
        """
        self._check_for_emb_space(emb_space)
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        if metric not in self.emb[emb_space].get("pairwise_distances", {}):
            print(f"Calculating pairwise distances using {metric}...")

            emb_pwd = self.__compute_pairwise_distances(
                emb_space, metric, self.emb[emb_space]["emb"]
            )

            # Compute ranks based on distances
            ranked_indices = np.argsort(emb_pwd, axis=1)

            if "pairwise_distances" not in self.emb[emb_space]:
                self.emb[emb_space]["pairwise_distances"] = {}
            if "ranks" not in self.emb[emb_space]:
                self.emb[emb_space]["ranks"] = {}

            self.emb[emb_space]["pairwise_distances"][metric] = emb_pwd
            self.emb[emb_space]["ranks"][metric] = ranked_indices

        else:
            print(f"Pairwise distances using {metric} already calculated.")

    def get_pairwise_distances(
        self, emb_space: str, metric: str = "euclidean"
    ) -> np.ndarray:
        """Get pairwise distances between samples in an embedding space. \
            Will calculate the distances if not already done.

        Args:
        emb_space (str): Name of the embedding space.
        metric (str): Distance metric to use. Default 'euclidean'.

        Returns:
        np.ndarray: Pairwise distances.
        """
        self._check_for_emb_space(emb_space)
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        if metric not in self.emb[emb_space].get("pairwise_distances", {}):
            self.calculate_pairwise_distances(
                emb_space=emb_space, metric=metric
            )

        return self.emb[emb_space]["pairwise_distances"][metric]

    def get_knn(
        self, emb_space: str, k: int, metric: str = "euclidean"
    ) -> np.ndarray:
        """Get the k-nearest neighbours for each sample in an embedding space. \
            Will calculate the neighbours if not already done.

        Args:
        emb_space (str): Name of the embedding space.
        k (int): Number of neighbours to consider.
        metric (str): Distance metric to use. Default 'euclidean'.

        Returns:
        np.ndarray: Indices of the k-nearest neighbours.
        """

        # Validate input
        self._check_for_emb_space(emb_space)
        if k < 1:
            raise ValueError("k must be a positive integer.")
        if k > len(self.sample_names):
            raise ValueError("k must be less than the number of samples.")
        if metric not in DISTANCE_METRIC_ALIASES:
            raise ValueError(f"Distance metric {metric} not supported.")

        try:
            ranked_indices = self.emb[emb_space]["ranks"][metric]
        except KeyError:
            self.calculate_pairwise_distances(emb_space, metric)
            ranked_indices = self.emb[emb_space]["ranks"][metric]

        return ranked_indices[:, 1 : k + 1]
