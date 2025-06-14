# EmmaEmb

EmmaEmb is a Python library designed to facilitate the initial **comparison** of diverse **embedding spaces** in embeddings for  molecular biology. By incorporating **user-defined feature data** on the natural grouping of data points, EmmaEmb enables users to compare global statistics and understand the differences in clustering of natural groupings across different embedding spaces.

Although designed for the application on embeddings of molecular biology data (e.g. protein sequences), the library is general and can be applied to any type of embedding space.

## How to cite

If you use EmmaEmb, please cite the [pre-print](https://www.biorxiv.org/content/10.1101/2024.06.21.600139v2):

- *Rissom, P. F., Yanez Sarmiento, P., Safer, J., Coley, C. W., Renard, B. Y., Heyne, H. O., Iqbal, S. Decoding protein language models: insights from embedding space analysis. bioRxiv (2025), [https://doi.org/10.1101/2024.06.21.600139](https://doi.org/10.1101/2024.06.21.600139)*

or, if you prefer the `BibTeX` format:

```
@article {Rissom2024.06.21.600139,
	author = {Rissom, Pia Francesca and Sarmiento, Paulo Yanez and Safer, Jordan and Coley, Connor W. and Renard, Bernhard Y. and Heyne, Henrike O. and Iqbal, Sumaiya},
	title = {Decoding protein language models: insights from embedding space analysis},
	year = {2025},
	doi = {10.1101/2024.06.21.600139},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```

## Overview

- **[Workflow](#workflow)**
- **[Input](#input)**
- **[Features](#features)**
    - **[Visualisation after dimensionality reduction](#visualisation-after-dimensionality-reduction)**
    - **[Computation of pairwise distances](#computation-of-pairwise-distances)**
    - **[Feature distribution across spaces](#feature-distribution-across-spaces)**
    - **[Pairwise space comparison](#pairwise-space-comparison)**
- **[Installation](#installation)**
- **[Scripts for protein language model embeddings](#scripts-for-protein-language-model-embeddings)**
- **[License](#license)**


## Workflow

The following figure provides an overview of the EmmaEmb workflow:

![EmmaEmb workflow](images/emma_overview.jpg)


EmmaEmb enables the comparative analysis of information captured in different embedding spaces. The workflow consists of the following steps:

**A. Embedding Generation**: Starting with a set of samples (e.g., proteins or genes), embeddings are extracted from multiple foundation models, which may differ in architecture or training.

**B. Feature Integration**: Sample-specific categorical data (e.g., functional annotations, protein families) is incorporated to the analysis.

**C. Feature Distribution Analysis**: The distribution of categorical features is assessed within local neighborhoods in each embedding space, using k-nearest neighbors to quantify class consistency and overlap.

**D. Pairwise Space Comparison**: Embedding spaces are compared based on pairwise distances and neighborhood similarity to identify global and local differences. Regions with high divergence can be further examined using feature data to understand variations in model representation.

## Input

EmmaEmb is centered around the `Emma` object, which serves as the core of the library. The following input data is required:

1. **Feature Data**: A pandas DataFrame containing sample-specific categorical features. Each row corresponds to a sample, and each column corresponds to a feature. The first column should contain the sample IDs.

2. **Embedding Spaces**: Precomputed embeddings for each sample (scripts for generating embeddings from protein language models are [provided](#scripts-for-protein-language-model-embeddings)). Embeddings should be stored in a directory with one file per sample. The file name should correspond to the sample ID, and the file should contain the embedding as a list of floats. Multiple embedding spaces can be added to the Emma object for comparison. Dimensions do not need to match across spaces.

The `Emma` object is initialized with feature data and embedding spaces can be added incrementally. 



## Features

### Visualisation after dimensionality reduction

EmmaEmb supports dimensionality reduction techniques such as PCA, t-SNE, and UMAP to visualize and analyze high-dimensional embeddings in lower-dimensional spaces. The plots can be colour coded by a feature of interest from the feature data.

### Computation of pairwise distances

To make embedding spaces comparable, EmmaEmb analyses rely on comparing not individual embeddings, but the relationships between them. The library calculates pairwise distances between samples in each embedding space. Users can select from multiple distance metrics, including:

- Euclidean
- Cosine
- Manhattan

For parts of the analysis only the k-nearest neighbors are considered, which will be based on the pairwise distances. The pairwise distances are only calculated once and can be reused for multiple analyses.

The distances can be visually inspected in a heatmap.


### Feature distribution across spaces

For a selected feature from the feature data, EmmaEmb provides two metrics to assess the alignment of features across embedding spaces:

- **KNN feature alignment scores**: Quantify the alignment of features by examining the nearest neighbors of each sample in different spaces. This score reveals the extent to which samples with a shared feature are embedded close to each other in different spaces.
- **KNN class similarity matrix**: Measure the consistency of class-level relationships by assessing the overlap of nearest neighbors for samples within the same class across spaces. This provides insights into the relationships between classes in different embedding spaces.

### Pairwise space comparison

EmmaEmb provides two metrics to directly compare two embedding spaces:

- **Global comparison of pairwise distances**: Compare the distribution of pairwise distances between samples in two embedding spaces. This metric is useful for assessing the overall similarity of the two spaces. The pairwise distances can also be visualized in a scatter plot.
- **Cross-space neighborhood similarity**: Assess the similarity of local neighborhoods in two embedding spaces. This metric is useful for identifying regions where the two spaces diverge. The similarity is calculated based on the overlap of k-nearest neighbors between samples in the two spaces. The regions of divergence can be characterized using the feature data.


## Installation

You can install the EmmaEmb library through pip, or access examples locally by cloning the github repo.

### Installing the EmmaEmb library
```
pip install emmaemb
```

### Cloning the EmmaEmb repo
```
git clone https://github.com/broadinstitute/EmmaEmb

cd emmaemb                 # enter project directory
pip3 install .                 # install dependencies
jupyter lab colab_notebooks    # open notebook examples in jupyter for local exploration
```

### Getting Started

To get started with the EmmaEmb library, load the metadata and embeddings, and initialize the `Emma` object. The following code snippet demonstrates how to use EmmaEmb to compare two embedding spaces:

```python
from emmaemb import Emma
from emmaemb.vizualization import *

# Initialize Emma object with feature data
emma = Emma(feature_data=feature_data)

# Add embedding spaces
emma.add_embedding_space("ProtT5", "embeddings/prot_t5_embeddings")
emma.add_embedding_space("ESM2", "embeddings/esm2_embeddings")

# Compute pairwise distances
emma.calculate_pairwise_distances("ProtT5", "cosine")
emma.calculate_pairwise_distances("ESM2", "cosine")

# Plot space after dimensionality reduction
fig_1 = plot_emb_space(
    emma, emb_space="ProtT5", color_by="enzyme_class", method="PCA"
)

# Analyze global comparison of pairwise distances
fig_2 = plot_pairwise_distance_comparison(
    emma, emb_space_x="ProtT5", emb_space_y="ESM2", metric="cosine", group_by="species"
)

# Analyze feature distribution across spaces
fig_3 = plot_knn_alignment_across_embedding_spaces(
    emma, feature="enzyme_class", k=10, metric="cosine"
)
```

A more detailed example can be found in the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/broadinstitute/EmmaEmb/blob/main/examples/Pla2g2/emmaemb_pla2g2.ipynb) notebook.

## Scripts for protein language model embeddings

The repository also contains a wrapper [script](plm_embeddings/get_embeddings.py) for retrieving protein embeddings from a diverse set of pre-trained Protein Language Models. 

The script includes a heuristic to chunk and aggregate long sequences to ensure compatibility with the models' input size constraints.

The script supports the following models:

- [Ankh](https://github.com/agemagician/Ankh)
- [ProtT5](https://github.com/agemagician/ProtTrans)
-  [ProstT5](https://github.com/mheinzinger/ProstT5)
- [ESM1 and ESM2](https://github.com/facebookresearch/esm)
- [ESM3 and ESMC](https://github.com/evolutionaryscale/esm)



## Contact 

If you have any questions or suggestions, please feel free to reach out to the authors: francesca.risom@hpi.de.


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
