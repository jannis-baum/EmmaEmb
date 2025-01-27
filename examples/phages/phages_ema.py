import os
import numpy as np
import pandas as pd

from emma.ema import EmbeddingHandler
from emma.cluster_metrics import compare_clusterings
from emma.knn_analysis import compare_knn_fraction_across_embeddings, plot_knn_fraction_heatmap, plot_class_mixing_heatmap, plot_heatmap_feature_distribution, analyze_low_similarity_distributions, plot_low_similarity_class_representation, plot_low_similarity_class_proportions, plot_high_similarity_class_proportions, plot_class_mixing_stacked_bar, plot_class_mixing_stacked_bar_absolute, compare_knn_statistics_across_embeddings

fp_metadata = "examples/phages/phages_metadata.csv"
embedding_dir = "embeddings/"
models = {
    "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
    "ESM2_3B": "esm2_t36_3B_UR50D/layer_36/chopped_1022_overlap_300",
    #"ProstT5": "Rostlab/ProstT5/layer_None/chopped_1022_overlap_300"
    }


metadata = pd.read_csv(fp_metadata)

ema = EmbeddingHandler(sample_meta_data=metadata)         
for model_alias, model_name in models.items():    
    ema.add_emb_space(embeddings_source=embedding_dir + model_name, 
                      emb_space_name=model_alias)
    
print()