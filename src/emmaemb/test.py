import pandas as pd
from src.emmaemb.core import Emma
from src.emmaemb.vizualization import *

fp_metadata = "examples/Pla2g2/Pla2g2_features.csv"
embedding_dir = "embeddings/"
models = {
    "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
    "ESMC": "esmc-300m-2024-12/layer_None/chopped_1022_overlap_300",
}

metadata = pd.read_csv(fp_metadata)
emma = Emma(feature_data=metadata)

for model_alias, model_name in models.items():
    emma.add_emb_space(
        embeddings_source=embedding_dir + model_name,
        emb_space_name=model_alias,
    )

fig_pca = plot_emb_space(emma, "ProtT5")

pwd = emma.get_pairwise_distances("ProtT5", "cityblock")

emma.calculate_knn("ProtT5", 5)

print()
