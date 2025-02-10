import os
import pandas as pd
import numpy as np

# script to find all embeddings for the proteins defined in the pla2g2 dataset
fp_pla2g2 = "examples/Pla2g2/Pla2g2_features.csv"
df_pla2g2 = pd.read_csv(fp_pla2g2)
pla2g2_proteins = df_pla2g2["identifier"].values

embedding_dirs = {
    "ProtT5": "Rostlab/prot_t5_xl_uniref50/layer_None/chopped_1022_overlap_300",
    "ESMC": "esmc-300m-2024-12/layer_None/chopped_1022_overlap_300",
}

output_dir = "examples/Pla2g2/embeddings/"
# ensure output_dir exists
os.makedirs(output_dir, exist_ok=True)

for embedding_model in embedding_dirs:
    print(f"Embedding model: {embedding_model}")
    embedding_dir = embedding_dirs[embedding_model]
    for protein in pla2g2_proteins:
        fp_embedding = f"embeddings/{embedding_dir}/{protein}.npy"
        try:
            # copy file into new directory output_dir + embedding_model
            embedding = np.load(fp_embedding)
            output_fp = f"{output_dir}/{embedding_model}/{protein}.npy"
            os.makedirs(os.path.dirname(output_fp), exist_ok=True)
            np.save(output_fp, embedding)
            print(f"Embedding found for protein {protein}")
        except FileNotFoundError:
            print(f"Embedding not found for protein {protein}")


print()
