import pandas as pd
from scipy.sparse import csr_matrix
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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

emb_ProtT5 = ema.emb['ProtT5']['emb']

X = csr_matrix(emb_ProtT5)

adata = sc.AnnData(X, obs=metadata)
adata.obsm['ESM2'] = ema.emb['ESM2']['emb']

adata.obs = metadata

sc.pp.neighbors(adata)
sc.tl.umap(adata)

sc.pp.neighbors(adata, use_rep='ESM2')  # Use the second embedding space
sc.tl.leiden(adata)

sc.pl.umap(adata, color='leiden')

sc.tl.leiden(adata, resolution=1.0)
sc.pl.umap(adata, color=['leiden'])

cos_sim = cosine_similarity(adata.X, adata.obsm['ESM2'])
print()