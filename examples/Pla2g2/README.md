# PLA2G2 example data and scripts

This directory contains a demonstration on how to use the `EmmaEmb` package using the PLA2G2 dataset.

## Data

The data was published by

> Ivan Koludarov, Timothy NW Jackson, Vivek Suranse, et
al. Reconstructing the evolutionary history of a functionally
diverse gene family reveals complexity at the genetic origins
of novelty. bioRxiv, 2020.

and can be accessed [here](https://github.com/tsenoner/protspace).

Contains proteins of the PLA2G2 enzyme family, their amino acid sequence and annotations for gene, group, enzyme_class  and species.

## Pre-processing

Only proteins with information for all features was used for the analysis. Leading to a subset of 446 proteins. A feature column for sequence length and sequence length in bins was added.

## Embeddings

Embeddings were retrieved by the following protein language models:

### ProtT5
- Model version: prot_t5_xl_uniref50 
- Github: https://github.com/agemagician/ProtTrans
- Reference: *Elnaggar, Ahmed, et al. "ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing." IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. https://doi.org/10.1109/TPAMI.2021.3095381.*

### ESMC
- Model version: esmc-300m-2024-12
- Github: https://github.com/evolutionaryscale/esm
- Reference: *ESM Team. "ESM Cambrian: Revealing the mysteries of proteins with unsupervised learning." EvolutionaryScale Website, December 4, 2024. https://evolutionaryscale.ai/blog/esm-cambrian.*

