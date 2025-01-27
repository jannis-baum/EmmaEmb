#!/bin/bash
#SBATCH -A renard
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=5-00:00:00
#SBATCH --output=./.slurm/%j.out

eval "$(conda shell.bash hook)"
conda activate emma
python -m emma.embedding.get_embeddings

