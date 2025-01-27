#!/bin/bash
#SBATCH -A renard
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=./.slurm/%j.out

eval "$(conda shell.bash hook)"
conda activate emma
python -m experiments.clustering_experiments
