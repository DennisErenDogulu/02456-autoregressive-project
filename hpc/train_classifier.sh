#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train_classifier
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train_classifier.out
#BSUB -e hpc/logs/train_classifier.err

# Phase 2: freeze the GRU and fit the family head on top of it.
#   bsub < hpc/train_classifier.sh
#
# Run hpc/train.sh FIRST - this job needs the backbone checkpoint that it writes, and
# --backbone below has to name that same file.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

set -eu
cd "$(dirname "$0")/.."
source hpc/env.sh   # uv on the PATH, caches off the home quota

uv run python train_classifier.py \
  --backbone results/checkpoints/gru_h256_l2.pt \
  --head-hidden-dim 128 \
  --dropout 0.1 \
  --epochs 30
