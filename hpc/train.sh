#!/bin/sh
#BSUB -q gpuv100
#BSUB -J train
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/train.out
#BSUB -e hpc/logs/train.err

# Submit your training run to a GPU queue:   bsub < hpc/train.sh
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

#set -eu
#cd "$(dirname "$0")/.."
source hpc/env.sh   # uv on the PATH, caches off the home quota

# The checkpoint is named explicitly so hpc/train_classifier.sh can point its
# --backbone at the same file; train.py would otherwise write results/checkpoints/gru.pt.
uv run python train.py --model gru \
  --embedding-dim 32 \
  --hidden-dim 256 \
  --num-layers 2 \
  --dropout 0.1 \
  --checkpoint results/checkpoints/gru_h256_l2.pt
