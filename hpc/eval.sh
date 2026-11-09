#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/logs/eval.out
#BSUB -e hpc/logs/eval.err

# Score sample sets on a GPU, structures included:   bsub < hpc/eval.sh
#
# Run `bash hpc/setup_env.sh` once before this, to install the environment and download the
# ESMFold weights. Add your own sample CSVs to the list below; a name with no CSV is skipped with
# a message rather than failing the job, so this can be submitted before every model has been
# sampled. Each set lands in results/eval/<name>/.
#
# Uncomment the #BSUB -u/-B/-N lines to be emailed when the job starts and ends.
##BSUB -u <EMAIL_ADDRESS>
##BSUB -B
##BSUB -N

source hpc/env.sh

uv run --group eval python eval/eval.py --samples results/sequences/mock_data.csv --fold   
