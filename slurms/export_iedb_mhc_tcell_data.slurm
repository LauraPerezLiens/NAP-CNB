#!/bin/bash
#SBATCH --job-name=iedb_export
#SBATCH --output=/home/nap/lperez_nn/NAP-CNB/slurms/log/iedbworkflow_%j.log
#SBATCH --error=/home/nap/lperez_nn/NAP-CNB/slurms/err/iedbworkflow_%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --partition=standard


echo "====================================="
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "====================================="

python3 /home/nap/lperez_nn/NAP-CNB/scripts/data_download/download_iedb_epitopes.py

echo "====================================="
echo "Job finished: $(date)"
echo "====================================="