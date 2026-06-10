#!/bin/bash
#SBATCH --job-name=uniref50_c30
#SBATCH --output=/home/nap/lperez_nn/slurms/log/uniref50_c30.out
#SBATCH --error=/home/nap/lperez_nn/slurms/err/uniref50_c30.err
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

cd /home/nap/lperez_nn/data/data_uniref50

/home/nap/lperez_nn/data/data_uniref50/mmseqs2/mmseqs/bin/mmseqs easy-cluster uniref50.fasta uniref50_c30 tmp \
    --min-seq-id 0.3 \
    -c 0.8 \
    --cov-mode 1
