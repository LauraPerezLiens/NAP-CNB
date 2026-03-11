#!/bin/bash
#SBATCH --job-name=proteins_from_data_raw
#SBATCH --output=/home/nap/lperez_nn/slurms/log/proteins_from_data_raw_%j.log
#SBATCH --error=/home/nap/lperez_nn/slurms/err/proteins_from_data_raw_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --partition=standard
#SBATCH --nodelist=hertz-tres

export CONDA_EXE=/home/nap/lperez_nn/miniconda3/bin/conda
export PATH=/home/nap/lperez_nn/miniconda3/bin:$PATH
source /home/nap/lperez_nn/miniconda3/etc/profile.d/conda.sh
conda activate base  # Cambia 'base' por el nombre de tu entorno si tienes uno específico

python /home/nap/lperez_nn/scripts/proteins_from_data_raw.py
