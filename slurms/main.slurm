#!/bin/bash
#SBATCH --job-name=iedb_full_workflow
#SBATCH --output=/home/nap/lperez_nn/slurms/log/iedb_full_workflow_%j.log
#SBATCH --error=/home/nap/lperez_nn/slurms/err/iedb_full_workflow_%j.err
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --nodelist=hertz-tres

export CONDA_EXE=/home/nap/lperez_nn/miniconda3/bin/conda
export PATH=/home/nap/lperez_nn/miniconda3/bin:$PATH
source /home/nap/lperez_nn/miniconda3/etc/profile.d/conda.sh
conda activate base  

# Ejecuta el workflow completo
python /home/nap/lperez_nn/scripts/export_iedb_mhc_tcell_data.py
python /home/nap/lperez_nn/scripts/proteins_from_data_raw.py
