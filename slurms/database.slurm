#!/bin/bash
#SBATCH --job-name=iedb_workflow
#SBATCH --output=/home/nap/lperez_nn/slurms/log/iedbworkflow_%j.log
#SBATCH --error=/home/nap/lperez_nn/slurms/err/iedbworkflow_%j.err
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=standard
#SBATCH --nodelist=hertz-tres


echo "====================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "JobID: $SLURM_JOB_ID"
echo "====================================="

export CONDA_EXE=/home/nap/lperez_nn/miniconda3/bin/conda
export PATH=/home/nap/lperez_nn/miniconda3/bin:$PATH
source /home/nap/lperez_nn/miniconda3/etc/profile.d/conda.sh
conda activate base  # Cambia 'base' por el nombre de tu entorno si tienes uno específico
python /home/nap/lperez_nn/scripts/export_iedb_mhc_tcell_data.py

echo "====================================="
echo "Job finished at: $(date)"
echo "====================================="