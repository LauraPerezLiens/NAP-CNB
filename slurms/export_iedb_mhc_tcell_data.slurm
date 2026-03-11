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


python3 /home/nap/lperez_nn/scripts/dump_mhc_tcell_mouse_human.py
