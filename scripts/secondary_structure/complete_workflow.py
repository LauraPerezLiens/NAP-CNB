import os
import csv
import subprocess
import time
from pathlib import Path

# Path to Python environment
CONDA_ENV = "/home/nap/lperez_nn/miniconda3/envs/protein_pipeline/bin/python"

# Directories
intermediate_dir = Path("/home/nap/lperez_nn/data/data_intermediate")
raw_dir = Path("/home/nap/lperez_nn/data/data_raw")
intermediate_dir.mkdir(exist_ok=True)

# FASTA files
fasta_files = {
    "human": intermediate_dir / "fasta_human.csv",
    "mouse": intermediate_dir / "fasta_mouse.csv"
}

# 1. Predict secondary structure for all proteins
for species, fasta_csv in fasta_files.items():
    print(f"----- [{species}] Predicting secondary structure for all proteins -----")
    secondary_struct_csv = intermediate_dir / f"secondary_struct_{species}.csv"
    start_time = time.time()
    subprocess.run([
        CONDA_ENV,
        "./ProteinUnet_run.py",
        "-i", str(fasta_csv),
        "-o", str(secondary_struct_csv)
    ], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "TF_CPP_MIN_LOG_LEVEL": "3"})
    end_time = time.time()
    print(f"Secondary structure prediction time [{species}]: {int(end_time - start_time)}s")

# 2. For each haplotype, generate epitope + secondary structure file
for species in ["human", "mouse"]:
    merged_files = list(raw_dir.glob(f"{species}/*/*/merged_tcell%mhc.csv"))
    # Load complete secondary structures
    sec_struct_dict = {}
    sec_struct_csv = intermediate_dir / f"secondary_struct_{species}.csv"
    with open(sec_struct_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec_struct_dict[row['protein_id']] = row['secondary_structure']
    for merged_file in merged_files:
        haplotype = merged_file.parts[-2]
        # Create output folder for each haplotype
        haplo_dir = intermediate_dir / species / haplotype
        haplo_dir.mkdir(parents=True, exist_ok=True)
        out_epitope_struct = haplo_dir / f"epitope_struct_{haplotype}_{species}.csv"
        with open(merged_file, 'r') as f_in, open(out_epitope_struct, 'w', newline='') as f_out:
            reader = csv.DictReader(f_in)
            writer = csv.writer(f_out)
            writer.writerow(['protein_id', 'epitope_sequence', 'secondary_structure_fragment', 'epitope__starting_position', 'epitope__ending_position'])
            for row in reader:
                pid = row['protein_id']
                start = int(row['epitope__starting_position'])
                end = int(row['epitope__ending_position'])
                # Get epitope sequence and secondary structure fragment
                # TODO: Implement loading the real FASTA sequence for the protein
                # epitope_seq = get_fasta_fragment(pid, start, end)
                sec_struct = sec_struct_dict.get(pid, '')
                sec_struct_frag = sec_struct[start-1:end] if sec_struct else ''
                writer.writerow([pid, 'epitope_seq_placeholder', sec_struct_frag, start, end])

# 3. Run BERT embeddings for each epitope_struct file
vocab_file_2 = "/home/nap/lperez_nn/scripts/secondary_structure_and_BERT/secondary-vocab.txt"
bert_config_file_2 = "/home/nap/lperez_nn/scripts/secondary_structure_and_BERT/bert_config.json"
init_checkpoint_2 = "/home/nap/lperez_nn/scripts/secondary_structure_and_BERT/model/model.ckpt-500000"

for species in ["human", "mouse"]:
    merged_files = list(raw_dir.glob(f"{species}/*/*/merged_tcell%mhc.csv"))
    for merged_file in merged_files:
        haplotype = merged_file.parts[-2]
        epitope_struct_file = intermediate_dir / f"epitope_struct_{haplotype}_{species}.csv"
        out_bert_secondary = intermediate_dir / f"out_bert_secondary_{haplotype}_{species}.txt"
        print(f"----- [{species}][{haplotype}] BERT embeddings for epitopes and secondary structure -----")
        start_time = time.time()
        subprocess.run([
            CONDA_ENV,
            "./extract_features4.py",
            "--input_file", str(epitope_struct_file),
            "--output_file", str(out_bert_secondary),
            "--vocab_file", vocab_file_2,
            "--bert_config_file", bert_config_file_2,
            "--init_checkpoint", init_checkpoint_2,
            "--max_seq_length", "30",
            "--layers", "-1",
            "--batch_size", "32"
        ], check=True)
        end_time = time.time()
        print(f"BERT embedding time [{species}][{haplotype}]: {int(end_time - start_time)}s")