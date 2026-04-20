#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

INPUT_FASTA = "/home/nap/lperez_nn/data/data_uniref50/uniref50_c30_rep_seq.fasta"
OUT_DIR = "/home/nap/lperez_nn/data/data_uniref50/secondary_structure_prediction/chunks"

SEQS_PER_CHUNK = 100000  # puedes subir luego a 20000 o 50000

os.makedirs(OUT_DIR, exist_ok=True)

chunk_idx = 1
seq_count_in_chunk = 0
total_seqs = 0
out_f = None

def open_chunk(idx):
    path = os.path.join(OUT_DIR, f"chunk_{idx:05d}.fasta")
    return open(path, "w"), path

with open(INPUT_FASTA, "r") as f:
    for line in f:
        if line.startswith(">"):
            if seq_count_in_chunk == 0:
                out_f, current_path = open_chunk(chunk_idx)
                print(f"[INFO] Creating {current_path}")

            elif seq_count_in_chunk >= SEQS_PER_CHUNK:
                out_f.close()
                chunk_idx += 1
                seq_count_in_chunk = 0
                out_f, current_path = open_chunk(chunk_idx)
                print(f"[INFO] Creating {current_path}")

            seq_count_in_chunk += 1
            total_seqs += 1

        if out_f is not None:
            out_f.write(line)

if out_f is not None:
    out_f.close()

print(f"[OK] Total sequences: {total_seqs}")
print(f"[OK] Total chunks: {chunk_idx}")
