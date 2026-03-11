#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys
import math
from pathlib import Path

WINDOW_SIZE = {"human": 25, "mouse": 12}

# Check arguments 
if len(sys.argv) != 3:
    print("Usage: python epitope_classification.py <proteins_csv> <epitope_file.csv>")
    sys.exit(1)

proteins_csv = sys.argv[1]
epitope_file = sys.argv[2]

# Detect species from filename
species = "human" if "human" in proteins_csv else "mouse"
WIN_SIZE = WINDOW_SIZE[species]
WINDOW_CENTER = math.ceil(WIN_SIZE / 2)
MAX_CENTER_DIST = WINDOW_CENTER - 1
output_cols = [f"{WIN_SIZE}aa_seq", "contains_epitope", "selected_epitope", "epitope_pos_score", "protein_id", "window_start"]

# Load proteins.csv
if not os.path.exists(proteins_csv):
    print(f"ERROR: {proteins_csv} not found.")
    sys.exit(1)
proteins_df = pd.read_csv(proteins_csv)

# Load epitope CSV
epitopes_df = pd.read_csv(epitope_file)

# Keep only epitopes with defined start and end
epitopes_df = epitopes_df.dropna(subset=["start", "end"])
epitopes_df = epitopes_df.astype({"start": int, "end": int})

# Filter proteins to keep only those present in the epitope file
proteins_df = proteins_df[proteins_df["protein_id"].isin(epitopes_df["protein_id"].unique())].reset_index(drop=True)


def contains_epitope(window, epitopes_inside):
    for e in epitopes_inside.itertuples():
        if (window[0] <= e.start) and (window[1] >= e.end):
            return 1
    return 0

def epitope_position_score(window, epitopes_inside):
    results = []
    for e in epitopes_inside.itertuples():
        if (window[0] <= e.start) and (window[1] >= e.end):
            ep_center = (e.start + e.end) / 2.0
            ep_center_in_window = ep_center - window[0] + 1
            signed_dist = ep_center_in_window - WINDOW_CENTER
            dist = abs(signed_dist)
            score = min(1.0, dist / MAX_CENTER_DIST)
            signed_score = math.copysign(score, signed_dist)
            results.append((round(signed_score, 3), e.epitope))
    return results

rows = []
n_proteins = len(proteins_df)
print(f"Starting sliding window: {n_proteins} proteins total (window={WIN_SIZE})")
PROGRESS_EVERY = 100

for idx, p in enumerate(proteins_df.itertuples(), start=1):
    protein_id = p.protein_id
    seq = p.aas

    if idx % PROGRESS_EVERY == 0 or idx == 1:
        print(f"Processing protein {idx} / {n_proteins}")

    if len(seq) < WIN_SIZE:
        continue

    prot_epitopes = epitopes_df[epitopes_df["protein_id"] == protein_id]
    if len(prot_epitopes) <= 0:
        continue

    for i in range(len(seq) - WIN_SIZE + 1):
        window_seq = seq[i:i + WIN_SIZE]
        window = [i + 1, i + WIN_SIZE]
        epitope_scores = epitope_position_score(window, prot_epitopes)
        if epitope_scores:
            for pos_score, selected_epitope in epitope_scores:
                rows.append([window_seq, 1, selected_epitope, pos_score, protein_id, i + 1])
        else:
            rows.append([window_seq, 0, None, 0, protein_id, i + 1])

# Save output
base_name = os.path.splitext(os.path.basename(proteins_csv))[0]
species_dir = Path(proteins_csv).parent / species
os.makedirs(species_dir, exist_ok=True)
out_name = f"{base_name}_epitope_marked.csv"
out_path = os.path.join(species_dir, out_name)

output_df = pd.DataFrame(rows, columns=output_cols)
output_df = output_df.drop_duplicates().reset_index(drop=True)
output_df.to_csv(out_path, index=False, float_format="%.3f")

print(f"Saved: {out_path}")
print(f"Total windows: {len(output_df)}")

# When saving output files, create a folder for each species and haplotype inside data_intermediate
output_base = Path("/home/nap/lperez_nn/data/data_intermediate")
# Example usage inside your loop:
output_dir = output_base / species / haplotype
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"classification_{haplotype}_{species}.csv"
# Save your results to output_file