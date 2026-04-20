#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
from Bio.Align import substitution_matrices

blosum62 = substitution_matrices.load("BLOSUM62")
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")


def get_blosum_score(a1, a2):
    try:
        return float(blosum62[a1, a2])
    except Exception:
        return -4.0


BEST_CHANGE = {}
for aa in amino_acids:
    candidates = [x for x in amino_acids if x != aa]
    BEST_CHANGE[aa] = max(candidates, key=lambda x: get_blosum_score(aa, x))


def most_probable_change(aa):
    return BEST_CHANGE.get(aa, aa)


def build_variants(seq):
    """
    Devuelve:
    - la secuencia original
    - una variante por cada posición, cambiando 1 aa
    """
    variants = [seq]
    seq_list = list(seq)

    for i, aa in enumerate(seq_list):
        change = most_probable_change(aa)
        mutated = seq_list.copy()
        mutated[i] = change
        variants.append("".join(mutated))

    return variants


def detect_seq_column(df):
    if "25aa_seq" in df.columns:
        return "25aa_seq"
    raise ValueError("No encuentro '25aa_seq' en el input.")


def flush_buffer(rows_buffer, output_path, write_header):
    if not rows_buffer:
        return write_header, 0

    chunk_df = pd.DataFrame(
        rows_buffer,
        columns=[
            "original_seq",
            "blosum_seq",
            "group_id",
            "contains_epitope",
            "selected_epitope",
            "epitope_pos_score",
        ],
    )

    chunk_df.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        float_format="%.3f"
    )

    written = len(chunk_df)
    return False, written


def main():
    if len(sys.argv) < 2:
        print("Uso: python blosum62.py input.csv [output.csv]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_blosum{ext}"

    df = pd.read_csv(input_path)

    seq_col = detect_seq_column(df)

    required_cols = [seq_col, "contains_epitope", "selected_epitope", "epitope_pos_score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el input: {missing}")

    df = df[required_cols].reset_index(drop=True)

    print(f"Input rows: {len(df)}")
    print(f"Unique {seq_col}: {df[seq_col].nunique()}")

    rows_buffer = []
    write_header = True
    chunk_size = 100000
    total_output_rows = 0
    total_groups = 0

    for group_id, (_, row) in enumerate(df.iterrows(), start=1):
        original_seq = str(row[seq_col]).strip().upper()
        contains_epitope = int(row["contains_epitope"])
        selected_epitope = row["selected_epitope"]
        epitope_pos_score = row["epitope_pos_score"]

        if not original_seq:
            continue

        total_groups += 1
        variants = build_variants(original_seq)

        for blosum_seq in variants:
            rows_buffer.append([
                original_seq,
                blosum_seq,
                group_id,
                contains_epitope,
                selected_epitope,
                epitope_pos_score,
            ])

        if len(rows_buffer) >= chunk_size:
            write_header, written = flush_buffer(rows_buffer, output_path, write_header)
            total_output_rows += written
            rows_buffer = []

        if group_id % 100000 == 0:
            print(f"Processed groups: {group_id}")

    write_header, written = flush_buffer(rows_buffer, output_path, write_header)
    total_output_rows += written

    print(f"Saved: {output_path}")
    print(f"Output rows: {total_output_rows}")
    print(f"Groups: {total_groups}")


if __name__ == "__main__":
    main()