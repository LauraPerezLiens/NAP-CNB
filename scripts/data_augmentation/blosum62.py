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
    if "12aa_seq" in df.columns:
        return "12aa_seq"
    raise ValueError("No encuentro ni '25aa_seq' ni '12aa_seq' en el input.")


def main():
    if len(sys.argv) < 2:
        print("Uso: python blosum_augmentation.py input.csv [output.csv]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_blosum{ext}"

    df = pd.read_csv(input_path)

    seq_col = detect_seq_column(df)

    required_cols = [seq_col, "contains_epitope", "epitope_pos_score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el input: {missing}")

    df = df[required_cols].drop_duplicates().reset_index(drop=True)

    print(f"Input rows: {len(df)}")
    print(f"Unique {seq_col}: {df[seq_col].nunique()}")

    all_rows = []

    for group_id, row in enumerate(df.itertuples(index=False), start=1):
        original_seq = str(getattr(row, seq_col)).strip().upper()
        contains_epitope = int(row.contains_epitope)
        epitope_pos_score = row.epitope_pos_score

        if not original_seq:
            continue

        variants = build_variants(original_seq)

        for blosum_seq in variants:
            all_rows.append([
                original_seq,
                blosum_seq,
                group_id,
                contains_epitope,
                epitope_pos_score,
            ])

    out_df = pd.DataFrame(
        all_rows,
        columns=[
            "original_seq",
            "blosum_seq",
            "group_id",
            "contains_epitope",
            "epitope_pos_score",
        ],
    )

    out_df.to_csv(
        output_path,
        index=False,
        float_format="%.3f"
    )

    print(f"Saved: {output_path}")
    print(f"Output rows: {len(out_df)}")
    print(f"Groups: {out_df['group_id'].nunique()}")


if __name__ == "__main__":
    main()