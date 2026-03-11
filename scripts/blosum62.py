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
        return -4


BEST_CHANGE = {}
for aa in amino_acids:
    candidates = [x for x in amino_acids if x != aa]
    BEST_CHANGE[aa] = max(candidates, key=lambda x: get_blosum_score(aa, x))


def most_probable_change(aa):
    return BEST_CHANGE.get(aa, aa)


def build_variants(seq):
    variants = [seq]
    seq_list = list(seq)
    best_changes = [most_probable_change(aa) for aa in seq_list]
    for i, change in enumerate(best_changes):
        mutated = seq_list.copy()
        mutated[i] = change
        variants.append("".join(mutated))
    return variants


def main():
    input_path = sys.argv[1]

    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_blosum{ext}"

    df = pd.read_csv(input_path)
    required_cols = ["original_seq", "blosum_seq", "contains_epitope", "epitope_pos_score"]
    df = df[["25aa_seq", "contains_epitope", "epitope_pos_score"]]
    df = df.drop_duplicates()

    print(f"Input rows: {len(df)}")
    print(f"Unique 25aa_seq: {df['25aa_seq'].nunique()}")

    # Escribir header una sola vez
    first_write = True

    for _, row in df.iterrows():
        seq = row["25aa_seq"]
        contains = int(row["contains_epitope"])
        dist = row["epitope_pos_score"]

        variants = build_variants(seq)

        # Primera línea: secuencia original dos veces
        rows = [[seq, seq, contains, dist]]
        # Resto: secuencia original, variante blosum, epítopo, score
        rows += [[seq, v, contains, dist] for v in variants[1:]]

        out_chunk = pd.DataFrame(
            rows,
            columns=required_cols
        )

        out_chunk.to_csv(
            output_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            float_format="%.3f"
        )

        first_write = False

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
