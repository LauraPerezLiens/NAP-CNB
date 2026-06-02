#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import sys
from typing import List, Tuple

import pandas as pd
from Bio.Align import substitution_matrices


# =========================
# CONFIGURATION
# =========================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
CHUNK_SIZE = 100_000
OUTPUT_FLOAT_FORMAT = "%.3f"

OUTPUT_COLUMNS = [
    "protein_group_id",
    "group_id",
    "original_seq",
    "blosum_seq",
    "contains_epitope",
    "selected_epitope",
    "epitope_pos_score",
]


# =========================
# LOGGING
# =========================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


# =========================
# BLOSUM62 SETUP
# =========================

BLOSUM62 = substitution_matrices.load("BLOSUM62")


def get_blosum_score(a1: str, a2: str) -> float:
    """Return the BLOSUM62 substitution score for two amino acids."""

    try:
        return float(BLOSUM62[a1, a2])
    except Exception:
        return -4.0


BEST_CHANGE = {}

for aa in AMINO_ACIDS:
    candidates = [candidate for candidate in AMINO_ACIDS if candidate != aa]
    BEST_CHANGE[aa] = max(candidates, key=lambda x: get_blosum_score(aa, x))


# =========================
# VARIANT GENERATION
# =========================

def most_probable_change(aa: str) -> str:
    """Return the most conservative BLOSUM62-based substitution."""

    return BEST_CHANGE.get(aa, aa)


def build_variants(seq: str) -> List[str]:
    """
    Generate BLOSUM62-based single-substitution variants.

    Returns:
        - Original sequence
        - One mutated sequence per position
    """

    variants = [seq]
    seq_list = list(seq)

    for i, aa in enumerate(seq_list):
        change = most_probable_change(aa)
        mutated = seq_list.copy()
        mutated[i] = change
        variants.append("".join(mutated))

    return variants


def is_standard_protein_sequence(seq: str) -> bool:
    """Check whether a sequence only contains standard amino acids."""

    return all(aa in AMINO_ACIDS for aa in seq)


# =========================
# INPUT / OUTPUT HELPERS
# =========================

def detect_seq_column(df: pd.DataFrame) -> str:
    """Detect the input sequence column."""

    if "25aa_seq" in df.columns:
        return "25aa_seq"

    raise ValueError("Input file must contain a '25aa_seq' column.")


def flush_buffer(
    rows_buffer: List[List],
    output_path: str,
    write_header: bool,
) -> Tuple[bool, int]:
    """Write buffered rows to disk."""

    if not rows_buffer:
        return write_header, 0

    chunk_df = pd.DataFrame(rows_buffer, columns=OUTPUT_COLUMNS)

    chunk_df.to_csv(
        output_path,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
        float_format=OUTPUT_FLOAT_FORMAT,
    )

    return False, len(chunk_df)


# =========================
# MAIN
# =========================

def main() -> None:
    setup_logging()

    if len(sys.argv) < 2:
        logging.error("Usage: python generate_blosum62_variants.py input.csv [output.csv]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_blosum{ext}"

    df = pd.read_csv(input_path, low_memory=False)

    seq_col = detect_seq_column(df)

    required_cols = [
        "protein_group_id",
        "group_id",
        seq_col,
        "contains_epitope",
        "selected_epitope",
        "epitope_pos_score",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    df = df[required_cols].reset_index(drop=True)

    logging.info("Input file: %s", input_path)
    logging.info("Output file: %s", output_path)
    logging.info("Input rows: %d", len(df))
    logging.info("Unique proteins: %d", df["protein_group_id"].nunique())
    logging.info("Unique groups/windows: %d", df["group_id"].nunique())
    logging.info("Unique %s: %d", seq_col, df[seq_col].nunique())

    rows_buffer: List[List] = []
    write_header = True

    total_output_rows = 0
    total_groups_generated = 0
    skipped_empty = 0
    skipped_non_standard = 0
    skipped_invalid_id = 0

    for _, row in df.iterrows():
        original_seq = str(row[seq_col]).strip().upper()

        if not original_seq:
            skipped_empty += 1
            continue

        if not is_standard_protein_sequence(original_seq):
            skipped_non_standard += 1
            continue

        try:
            protein_group_id = int(row["protein_group_id"])
            group_id = int(row["group_id"])
        except Exception:
            skipped_invalid_id += 1
            continue

        contains_epitope = int(row["contains_epitope"])
        selected_epitope = row["selected_epitope"]
        epitope_pos_score = row["epitope_pos_score"]

        variants = build_variants(original_seq)
        total_groups_generated += 1

        for blosum_seq in variants:
            rows_buffer.append([
                protein_group_id,
                group_id,
                original_seq,
                blosum_seq,
                contains_epitope,
                selected_epitope,
                epitope_pos_score,
            ])

        if len(rows_buffer) >= CHUNK_SIZE:
            write_header, written = flush_buffer(
                rows_buffer,
                output_path,
                write_header,
            )
            total_output_rows += written
            rows_buffer = []

        if total_groups_generated % CHUNK_SIZE == 0:
            logging.info("Generated groups/windows: %d", total_groups_generated)

    write_header, written = flush_buffer(
        rows_buffer,
        output_path,
        write_header,
    )
    total_output_rows += written

    logging.info("Saved: %s", output_path)
    logging.info("Output rows: %d", total_output_rows)
    logging.info("Groups/windows generated: %d", total_groups_generated)
    logging.info("Skipped empty sequences: %d", skipped_empty)
    logging.info("Skipped non-standard sequences: %d", skipped_non_standard)
    logging.info("Skipped invalid IDs: %d", skipped_invalid_id)


if __name__ == "__main__":
    main()