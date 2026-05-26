#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# =========================
# CONFIGURATION
# =========================

DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")
DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")

SPECIES = ["human", "mouse"]

WINDOW_SIZE = 25
PROGRESS_EVERY = 100
OUTPUT_FLOAT_FORMAT = "%.3f"


# =========================
# LOGGING
# =========================

class MaxLevelFilter(logging.Filter):
    """Filter log records up to a maximum logging level."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def setup_logging() -> None:
    """Configure logging: INFO goes to stdout, WARNING/ERROR goes to stderr."""

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)


# =========================
# FASTA PROCESSING
# =========================

def fasta_to_sequence(fasta_text: str) -> str:
    """Convert FASTA text into a plain amino acid sequence."""

    if pd.isna(fasta_text) or not fasta_text:
        return ""

    lines = str(fasta_text).splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))

    return seq.strip().upper()


def build_protein_sequence_map(species: str) -> Dict[str, str]:
    """Load parent protein FASTA sequences and build a protein_url -> sequence map."""

    fasta_csv = DATA_INTERMEDIATE / f"{species}_parent_protein_fasta.csv"

    if not fasta_csv.exists():
        raise FileNotFoundError(f"Missing FASTA CSV file: {fasta_csv}")

    df = pd.read_csv(fasta_csv)

    required_cols = ["protein_url", "fasta"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{fasta_csv} must contain column: {col}")

    df["sequence"] = df["fasta"].apply(fasta_to_sequence)
    df["protein_url"] = df["protein_url"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip()

    df = df[(df["protein_url"] != "") & (df["sequence"] != "")].copy()

    logging.info("[%s] loaded protein sequences: %d", species, len(df))

    return {
        row.protein_url: row.sequence
        for row in df.itertuples()
    }


# =========================
# EPITOPE SCORING
# =========================

def compute_epitope_position_score(
    window_start: int,
    window_end: int,
    ep_start: int,
    ep_end: int,
    window_size: int,
) -> float:
    """
    Compute the relative position of the epitope inside the window.

    Interpretation:
        0.0  -> epitope center is aligned with the window center
        <0.0 -> epitope is shifted towards the left side of the window
        >0.0 -> epitope is shifted towards the right side of the window

    The score is clipped to the range [-1, 1].
    """

    window_center = math.ceil(window_size / 2)
    max_center_dist = window_center - 1

    ep_center = (ep_start + ep_end) / 2.0
    ep_center_in_window = ep_center - window_start + 1
    signed_dist = ep_center_in_window - window_center

    if max_center_dist == 0:
        return 0.0

    score = min(1.0, abs(signed_dist) / max_center_dist)
    signed_score = math.copysign(score, signed_dist)

    return round(signed_score, 3)


def validate_exact_match(
    protein_seq: str,
    epitope_seq: str,
    ep_start: int,
    ep_end: int,
) -> bool:
    """
    Validate that the epitope sequence exactly matches the protein sequence
    at the annotated start/end coordinates.
    """

    if not protein_seq or not epitope_seq:
        return False

    if ep_start < 1 or ep_end < 1 or ep_start > ep_end or ep_end > len(protein_seq):
        return False

    observed = protein_seq[ep_start - 1:ep_end]

    return observed == epitope_seq


# =========================
# CLASSIFICATION DATASET
# =========================

def classify_one_haplotype(
    species: str,
    class_type: str,
    haplotype: str,
    seq_map: Dict[str, str],
) -> None:
    """Generate 25-aa classification windows for one species/class/haplotype."""

    window_size = WINDOW_SIZE

    epitope_file = DATA_RAW / species / class_type / haplotype / "merged_unique_events.csv"

    if not epitope_file.exists():
        logging.warning("Missing epitope file: %s", epitope_file)
        return

    logging.info("Processing %s", epitope_file)

    epitopes_df = pd.read_csv(epitope_file)

    required_cols = [
        "epitope__name_canonical",
        "epitope__starting_position",
        "epitope__ending_position",
        "parent_id",
    ]

    for col in required_cols:
        if col not in epitopes_df.columns:
            raise ValueError(f"Missing column {col} in {epitope_file}")

    initial_epitopes = len(epitopes_df)

    epitopes_df = epitopes_df.dropna(
        subset=[
            "epitope__name_canonical",
            "epitope__starting_position",
            "epitope__ending_position",
            "parent_id",
        ]
    ).copy()

    after_required_filter = len(epitopes_df)

    epitopes_df["parent_id"] = epitopes_df["parent_id"].astype(str).str.strip()

    epitopes_df["epitope__starting_position"] = pd.to_numeric(
        epitopes_df["epitope__starting_position"], errors="coerce"
    )
    epitopes_df["epitope__ending_position"] = pd.to_numeric(
        epitopes_df["epitope__ending_position"], errors="coerce"
    )

    epitopes_df["epitope__name_canonical"] = (
        epitopes_df["epitope__name_canonical"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    epitopes_df = epitopes_df.dropna(
        subset=["epitope__starting_position", "epitope__ending_position"]
    ).copy()

    epitopes_df["epitope__starting_position"] = (
        epitopes_df["epitope__starting_position"].astype(int)
    )
    epitopes_df["epitope__ending_position"] = (
        epitopes_df["epitope__ending_position"].astype(int)
    )

    epitopes_df = epitopes_df[
        (epitopes_df["parent_id"] != "")
        & (epitopes_df["epitope__name_canonical"] != "")
    ].copy()

    after_cleaning = len(epitopes_df)

    protein_urls = sorted(set(epitopes_df["parent_id"]))
    rows: List[List] = []

    total_proteins = len(protein_urls)
    proteins_missing_fasta = 0
    proteins_too_short = 0
    proteins_without_exact_match = 0
    proteins_used = 0

    total_epitopes_before_exact_match = 0
    total_epitopes_after_exact_match = 0

    logging.info(
        "[%s %s %s] proteins with epitopes: %d",
        species,
        class_type,
        haplotype,
        total_proteins,
    )

    for idx, protein_url in enumerate(protein_urls, start=1):
        seq = seq_map.get(protein_url, "")

        if not seq:
            proteins_missing_fasta += 1
            continue

        if len(seq) < window_size:
            proteins_too_short += 1
            continue

        prot_epitopes = epitopes_df[epitopes_df["parent_id"] == protein_url].copy()

        if prot_epitopes.empty:
            continue

        total_epitopes_before_exact_match += len(prot_epitopes)

        # Keep only epitopes that exactly match the protein sequence coordinates.
        prot_epitopes["exact_match"] = prot_epitopes.apply(
            lambda r: validate_exact_match(
                protein_seq=seq,
                epitope_seq=r["epitope__name_canonical"],
                ep_start=r["epitope__starting_position"],
                ep_end=r["epitope__ending_position"],
            ),
            axis=1,
        )

        prot_epitopes = prot_epitopes[prot_epitopes["exact_match"]].copy()

        total_epitopes_after_exact_match += len(prot_epitopes)

        if prot_epitopes.empty:
            proteins_without_exact_match += 1
            continue

        proteins_used += 1

        if idx % PROGRESS_EVERY == 0 or idx == 1:
            logging.info(
                "[%s %s %s] protein %d/%d",
                species,
                class_type,
                haplotype,
                idx,
                total_proteins,
            )

        # Generate all possible sliding windows of fixed length.
        for i in range(len(seq) - window_size + 1):
            window_seq = seq[i:i + window_size]
            window_start = i + 1
            window_end = i + window_size

            positive_hits: List[Tuple[str, float]] = []

            for e in prot_epitopes.itertuples():
                ep_start = e.epitope__starting_position
                ep_end = e.epitope__ending_position
                ep_name = e.epitope__name_canonical

                # A window is positive if it fully contains the epitope.
                if window_start <= ep_start and window_end >= ep_end:
                    pos_score = compute_epitope_position_score(
                        window_start=window_start,
                        window_end=window_end,
                        ep_start=ep_start,
                        ep_end=ep_end,
                        window_size=window_size,
                    )
                    positive_hits.append((ep_name, pos_score))

            if positive_hits:
                for ep_name, pos_score in positive_hits:
                    rows.append([
                        window_seq,
                        1,
                        ep_name,
                        pos_score,
                        protein_url,
                        window_start,
                    ])
            else:
                rows.append([
                    window_seq,
                    0,
                    "",
                    0.0,
                    protein_url,
                    window_start,
                ])

    output_cols = [
        "25aa_seq",
        "contains_epitope",
        "selected_epitope",
        "epitope_pos_score",
        "protein_url",
        "window_start",
    ]

    output_df = pd.DataFrame(rows, columns=output_cols)
    before_dedup = len(output_df)

    output_df = output_df.drop_duplicates().reset_index(drop=True)

    after_dedup = len(output_df)

    out_dir = DATA_INTERMEDIATE / species / class_type / haplotype
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"classification_{species}_{class_type}_{haplotype}.csv"

    if output_df.empty:
        logging.warning(
            "[%s %s %s] output dataset is empty: %s",
            species,
            class_type,
            haplotype,
            out_file,
        )

    output_df.to_csv(out_file, index=False, float_format=OUTPUT_FLOAT_FORMAT)

    positive_windows = int(output_df["contains_epitope"].sum()) if not output_df.empty else 0
    total_windows = len(output_df)
    negative_windows = total_windows - positive_windows

    discarded_missing_required = initial_epitopes - after_required_filter
    discarded_after_cleaning = after_required_filter - after_cleaning
    discarded_exact_match = total_epitopes_before_exact_match - total_epitopes_after_exact_match
    duplicated_rows_removed = before_dedup - after_dedup

    logging.info("[%s %s %s] saved: %s", species, class_type, haplotype, out_file)
    logging.info("[%s %s %s] total windows: %d", species, class_type, haplotype, total_windows)
    logging.info("[%s %s %s] positive windows: %d", species, class_type, haplotype, positive_windows)
    logging.info("[%s %s %s] negative windows: %d", species, class_type, haplotype, negative_windows)

    logging.info("[%s %s %s] proteins total: %d", species, class_type, haplotype, total_proteins)
    logging.info("[%s %s %s] proteins used: %d", species, class_type, haplotype, proteins_used)
    logging.info("[%s %s %s] proteins missing FASTA: %d", species, class_type, haplotype, proteins_missing_fasta)
    logging.info("[%s %s %s] proteins shorter than window: %d", species, class_type, haplotype, proteins_too_short)
    logging.info("[%s %s %s] proteins without exact epitope match: %d", species, class_type, haplotype, proteins_without_exact_match)

    logging.info("[%s %s %s] initial epitopes: %d", species, class_type, haplotype, initial_epitopes)
    logging.info("[%s %s %s] discarded missing required values: %d", species, class_type, haplotype, discarded_missing_required)
    logging.info("[%s %s %s] discarded during cleaning: %d", species, class_type, haplotype, discarded_after_cleaning)
    logging.info("[%s %s %s] epitopes before exact-match filter: %d", species, class_type, haplotype, total_epitopes_before_exact_match)
    logging.info("[%s %s %s] epitopes after exact-match filter: %d", species, class_type, haplotype, total_epitopes_after_exact_match)
    logging.info("[%s %s %s] discarded by exact-match filter: %d", species, class_type, haplotype, discarded_exact_match)
    logging.info("[%s %s %s] duplicated rows removed: %d", species, class_type, haplotype, duplicated_rows_removed)


# =========================
# MAIN
# =========================

def main() -> None:
    setup_logging()

    for species in SPECIES:
        logging.info("Loading sequences for %s", species)

        seq_map = build_protein_sequence_map(species)

        species_dir = DATA_RAW / species

        if not species_dir.exists():
            logging.warning("Missing species directory: %s", species_dir)
            continue

        for class_dir in sorted(species_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_type = class_dir.name

            for haplo_dir in sorted(class_dir.iterdir()):
                if not haplo_dir.is_dir():
                    continue

                haplotype = haplo_dir.name

                classify_one_haplotype(
                    species=species,
                    class_type=class_type,
                    haplotype=haplotype,
                    seq_map=seq_map,
                )

    logging.info("DONE")


if __name__ == "__main__":
    main()