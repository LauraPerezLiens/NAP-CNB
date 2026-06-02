#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")

SECONDARY_FILES = {
    "human": DATA_INTERMEDIATE / "secondary_struct_human.csv",
    "mouse": DATA_INTERMEDIATE / "secondary_struct_mouse.csv",
}

SPECIES = ["human", "mouse"]
MHC_CLASSES = ["mhc-I", "mhc-II"]
WINDOW_SIZE = 25


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def setup_logging() -> None:
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


def normalize_url(url: str) -> str:
    if pd.isna(url):
        return ""
    return str(url).strip().replace("https://", "http://")


def validate_columns(df: pd.DataFrame, required_cols: set, context: str) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {missing}")


def summarize_status(df: pd.DataFrame) -> str:
    if "ss_status" not in df.columns:
        return "missing ss_status column"

    counts = df["ss_status"].value_counts(dropna=False).to_dict()
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def build_ss_lookup(ss_csv: Path) -> Dict[str, dict]:
    df = pd.read_csv(ss_csv)

    required_cols = {"protein_url", "sequence", "secondary_structure"}
    validate_columns(df, required_cols, str(ss_csv))

    lookup = {}

    for _, row in df.iterrows():
        protein_url = normalize_url(row["protein_url"])

        seq = "" if pd.isna(row["sequence"]) else str(row["sequence"]).strip()
        ss = "" if pd.isna(row["secondary_structure"]) else str(row["secondary_structure"]).strip()

        if not protein_url or not seq or not ss:
            continue

        if len(seq) != len(ss):
            continue

        lookup[protein_url] = {
            "sequence": seq,
            "secondary_structure": ss,
        }

    logging.info("Loaded secondary structure entries from %s: %d", ss_csv, len(lookup))

    return lookup


def add_ss_to_normal_classification(
    class_df: pd.DataFrame,
    ss_lookup: Dict[str, dict],
    window_size: int,
) -> pd.DataFrame:
    required_cols = {
        "protein_group_id",
        "group_id",
        "25aa_seq",
        "protein_url",
        "window_start",
    }
    validate_columns(class_df, required_cols, "normal classification dataframe")

    out = class_df.copy()

    ss_windows = []
    ss_status = []
    protein_found = []
    sequence_match = []

    for _, row in out.iterrows():
        seq_window_expected = "" if pd.isna(row["25aa_seq"]) else str(row["25aa_seq"]).strip()
        protein_url = normalize_url(row["protein_url"])

        try:
            window_start = int(row["window_start"])
        except Exception:
            window_start = None

        if protein_url not in ss_lookup:
            ss_windows.append(None)
            ss_status.append("protein_not_found")
            protein_found.append(0)
            sequence_match.append(None)
            continue

        protein_found.append(1)

        full_seq = ss_lookup[protein_url]["sequence"]
        full_ss = ss_lookup[protein_url]["secondary_structure"]

        if window_start is None or window_start < 1:
            ss_windows.append(None)
            ss_status.append("invalid_window_start")
            sequence_match.append(None)
            continue

        start_idx = window_start - 1
        end_idx = start_idx + window_size

        if end_idx > len(full_seq):
            ss_windows.append(None)
            ss_status.append("window_out_of_range")
            sequence_match.append(None)
            continue

        seq_window_real = full_seq[start_idx:end_idx]
        ss_window = full_ss[start_idx:end_idx]

        if seq_window_real != seq_window_expected:
            ss_windows.append(None)
            ss_status.append("sequence_mismatch")
            sequence_match.append(0)
            continue

        ss_windows.append(ss_window)
        ss_status.append("ok")
        sequence_match.append(1)

    out["secondary_structure"] = ss_windows
    out["protein_found"] = protein_found
    out["sequence_match"] = sequence_match
    out["ss_status"] = ss_status

    return out


def propagate_ss_to_blosum(
    blosum_df: pd.DataFrame,
    normal_df: pd.DataFrame,
) -> pd.DataFrame:
    required_blosum = {
        "protein_group_id",
        "group_id",
    }
    validate_columns(blosum_df, required_blosum, "BLOSUM dataframe")

    required_normal = {
        "protein_group_id",
        "group_id",
        "secondary_structure",
        "protein_found",
        "sequence_match",
        "ss_status",
    }
    validate_columns(normal_df, required_normal, "normal dataframe with secondary structure")

    mapping = normal_df[
        [
            "protein_group_id",
            "group_id",
            "secondary_structure",
            "protein_found",
            "sequence_match",
            "ss_status",
        ]
    ].drop_duplicates(subset=["protein_group_id", "group_id"])

    out = blosum_df.merge(
        mapping,
        on=["protein_group_id", "group_id"],
        how="left",
    )

    return out


def process_one_haplotype(
    species: str,
    mhc_class: str,
    haplotype_dir: Path,
    ss_lookup: Dict[str, dict],
) -> None:
    normal_csvs = sorted(
        path
        for path in haplotype_dir.glob("classification_*.csv")
        if "_blosum" not in path.stem and not path.stem.endswith("_ss")
    )

    if not normal_csvs:
        logging.info("Skipping %s: no normal classification file found", haplotype_dir)
        return

    for normal_csv in normal_csvs:
        logging.info("=" * 60)
        logging.info("Processing: %s", normal_csv)
        logging.info("=" * 60)

        try:
            class_df = pd.read_csv(normal_csv, low_memory=False)

            out_normal = add_ss_to_normal_classification(
                class_df=class_df,
                ss_lookup=ss_lookup,
                window_size=WINDOW_SIZE,
            )

            out_normal_csv = normal_csv.with_name(normal_csv.stem + "_ss.csv")
            out_normal.to_csv(out_normal_csv, index=False)

            logging.info("Normal saved: %s", out_normal_csv)
            logging.info("Status: %s", summarize_status(out_normal))

            blosum_csv = normal_csv.with_name(normal_csv.stem + "_blosum.csv")

            if blosum_csv.exists():
                blosum_df = pd.read_csv(blosum_csv, low_memory=False)

                out_blosum = propagate_ss_to_blosum(
                    blosum_df=blosum_df,
                    normal_df=out_normal,
                )

                out_blosum_csv = blosum_csv.with_name(blosum_csv.stem + "_ss.csv")
                out_blosum.to_csv(out_blosum_csv, index=False)

                rows_with_ss = out_blosum["secondary_structure"].notna().sum()

                logging.info("BLOSUM saved: %s", out_blosum_csv)
                logging.info(
                    "Rows with secondary structure: %d/%d",
                    rows_with_ss,
                    len(out_blosum),
                )
                logging.info("BLOSUM status: %s", summarize_status(out_blosum))

            else:
                logging.info("No BLOSUM file found for: %s", normal_csv.name)

        except Exception as exc:
            logging.error("Failed processing %s: %s", normal_csv, exc)


def main() -> None:
    setup_logging()

    ss_lookups = {}

    for species in SPECIES:
        ss_csv = SECONDARY_FILES[species]

        if not ss_csv.exists():
            raise FileNotFoundError(f"Missing secondary structure file: {ss_csv}")

        logging.info("Loading secondary structure for %s: %s", species, ss_csv)
        ss_lookups[species] = build_ss_lookup(ss_csv)

    for species in SPECIES:
        for mhc_class in MHC_CLASSES:
            class_dir = DATA_INTERMEDIATE / species / mhc_class

            if not class_dir.exists():
                logging.info("Skipping missing directory: %s", class_dir)
                continue

            haplotype_dirs = sorted(path for path in class_dir.iterdir() if path.is_dir())

            if not haplotype_dirs:
                logging.info("Skipping %s: no haplotype directories found", class_dir)
                continue

            for haplotype_dir in haplotype_dirs:
                logging.info("#" * 60)
                logging.info(
                    "Species: %s | Class: %s | Haplotype: %s",
                    species,
                    mhc_class,
                    haplotype_dir.name,
                )
                logging.info("#" * 60)

                process_one_haplotype(
                    species=species,
                    mhc_class=mhc_class,
                    haplotype_dir=haplotype_dir,
                    ss_lookup=ss_lookups[species],
                )


if __name__ == "__main__":
    main()