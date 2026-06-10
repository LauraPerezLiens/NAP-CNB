#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the secondary-structure MLM dataset for BERT pretraining.

This script merges ProteinUnet secondary-structure prediction outputs,
cleans invalid rows, converts C/H/E strings into space-separated token
sequences, and splits the dataset into train, validation and test parquet
files.

Input:
    proteinunet_prediction/outputs/chunk_XXXXX.csv

Outputs:
    bert_mlm_dataset/all_cleaned.parquet
    bert_mlm_dataset/train.parquet
    bert_mlm_dataset/val.parquet
    bert_mlm_dataset/test.parquet
    bert_mlm_dataset/vocab.txt
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Configuration
# ============================================================

VALID_SS = set("CHE")

VOCAB_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "C",
    "H",
    "E",
]


# ============================================================
# Logging
# ============================================================

def setup_logging() -> None:
    """Configure logging format and verbosity level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# Arguments
# ============================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build BERT MLM dataset from ProteinUnet secondary-structure predictions."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nap/lperez_nn/data/data_uniref50"),
        help="Base UniRef data directory.",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing ProteinUnet prediction CSV files. "
            "Default: <data-dir>/proteinunet_prediction/outputs"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for BERT MLM parquet files. "
            "Default: <data-dir>/bert_mlm_dataset"
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.10,
        help="Fraction of the full dataset reserved for the test split.",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.10,
        help="Fraction of the full dataset reserved for the validation split.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for train/validation/test splitting.",
    )

    return parser.parse_args()


# ============================================================
# Validation helpers
# ============================================================

def validate_split_sizes(test_size: float, val_size: float) -> None:
    """Validate train/validation/test split proportions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be lower than 1")


def validate_secondary_structure(ss: str) -> bool:
    """Return True if a secondary-structure string only contains C/H/E."""
    if not isinstance(ss, str):
        return False

    ss = ss.strip()

    if not ss:
        return False

    return set(ss).issubset(VALID_SS)


# ============================================================
# Processing helpers
# ============================================================

def space_secondary_structure(ss: str) -> str:
    """Convert a C/H/E string into a space-separated token sequence."""
    return " ".join(ss.strip())


def load_prediction_outputs(input_dir: Path) -> pd.DataFrame:
    """Load and merge all ProteinUnet prediction CSV files."""
    csv_files = sorted(input_dir.glob("chunk_*.csv"))

    csv_files = [
        path for path in csv_files
        if not path.name.endswith("_skipped.csv")
    ]

    if not csv_files:
        raise FileNotFoundError(f"No prediction CSV files found in {input_dir}")

    logging.info("Found %d prediction CSV files", len(csv_files))

    dataframes = []

    for csv_file in csv_files:
        logging.info("Loading %s", csv_file.name)

        df = pd.read_csv(csv_file)

        required_columns = {"protein_id", "sequence", "secondary_structure"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {csv_file}: {sorted(missing_columns)}"
            )

        df["source_file"] = csv_file.name
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def clean_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ProteinUnet prediction rows before BERT MLM dataset generation."""
    df = df.copy()

    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    df["secondary_structure"] = (
        df["secondary_structure"].astype(str).str.strip().str.upper()
    )

    df["sequence_length"] = df["sequence"].str.len()
    df["secondary_structure_length"] = df["secondary_structure"].str.len()

    before_cleaning = len(df)

    df = df[
        df["secondary_structure"].apply(validate_secondary_structure)
        & (df["sequence_length"] == df["secondary_structure_length"])
    ].copy()

    after_cleaning = len(df)

    logging.info("Rows removed during cleaning: %d", before_cleaning - after_cleaning)
    logging.info("Rows retained after cleaning: %d", after_cleaning)

    df["secondary_structure_spaced"] = df["secondary_structure"].apply(
        space_secondary_structure
    )

    return df[
        [
            "protein_id",
            "sequence",
            "secondary_structure",
            "secondary_structure_spaced",
            "sequence_length",
            "source_file",
        ]
    ]


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation and test sets.

    test_size and val_size are interpreted as fractions of the full dataset.
    For example, test_size=0.10 and val_size=0.10 produce an 80/10/10 split.
    """
    validate_split_sizes(test_size, val_size)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=True,
    )

    return train_df, val_df, test_df


def write_vocab(vocab_path: Path) -> None:
    """Write BERT vocabulary file for secondary-structure tokens."""
    with vocab_path.open("w") as handle:
        for token in VOCAB_TOKENS:
            handle.write(f"{token}\n")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Build train, validation and test parquet files for secondary BERT MLM."""
    setup_logging()
    args = parse_args()

    data_dir = args.data_dir
    input_dir = args.input_dir or data_dir / "proteinunet_prediction" / "outputs"
    output_dir = args.output_dir or data_dir / "bert_mlm_dataset"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    validate_split_sizes(args.test_size, args.val_size)

    logging.info("Starting secondary MLM dataset generation")
    logging.info("Input directory: %s", input_dir)
    logging.info("Output directory: %s", output_dir)
    logging.info("Test size: %.2f of full dataset", args.test_size)
    logging.info("Validation size: %.2f of full dataset", args.val_size)
    logging.info("Random state: %d", args.random_state)

    df_raw = load_prediction_outputs(input_dir)

    logging.info("Raw rows loaded: %d", len(df_raw))

    df = clean_predictions(df_raw)

    all_cleaned_path = output_dir / "all_cleaned.parquet"
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    vocab_path = output_dir / "vocab.txt"

    df.to_parquet(all_cleaned_path, index=False)

    train_df, val_df, test_df = split_dataset(
        df=df,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    write_vocab(vocab_path)

    total_rows = len(df)

    logging.info("Dataset split summary")
    logging.info("Total cleaned rows: %d", total_rows)
    logging.info(
        "Train rows: %d (%.4f%%)",
        len(train_df),
        100 * len(train_df) / total_rows,
    )
    logging.info(
        "Validation rows: %d (%.4f%%)",
        len(val_df),
        100 * len(val_df) / total_rows,
    )
    logging.info(
        "Test rows: %d (%.4f%%)",
        len(test_df),
        100 * len(test_df) / total_rows,
    )
    logging.info("All cleaned dataset written to: %s", all_cleaned_path)
    logging.info("Train dataset written to: %s", train_path)
    logging.info("Validation dataset written to: %s", val_path)
    logging.info("Test dataset written to: %s", test_path)
    logging.info("Vocabulary written to: %s", vocab_path)
    logging.info("Secondary MLM dataset generation completed successfully")


if __name__ == "__main__":
    main()