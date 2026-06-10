#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


# ======================================================
# TENSORFLOW SETTINGS
# ======================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


# ======================================================
# CONFIGURATION
# ======================================================

MODELS_FOLDER = Path("/home/nap/lperez_nn/model")
MODEL_PATH = MODELS_FOLDER / "unet_c_ensemble"

SS_LIST = ["C", "H", "E"]

FASTA_RESIDUE_LIST = [
    "A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

VALID_RESIDUES = set(FASTA_RESIDUE_LIST)

NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

WINDOW_SIZE = 1024
OVERLAP = 200
STEP = WINDOW_SIZE - OVERLAP


# ======================================================
# LOGGING
# ======================================================

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


# ======================================================
# MODEL
# ======================================================

def load_protein_unet_model(model_path: Path):
    """Load the trained ProteinUnet model."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logging.info("Loading ProteinUnet model from: %s", model_path)

    return keras.models.load_model(model_path)


# ======================================================
# FASTA / SEQUENCE HELPERS
# ======================================================

def fasta_to_sequence(fasta_text: str) -> str:
    """Convert FASTA text into a plain amino acid sequence."""

    if pd.isna(fasta_text) or not fasta_text:
        return ""

    lines = str(fasta_text).splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))

    return seq.strip().upper()


def get_invalid_residues(seq: str) -> List[str]:
    """Return non-standard amino acid residues found in the sequence."""

    return sorted(set(residue for residue in seq if residue not in VALID_RESIDUES))


def fill_array_with_value(array: np.ndarray, length_limit: int, value: float) -> np.ndarray:
    """Pad an array up to length_limit using a constant value."""

    array_length = len(array)

    filler = value * np.ones(
        (length_limit - array_length, array.shape[1]),
        dtype=array.dtype,
    )

    return np.concatenate((array, filler))


# ======================================================
# SECONDARY STRUCTURE PREDICTION
# ======================================================

def predict_window_probabilities(seq: str, model) -> np.ndarray:
    """
    Predict secondary structure probabilities for one sequence window.

    The input sequence must be shorter than or equal to WINDOW_SIZE.
    """

    seq = str(seq).strip().upper()

    if not seq:
        return np.empty((0, len(SS_LIST)), dtype=np.float32)

    invalid_residues = get_invalid_residues(seq)
    if invalid_residues:
        raise ValueError(
            f"Sequence contains non-standard residues: {','.join(invalid_residues)}"
        )

    if len(seq) > WINDOW_SIZE:
        raise ValueError(f"Window length cannot exceed {WINDOW_SIZE} aa")

    sequence = to_categorical(
        [RESIDUE_DICT[residue] for residue in seq],
        num_classes=NB_RESIDUES,
    )

    sequence = fill_array_with_value(sequence, WINDOW_SIZE, 0)

    pred_c = model.predict(np.array([sequence]), verbose=0)

    if hasattr(pred_c, "numpy"):
        pred_c = pred_c.numpy()

    if isinstance(pred_c, list):
        pred_c = pred_c[0]

    pred_c = np.array(pred_c)

    if pred_c.ndim == 3:
        pred_c = pred_c[0]

    return pred_c[:len(seq)]


def probabilities_to_ss(prob_matrix: np.ndarray) -> str:
    """Convert probability matrix into C/H/E secondary structure labels."""

    if prob_matrix.size == 0:
        return ""

    indices = np.argmax(prob_matrix, axis=-1)

    return "".join(SS_LIST[int(idx)] for idx in indices)


def predict_secondary_structure(seq: str, model) -> str:
    """
    Predict secondary structure for a full protein sequence.

    Short sequences are predicted directly. Long sequences are split into
    overlapping windows and recombined using averaged probabilities.
    """

    seq = str(seq).strip().upper()

    if not seq:
        return ""

    invalid_residues = get_invalid_residues(seq)
    if invalid_residues:
        raise ValueError(
            f"Sequence contains non-standard residues: {','.join(invalid_residues)}"
        )

    seq_len = len(seq)

    if seq_len <= WINDOW_SIZE:
        prob_matrix = predict_window_probabilities(seq, model)
        return probabilities_to_ss(prob_matrix)

    sum_probs = np.zeros((seq_len, len(SS_LIST)), dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)

    starts = list(range(0, seq_len, STEP))

    if starts[-1] + WINDOW_SIZE < seq_len:
        starts.append(seq_len - WINDOW_SIZE)

    starts = sorted(set(min(start, seq_len - WINDOW_SIZE) for start in starts))

    for start in starts:
        end = min(start + WINDOW_SIZE, seq_len)
        window_seq = seq[start:end]

        window_probs = predict_window_probabilities(window_seq, model)
        win_len = len(window_seq)

        # Trim overlapping borders except for the first and last windows.
        left_trim = 0 if start == 0 else OVERLAP // 2
        right_trim = 0 if end == seq_len else OVERLAP // 2

        usable_start = left_trim
        usable_end = win_len - right_trim

        global_start = start + usable_start
        global_end = start + usable_end

        sum_probs[global_start:global_end] += window_probs[usable_start:usable_end]
        counts[global_start:global_end] += 1.0

    uncovered = np.where(counts == 0)[0]

    if len(uncovered) > 0:
        # Fallback: if any residue was not covered after trimming, use full windows.
        for start in starts:
            end = min(start + WINDOW_SIZE, seq_len)
            window_seq = seq[start:end]
            window_probs = predict_window_probabilities(window_seq, model)

            sum_probs[start:end] += window_probs
            counts[start:end] += 1.0

    avg_probs = sum_probs / counts[:, None]

    return probabilities_to_ss(avg_probs)


# ======================================================
# INPUT / OUTPUT
# ======================================================

def load_input_csv(input_path: Path) -> pd.DataFrame:
    """Load input CSV and extract protein sequences from FASTA."""

    df = pd.read_csv(input_path)

    required_cols = ["protein_url", "fasta"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input file must contain column: {col}")

    df["protein_url"] = df["protein_url"].astype(str).str.strip()
    df["sequence"] = df["fasta"].apply(fasta_to_sequence)

    df = df[(df["protein_url"] != "") & (df["sequence"] != "")].copy()

    return df


def save_secondary_structure_csv(results: List[dict], output_path: Path) -> None:
    """Save secondary structure predictions to CSV."""

    out_df = pd.DataFrame(
        results,
        columns=[
            "protein_url",
            "sequence",
            "secondary_structure",
        ],
    )

    out_df.to_csv(output_path, index=False)


# ======================================================
# MAIN PROCESS
# ======================================================

def process_proteins(input_path: Path, output_path: Path, model) -> None:
    """Predict secondary structure for all proteins in the input CSV."""

    df = load_input_csv(input_path)

    total = len(df)

    logging.info("Total proteins to process: %d", total)

    results = []

    n_long = 0
    n_skipped_invalid = 0
    n_ok = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        protein_url = row.protein_url
        sequence = row.sequence

        invalid_residues = get_invalid_residues(sequence)

        if invalid_residues:
            n_skipped_invalid += 1
            logging.warning(
                "Skipping %s because of invalid residues: %s",
                protein_url,
                ",".join(invalid_residues),
            )
            continue

        if len(sequence) > WINDOW_SIZE:
            n_long += 1

        try:
            secondary_structure = predict_secondary_structure(sequence, model)

        except Exception as exc:
            logging.warning("Failed prediction for %s: %s", protein_url, exc)
            continue

        if not secondary_structure:
            logging.warning("Empty secondary structure output for %s", protein_url)
            continue

        if len(secondary_structure) != len(sequence):
            logging.warning(
                "Length mismatch for %s: sequence=%d ss=%d",
                protein_url,
                len(sequence),
                len(secondary_structure),
            )
            continue

        n_ok += 1

        results.append({
            "protein_url": protein_url,
            "sequence": sequence,
            "secondary_structure": secondary_structure,
        })

        if i % 100 == 0 or i == total:
            logging.info("Processed %d/%d", i, total)

    save_secondary_structure_csv(results, output_path)

    logging.info("Proteins longer than %d: %d", WINDOW_SIZE, n_long)
    logging.info("Proteins skipped due to invalid residues: %d", n_skipped_invalid)
    logging.info("Proteins successfully predicted: %d", n_ok)
    logging.info("Done secondary structure prediction -> %s", output_path)


# ======================================================
# CLI
# ======================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Predict protein secondary structure using ProteinUnet."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input CSV file with protein_url and fasta columns.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model = load_protein_unet_model(MODEL_PATH)

    process_proteins(
        input_path=input_path,
        output_path=output_path,
        model=model,
    )


if __name__ == "__main__":
    main()