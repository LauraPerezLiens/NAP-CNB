#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict secondary structure for UniRef30 representative sequences.

This script applies a trained ProteinUnet model to FASTA chunks generated
from UniRef50 representative sequences clustered at 30% identity.

Although the original database is UniRef50, the working dataset is referred
to as UniRef30 because sequences were previously clustered with MMseqs2 at
30% sequence identity.

The ProteinUnet model expects sequences containing only the 20 standard
amino acids. Sequences with non-standard residues such as X, U, B, Z, J or O
are skipped and reported separately.

Input:
    proteinunet_prediction/chunks/chunk_XXXXX.fasta

Outputs:
    proteinunet_prediction/outputs/chunk_XXXXX.csv
    proteinunet_prediction/outputs/chunk_XXXXX_skipped.csv
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Generator, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


SS_LIST = ["C", "H", "E"]

STANDARD_RESIDUES = [
    "A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",
]

VALID_RESIDUES = set(STANDARD_RESIDUES)
RESIDUE_DICT = dict(zip(STANDARD_RESIDUES, range(len(STANDARD_RESIDUES))))

WINDOW_SIZE = 1024
OVERLAP = 200
STEP = WINDOW_SIZE - OVERLAP


def setup_logging() -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict secondary structure for UniRef30 FASTA chunks."
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input FASTA chunk file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output CSV filename. Default: input FASTA basename with .csv extension."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nap/lperez_nn/data/data_uniref50"),
        help="Base UniRef data directory.",
    )

    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <data-dir>/proteinunet_prediction/outputs",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/nap/lperez_nn/model/unet_c_ensemble"),
        help="Path to the trained ProteinUnet model.",
    )

    return parser.parse_args()


def parse_fasta(fasta_path: Path) -> Generator[Tuple[str, str], None, None]:
    """Parse a FASTA file and yield header and sequence."""
    header = None
    seq_lines = []

    with fasta_path.open("r") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines).upper()

                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            yield header, "".join(seq_lines).upper()


def get_invalid_residues(sequence: str) -> list[str]:
    """Return sorted non-standard residues found in a sequence."""
    return sorted(set(res for res in sequence if res not in VALID_RESIDUES))


def pad_sequence_array(
    array: np.ndarray,
    target_length: int,
    value: float = 0.0,
) -> np.ndarray:
    """Pad a one-hot encoded sequence array to the required model input length."""
    current_length = len(array)

    if current_length > target_length:
        raise ValueError(
            f"Array length {current_length} exceeds target length {target_length}"
        )

    padding = value * np.ones(
        (target_length - current_length, array.shape[1]),
        dtype=array.dtype,
    )

    return np.concatenate((array, padding), axis=0)


def predict_window_probabilities(
    sequence: str,
    model: keras.Model,
) -> np.ndarray:
    """Predict secondary-structure probabilities for one sequence window."""
    sequence = sequence.strip().upper()

    if not sequence:
        return np.empty((0, len(SS_LIST)), dtype=np.float32)

    invalid_residues = get_invalid_residues(sequence)
    if invalid_residues:
        raise ValueError(
            f"Sequence contains non-standard residues: {','.join(invalid_residues)}"
        )

    if len(sequence) > WINDOW_SIZE:
        raise ValueError(f"Window length cannot exceed {WINDOW_SIZE} residues")

    one_hot = to_categorical(
        [RESIDUE_DICT[res] for res in sequence],
        num_classes=len(STANDARD_RESIDUES),
    )

    one_hot = pad_sequence_array(one_hot, WINDOW_SIZE, value=0.0)

    prediction = model.predict(np.array([one_hot]), verbose=0)

    if hasattr(prediction, "numpy"):
        prediction = prediction.numpy()

    if isinstance(prediction, list):
        prediction = prediction[0]

    prediction = np.asarray(prediction)

    if prediction.ndim == 3:
        prediction = prediction[0]

    return prediction[:len(sequence)]


def probabilities_to_secondary_structure(probabilities: np.ndarray) -> str:
    """Convert a probability matrix into a C/H/E secondary-structure string."""
    if probabilities.size == 0:
        return ""

    indices = np.argmax(probabilities, axis=-1)

    return "".join(SS_LIST[int(index)] for index in indices)


def predict_secondary_structure(sequence: str, model: keras.Model) -> str:
    """
    Predict secondary structure for a full protein sequence.

    Sequences longer than WINDOW_SIZE are split into overlapping windows.
    The central region of each overlapping window is used to reduce edge
    effects. If any positions remain uncovered, full windows are used as a
    fallback.
    """
    sequence = sequence.strip().upper()

    if not sequence:
        return ""

    invalid_residues = get_invalid_residues(sequence)
    if invalid_residues:
        raise ValueError(
            f"Sequence contains non-standard residues: {','.join(invalid_residues)}"
        )

    sequence_length = len(sequence)

    if sequence_length <= WINDOW_SIZE:
        probabilities = predict_window_probabilities(sequence, model)
        return probabilities_to_secondary_structure(probabilities)

    sum_probabilities = np.zeros((sequence_length, len(SS_LIST)), dtype=np.float32)
    counts = np.zeros(sequence_length, dtype=np.float32)

    starts = list(range(0, sequence_length, STEP))

    if starts[-1] + WINDOW_SIZE < sequence_length:
        starts.append(sequence_length - WINDOW_SIZE)

    starts = sorted(set(min(start, sequence_length - WINDOW_SIZE) for start in starts))

    for start in starts:
        end = min(start + WINDOW_SIZE, sequence_length)
        window_sequence = sequence[start:end]
        window_probabilities = predict_window_probabilities(window_sequence, model)

        window_length = len(window_sequence)

        left_trim = 0 if start == 0 else OVERLAP // 2
        right_trim = 0 if end == sequence_length else OVERLAP // 2

        usable_start = left_trim
        usable_end = window_length - right_trim

        global_start = start + usable_start
        global_end = start + usable_end

        sum_probabilities[global_start:global_end] += window_probabilities[
            usable_start:usable_end
        ]
        counts[global_start:global_end] += 1.0

    uncovered_positions = np.where(counts == 0)[0]

    if len(uncovered_positions) > 0:
        logging.warning(
            "Detected %d uncovered positions. Recomputing using full windows.",
            len(uncovered_positions),
        )

        for start in starts:
            end = min(start + WINDOW_SIZE, sequence_length)
            window_sequence = sequence[start:end]
            window_probabilities = predict_window_probabilities(window_sequence, model)

            sum_probabilities[start:end] += window_probabilities
            counts[start:end] += 1.0

    if np.any(counts == 0):
        raise RuntimeError("Some sequence positions remain uncovered after prediction")

    average_probabilities = sum_probabilities / counts[:, None]

    return probabilities_to_secondary_structure(average_probabilities)


def main() -> None:
    """Run secondary-structure prediction for one UniRef30 FASTA chunk."""
    setup_logging()
    args = parse_arguments()

    input_fasta = args.input
    outputs_dir = args.outputs_dir or args.data_dir / "proteinunet_prediction" / "outputs"

    if args.output is None:
        output_name = input_fasta.with_suffix(".csv").name
    else:
        output_name = args.output

    output_csv = outputs_dir / output_name
    skipped_csv = outputs_dir / output_name.replace(".csv", "_skipped.csv")

    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"ProteinUnet model not found: {args.model_path}")

    logging.info("Starting UniRef30 secondary-structure prediction")
    logging.info("Input FASTA: %s", input_fasta)
    logging.info("Output CSV: %s", output_csv)
    logging.info("Skipped CSV: %s", skipped_csv)
    logging.info("Model path: %s", args.model_path)
    logging.info("Window size: %d", WINDOW_SIZE)
    logging.info("Overlap: %d", OVERLAP)

    logging.info("Loading ProteinUnet model")
    model = keras.models.load_model(args.model_path)
    logging.info("Model loaded successfully")

    n_total = 0
    n_long = 0
    n_skipped_invalid = 0
    n_failed = 0
    n_ok = 0

    with output_csv.open("w", newline="") as out_handle, skipped_csv.open(
        "w", newline=""
    ) as skipped_handle:

        writer = csv.writer(out_handle)
        skipped_writer = csv.writer(skipped_handle)

        writer.writerow(["id", "protein_id", "sequence", "secondary_structure"])
        skipped_writer.writerow(["protein_id", "reason", "details"])

        for n_total, (header, sequence) in enumerate(parse_fasta(input_fasta), start=1):
            protein_id = header.split()[0]
            invalid_residues = get_invalid_residues(sequence)

            if invalid_residues:
                n_skipped_invalid += 1
                skipped_writer.writerow(
                    [
                        protein_id,
                        "non_standard_residues",
                        ",".join(invalid_residues),
                    ]
                )
                continue

            if len(sequence) > WINDOW_SIZE:
                n_long += 1

            try:
                secondary_structure = predict_secondary_structure(sequence, model)

            except Exception as error:
                n_failed += 1
                logging.warning("Prediction failed for %s: %s", protein_id, error)
                skipped_writer.writerow(
                    [
                        protein_id,
                        "prediction_failed",
                        str(error),
                    ]
                )
                continue

            if not secondary_structure:
                n_failed += 1
                skipped_writer.writerow(
                    [
                        protein_id,
                        "empty_prediction",
                        "Predicted secondary structure is empty",
                    ]
                )
                continue

            if len(secondary_structure) != len(sequence):
                n_failed += 1
                details = (
                    f"sequence_length={len(sequence)}; "
                    f"secondary_structure_length={len(secondary_structure)}"
                )
                logging.warning("Length mismatch for %s: %s", protein_id, details)
                skipped_writer.writerow(
                    [
                        protein_id,
                        "length_mismatch",
                        details,
                    ]
                )
                continue

            n_ok += 1
            writer.writerow([n_ok, protein_id, sequence, secondary_structure])

            if n_total % 100 == 0:
                logging.info(
                    "Processed=%d | OK=%d | skipped_invalid=%d | failed=%d",
                    n_total,
                    n_ok,
                    n_skipped_invalid,
                    n_failed,
                )

    logging.info("Total proteins read: %d", n_total)
    logging.info("Proteins longer than %d residues: %d", WINDOW_SIZE, n_long)
    logging.info("Proteins skipped due to non-standard residues: %d", n_skipped_invalid)
    logging.info("Failed predictions: %d", n_failed)
    logging.info("Proteins successfully predicted: %d", n_ok)
    logging.info("Output written to: %s", output_csv)
    logging.info("Skipped sequences written to: %s", skipped_csv)
    logging.info("Secondary-structure prediction completed successfully")


if __name__ == "__main__":
    main()