#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audit missing UniRef30 secondary-structure predictions.

This script compares each input FASTA chunk against its corresponding
ProteinUnet secondary-structure prediction CSV file.

For every protein present in the FASTA chunk but missing from the prediction
output, the script checks whether the sequence contains non-standard amino
acids.

Expected input structure:
    proteinunet_prediction/chunks/
        chunk_00001.fasta
        chunk_00002.fasta
        ...

    proteinunet_prediction/outputs/
        chunk_00001.csv
        chunk_00002.csv
        ...

Output:
    proteinunet_prediction/missing_secondary_structure_audit.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Set


STANDARD_AA = set("ADNRCEQGHILKMF PSTWYV".replace(" ", ""))


def setup_logging() -> None:
    """Configure logging format and verbosity level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Audit missing UniRef30 ProteinUnet predictions."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nap/lperez_nn/data/data_uniref50"),
        help="Base UniRef data directory.",
    )

    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing input FASTA chunks. "
            "Default: <data-dir>/proteinunet_prediction/chunks"
        ),
    )

    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing secondary-structure prediction CSV files. "
            "Default: <data-dir>/proteinunet_prediction/outputs"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output audit CSV file. "
            "Default: <data-dir>/proteinunet_prediction/missing_secondary_structure_audit.csv"
        ),
    )

    return parser.parse_args()


def parse_fasta(fasta_path: Path) -> Dict[str, str]:
    """Return a dictionary mapping protein_id to sequence from a FASTA file."""
    sequences = {}
    header = None
    seq_lines = []

    with fasta_path.open("r") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    protein_id = header.split()[0]
                    sequences[protein_id] = "".join(seq_lines).upper()

                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            protein_id = header.split()[0]
            sequences[protein_id] = "".join(seq_lines).upper()

    return sequences


def read_predicted_ids(csv_path: Path) -> Set[str]:
    """Return protein IDs present in a prediction CSV file."""
    predicted_ids = set()

    if not csv_path.is_file():
        return predicted_ids

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"Empty or invalid CSV file: {csv_path}")

        if "protein_id" not in reader.fieldnames:
            raise ValueError(f"'protein_id' column not found in {csv_path}")

        for row in reader:
            predicted_ids.add(row["protein_id"])

    return predicted_ids


def get_invalid_residues(sequence: str) -> list[str]:
    """Return non-standard residues found in a protein sequence."""
    return sorted(set(residue for residue in sequence if residue not in STANDARD_AA))


def main() -> None:
    """Audit missing ProteinUnet predictions across all FASTA chunks."""
    setup_logging()
    args = parse_args()

    data_dir = args.data_dir
    prediction_dir = data_dir / "proteinunet_prediction"

    chunks_dir = args.chunks_dir or prediction_dir / "chunks"
    outputs_dir = args.outputs_dir or prediction_dir / "outputs"
    output_csv = args.output or prediction_dir / "missing_secondary_structure_audit.csv"

    if not chunks_dir.is_dir():
        raise NotADirectoryError(f"Chunks directory not found: {chunks_dir}")

    if not outputs_dir.is_dir():
        raise NotADirectoryError(f"Outputs directory not found: {outputs_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fasta_files = sorted(chunks_dir.glob("chunk_*.fasta"))

    if not fasta_files:
        raise FileNotFoundError(f"No FASTA chunks found in {chunks_dir}")

    logging.info("Starting UniRef30 secondary-structure prediction audit")
    logging.info("Chunks directory: %s", chunks_dir)
    logging.info("Outputs directory: %s", outputs_dir)
    logging.info("Audit output: %s", output_csv)

    total_fasta = 0
    total_predicted = 0
    total_missing = 0
    total_missing_with_invalid = 0
    total_missing_csv = 0
    total_unknown_missing = 0

    with output_csv.open("w", newline="") as out_handle:
        writer = csv.writer(out_handle)

        writer.writerow(
            [
                "chunk",
                "protein_id",
                "sequence_length",
                "missing_reason",
                "invalid_residues",
                "sequence",
            ]
        )

        for fasta_path in fasta_files:
            chunk_name = fasta_path.stem
            csv_path = outputs_dir / f"{chunk_name}.csv"

            logging.info("Checking %s", chunk_name)

            fasta_sequences = parse_fasta(fasta_path)
            predicted_ids = read_predicted_ids(csv_path)

            fasta_ids = set(fasta_sequences)
            missing_ids = sorted(fasta_ids - predicted_ids)

            total_fasta += len(fasta_ids)
            total_predicted += len(predicted_ids)
            total_missing += len(missing_ids)

            if not csv_path.is_file():
                total_missing_csv += 1
                logging.warning("Prediction CSV not found for %s: %s", chunk_name, csv_path)

            for protein_id in missing_ids:
                sequence = fasta_sequences[protein_id]
                invalid_residues = get_invalid_residues(sequence)

                if invalid_residues:
                    missing_reason = "non_standard_residues"
                    total_missing_with_invalid += 1
                elif not csv_path.is_file():
                    missing_reason = "missing_output_csv"
                else:
                    missing_reason = "missing_unknown_reason"
                    total_unknown_missing += 1

                writer.writerow(
                    [
                        chunk_name,
                        protein_id,
                        len(sequence),
                        missing_reason,
                        ",".join(invalid_residues),
                        sequence,
                    ]
                )

            logging.info(
                "%s | FASTA=%d | predicted=%d | missing=%d",
                chunk_name,
                len(fasta_ids),
                len(predicted_ids),
                len(missing_ids),
            )

    logging.info("====================================================")
    logging.info("Total FASTA proteins: %d", total_fasta)
    logging.info("Total predicted proteins: %d", total_predicted)
    logging.info("Total missing proteins: %d", total_missing)
    logging.info(
        "Missing proteins with non-standard residues: %d",
        total_missing_with_invalid,
    )
    logging.info("Chunks without prediction CSV: %d", total_missing_csv)
    logging.info("Missing proteins with unknown reason: %d", total_unknown_missing)
    logging.info("Audit written to: %s", output_csv)
    logging.info("Completed successfully")


if __name__ == "__main__":
    main()