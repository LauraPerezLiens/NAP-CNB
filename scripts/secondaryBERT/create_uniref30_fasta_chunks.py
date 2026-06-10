#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split UniRef30 representative sequences into FASTA chunks.

This script splits the representative sequences obtained after MMseqs2
clustering of UniRef50 at 30% sequence identity.

The resulting FASTA chunks are used as independent inputs for parallel
secondary-structure prediction with ProteinUnet.

Input:
    clustering/uniref50_c30_rep_seq.fasta

Output:
    proteinunet_prediction/chunks/chunk_00001.fasta
    proteinunet_prediction/chunks/chunk_00002.fasta
    ...

Notes:
    Although the input originates from UniRef50, after clustering at 30%
    identity the working dataset is referred to as UniRef30.
"""

import argparse
import logging
from pathlib import Path
from typing import TextIO, Tuple


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
        description="Split UniRef30 representative sequences into FASTA chunks."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/nap/lperez_nn/data/data_uniref50"),
        help="Base UniRef data directory.",
    )

    parser.add_argument(
        "--input-fasta",
        type=Path,
        default=None,
        help=(
            "Input representative FASTA file. "
            "Default: <data-dir>/clustering/uniref50_c30_rep_seq.fasta"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for FASTA chunks. "
            "Default: <data-dir>/proteinunet_prediction/chunks"
        ),
    )

    parser.add_argument(
        "--sequences-per-chunk",
        type=int,
        default=100_000,
        help="Number of protein sequences per FASTA chunk.",
    )

    return parser.parse_args()


# ============================================================
# Helper functions
# ============================================================

def open_chunk(output_dir: Path, chunk_index: int) -> Tuple[TextIO, Path]:
    """
    Open a new FASTA chunk file for writing.

    Parameters
    ----------
    output_dir : Path
        Directory where FASTA chunks are written.
    chunk_index : int
        Sequential chunk identifier.

    Returns
    -------
    Tuple[TextIO, Path]
        Open file handle and output chunk path.
    """
    chunk_path = output_dir / f"chunk_{chunk_index:05d}.fasta"
    return chunk_path.open("w"), chunk_path


def validate_inputs(input_fasta: Path, sequences_per_chunk: int) -> None:
    """Validate input file and chunking parameters."""
    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    if sequences_per_chunk <= 0:
        raise ValueError("sequences_per_chunk must be greater than zero")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Split the UniRef30 representative FASTA file into fixed-size chunks."""
    setup_logging()
    args = parse_args()

    data_dir = args.data_dir
    input_fasta = args.input_fasta or data_dir / "clustering" / "uniref50_c30_rep_seq.fasta"
    output_dir = args.output_dir or data_dir / "proteinunet_prediction" / "chunks"
    sequences_per_chunk = args.sequences_per_chunk

    validate_inputs(input_fasta, sequences_per_chunk)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting UniRef30 FASTA chunking")
    logging.info("Input FASTA: %s", input_fasta)
    logging.info("Output directory: %s", output_dir)
    logging.info("Sequences per chunk: %d", sequences_per_chunk)

    chunk_index = 1
    sequences_in_chunk = 0
    total_sequences = 0
    output_handle = None

    with input_fasta.open("r") as fasta_handle:
        for line in fasta_handle:
            if line.startswith(">"):

                if sequences_in_chunk == 0:
                    output_handle, chunk_path = open_chunk(output_dir, chunk_index)
                    logging.info("Creating chunk %05d: %s", chunk_index, chunk_path.name)

                elif sequences_in_chunk >= sequences_per_chunk:
                    output_handle.close()
                    logging.info(
                        "Finished chunk %05d with %d sequences",
                        chunk_index,
                        sequences_in_chunk,
                    )

                    chunk_index += 1
                    sequences_in_chunk = 0

                    output_handle, chunk_path = open_chunk(output_dir, chunk_index)
                    logging.info("Creating chunk %05d: %s", chunk_index, chunk_path.name)

                sequences_in_chunk += 1
                total_sequences += 1

            if output_handle is not None:
                output_handle.write(line)

    if output_handle is not None:
        output_handle.close()
        logging.info(
            "Finished chunk %05d with %d sequences",
            chunk_index,
            sequences_in_chunk,
        )

    logging.info("Total sequences processed: %d", total_sequences)
    logging.info("Total chunks generated: %d", chunk_index if total_sequences else 0)
    logging.info("FASTA chunking completed successfully")


if __name__ == "__main__":
    main()