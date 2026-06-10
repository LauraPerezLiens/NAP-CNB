#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split UniRef50 representative sequences into FASTA chunks.

This script divides the MMseqs2 representative sequence FASTA file
into smaller FASTA files to enable parallel secondary-structure
prediction with ProteinUnet.

Input:
    uniref50_c30_rep_seq.fasta

Output:
    chunk_00001.fasta
    chunk_00002.fasta
    ...
"""

import logging
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

INPUT_FASTA = Path(
    "/home/nap/lperez_nn/data/data_uniref50/uniref50_c30_rep_seq.fasta"
)

OUTPUT_DIR = Path(
    "/home/nap/lperez_nn/data/data_uniref50/secondary_structure_prediction/chunks"
)

SEQUENCES_PER_CHUNK = 100_000


# ============================================================
# Logging
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )


# ============================================================
# Helpers
# ============================================================

def open_chunk(chunk_index: int):
    chunk_path = OUTPUT_DIR / f"chunk_{chunk_index:05d}.fasta"
    return open(chunk_path, "w"), chunk_path


# ============================================================
# Main
# ============================================================

def main() -> None:
    setup_logging()

    if not INPUT_FASTA.exists():
        raise FileNotFoundError(f"Input FASTA not found: {INPUT_FASTA}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info("Input FASTA: %s", INPUT_FASTA)
    logging.info("Output directory: %s", OUTPUT_DIR)
    logging.info("Sequences per chunk: %d", SEQUENCES_PER_CHUNK)

    chunk_index = 1
    sequences_in_chunk = 0
    total_sequences = 0

    output_handle = None

    with open(INPUT_FASTA, "r") as fasta_handle:

        for line in fasta_handle:

            if line.startswith(">"):

                if sequences_in_chunk == 0:
                    output_handle, chunk_path = open_chunk(chunk_index)

                    logging.info(
                        "Creating chunk %05d -> %s",
                        chunk_index,
                        chunk_path.name,
                    )

                elif sequences_in_chunk >= SEQUENCES_PER_CHUNK:

                    output_handle.close()

                    logging.info(
                        "Finished chunk %05d (%d sequences)",
                        chunk_index,
                        sequences_in_chunk,
                    )

                    chunk_index += 1
                    sequences_in_chunk = 0

                    output_handle, chunk_path = open_chunk(chunk_index)

                    logging.info(
                        "Creating chunk %05d -> %s",
                        chunk_index,
                        chunk_path.name,
                    )

                sequences_in_chunk += 1
                total_sequences += 1

            if output_handle is not None:
                output_handle.write(line)

    if output_handle is not None:
        output_handle.close()

    logging.info("Total sequences processed: %d", total_sequences)
    logging.info("Total chunks generated: %d", chunk_index)
    logging.info("Completed successfully.")


if __name__ == "__main__":
    main()