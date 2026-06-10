#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cluster UniRef50 sequences into UniRef30 representatives with MMseqs2.

This script runs MMseqs2 easy-cluster on the UniRef50 protein database using
30% minimum sequence identity and 80% target coverage.

Although the input database is UniRef50, the resulting representative set is
referred to as UniRef30 throughout the secondary-BERT pipeline.

Main outputs:
    uniref50_c30_cluster.tsv
    uniref50_c30_rep_seq.fasta
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cluster UniRef50 into UniRef30 representatives using MMseqs2."
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
        help="Input UniRef50 FASTA file. Default: <data-dir>/uniref50.fasta",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output clustering directory. Default: <data-dir>/clustering",
    )

    parser.add_argument(
        "--mmseqs-bin",
        type=Path,
        default=None,
        help="Path to MMseqs2 executable. Default: <data-dir>/mmseqs2/mmseqs/bin/mmseqs",
    )

    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Temporary directory. Default: <output-dir>/tmp",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of threads to use.",
    )

    parser.add_argument(
        "--min-seq-id",
        type=float,
        default=0.3,
        help="Minimum sequence identity for MMseqs2 clustering.",
    )

    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Coverage threshold for MMseqs2 clustering.",
    )

    parser.add_argument(
        "--cov-mode",
        type=int,
        default=1,
        help="MMseqs2 coverage mode. 1 corresponds to target sequence coverage.",
    )

    return parser.parse_args()


def validate_inputs(input_fasta: Path, mmseqs_bin: Path) -> None:
    """Validate required input files."""
    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    if not mmseqs_bin.is_file():
        raise FileNotFoundError(f"MMseqs2 binary not found: {mmseqs_bin}")

    if not os.access(mmseqs_bin, os.X_OK):
        raise PermissionError(f"MMseqs2 binary is not executable: {mmseqs_bin}")


def run_mmseqs_clustering(
    mmseqs_bin: Path,
    input_fasta: Path,
    output_prefix: Path,
    tmp_dir: Path,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
) -> None:
    """Run MMseqs2 easy-cluster."""
    command = [
        str(mmseqs_bin),
        "easy-cluster",
        str(input_fasta),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]

    logging.info("Running command:")
    logging.info(" ".join(command))

    subprocess.run(command, check=True)
    

def main() -> None:
    """Cluster UniRef50 into UniRef30 representative sequences."""
    setup_logging()
    args = parse_args()

    data_dir = args.data_dir
    input_fasta = args.input_fasta or data_dir / "uniref50.fasta"
    output_dir = args.output_dir or data_dir / "clustering"
    mmseqs_bin = args.mmseqs_bin or data_dir / "mmseqs2" / "mmseqs" / "bin" / "mmseqs"
    tmp_dir = args.tmp_dir or output_dir / "tmp"

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = output_dir / "uniref50_c30"

    validate_inputs(input_fasta, mmseqs_bin)

    logging.info("Starting UniRef50 to UniRef30 clustering")
    logging.info("Input FASTA: %s", input_fasta)
    logging.info("Output directory: %s", output_dir)
    logging.info("Output prefix: %s", output_prefix)
    logging.info("Temporary directory: %s", tmp_dir)
    logging.info("MMseqs2 binary: %s", mmseqs_bin)
    logging.info("Minimum sequence identity: %.2f", args.min_seq_id)
    logging.info("Coverage threshold: %.2f", args.coverage)
    logging.info("Coverage mode: %d", args.cov_mode)
    logging.info("Threads: %d", args.threads)

    run_mmseqs_clustering(
        mmseqs_bin=mmseqs_bin,
        input_fasta=input_fasta,
        output_prefix=output_prefix,
        tmp_dir=tmp_dir,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        threads=args.threads,
    )
    

    expected_outputs = [
        output_dir / "uniref50_c30_cluster.tsv",
        output_dir / "uniref50_c30_rep_seq.fasta",
    ]

    for expected_file in expected_outputs:
        if not expected_file.is_file():
            raise FileNotFoundError(
                f"Expected output file was not generated: {expected_file}"
            )

    logging.info("MMseqs2 clustering completed successfully")
    logging.info("Generated files:")

    for output_file in sorted(output_dir.glob("uniref50_c30*")):
        logging.info("  %s", output_file)


if __name__ == "__main__":
    main()