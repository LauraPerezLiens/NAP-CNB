#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import BertForMaskedLM, BertTokenizerFast


# ======================================================
# PATHS
# ======================================================

DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INPUT_NN = Path("/data/nap/lperez_nn/data/data_input_nn")
MODEL_DIR = Path("/home/nap/lperez_nn/model/secondary_bert_mlm")


# ======================================================
# INPUT COLUMNS
# ======================================================

PROTEIN_GROUP_COL = "protein_group_id"
GROUP_COL = "group_id"
SEQ_COL = "25aa_seq"
SS_COL = "secondary_structure"
LABEL_COL = "contains_epitope"
POS_COL = "epitope_pos_score"
PROTEIN_COL = "protein_url"
WINDOW_COL = "window_start"
STATUS_COL = "ss_status"


# ======================================================
# PARAMETERS
# ======================================================

BATCH_SIZE = 512
CHUNK_SIZE = 102_400
MAX_LEN = 30
WINDOW_SIZE = 25
HIDDEN_SIZE = 768
SAVE_DTYPE = np.float32

VALID_SS = {"C", "H", "E"}


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
# SECONDARY STRUCTURE PREPROCESSING
# ======================================================

def clean_secondary_structure(ss: str) -> str:
    """
    Clean and normalize a secondary structure string.

    Valid labels are:
        C -> coil
        H -> helix
        E -> beta strand

    Invalid characters are mapped to C.
    Sequences are truncated or padded to WINDOW_SIZE.
    """

    if pd.isna(ss):
        return ""

    ss = str(ss).replace(" ", "").strip().upper()
    ss = "".join([char if char in VALID_SS else "C" for char in ss])

    if len(ss) > WINDOW_SIZE:
        ss = ss[:WINDOW_SIZE]

    if len(ss) < WINDOW_SIZE:
        ss = ss + ("C" * (WINDOW_SIZE - len(ss)))

    return ss


def spaced_secondary_structure(ss: str) -> str:
    """Convert secondary structure labels into BERT token format."""

    return " ".join(ss)


# ======================================================
# MODEL
# ======================================================

def load_model_and_tokenizer(
    model_dir: Path,
) -> Tuple[BertTokenizerFast, BertForMaskedLM, torch.device]:
    """Load SecondaryBERT tokenizer and model."""

    logging.info("Loading SecondaryBERT model from: %s", model_dir)

    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
    model = BertForMaskedLM.from_pretrained(str(model_dir))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    logging.info("Device: %s", device)

    return tokenizer, model, device


# ======================================================
# EMBEDDING GENERATION
# ======================================================

@torch.inference_mode()
def embed_batch_secondary_structures(
    batch: List[str],
    tokenizer: BertTokenizerFast,
    model: BertForMaskedLM,
    device: torch.device,
) -> np.ndarray:
    """
    Generate token-level SecondaryBERT embeddings.

    Output shape:
        batch_size x WINDOW_SIZE x HIDDEN_SIZE
    """

    batch_spaced = [spaced_secondary_structure(ss) for ss in batch]

    tokens = tokenizer(
        batch_spaced,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    input_ids = tokens["input_ids"].to(device, non_blocking=True)
    attention_mask = tokens["attention_mask"].to(device, non_blocking=True)

    outputs = model.bert(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    hidden = outputs.last_hidden_state

    # Remove [CLS] and keep only the 25 secondary structure positions.
    token_embeddings = hidden[:, 1:1 + WINDOW_SIZE, :]

    # Safety padding if tokenization returns fewer than WINDOW_SIZE tokens.
    if token_embeddings.shape[1] < WINDOW_SIZE:
        pad_len = WINDOW_SIZE - token_embeddings.shape[1]
        padding = torch.zeros(
            token_embeddings.shape[0],
            pad_len,
            token_embeddings.shape[2],
            device=device,
            dtype=token_embeddings.dtype,
        )
        token_embeddings = torch.cat([token_embeddings, padding], dim=1)

    return token_embeddings.float().cpu().numpy().astype(SAVE_DTYPE, copy=False)


# ======================================================
# OUTPUT HELPERS
# ======================================================

def build_output_dir(ss_csv_path: Path) -> Path:
    """Build output directory while preserving dataset hierarchy."""

    relative_parent = ss_csv_path.parent.relative_to(DATA_INTERMEDIATE)
    dataset_name = ss_csv_path.stem.replace("_ss", "")

    out_dir = (
        DATA_INPUT_NN
        / relative_parent
        / f"{dataset_name}_SecondaryBERT_by_group"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def save_metadata_and_index(
    df: pd.DataFrame,
    protein_group_ids: np.ndarray,
    group_ids: np.ndarray,
    out_dir: Path,
) -> None:
    """Save metadata, group index and group ID arrays."""

    metadata_path = out_dir / "metadata.csv"
    index_path = out_dir / "index.csv"
    protein_group_ids_path = out_dir / "protein_group_id.npy"
    group_ids_path = out_dir / "group_id.npy"

    metadata_cols = [
        PROTEIN_GROUP_COL,
        GROUP_COL,
        SEQ_COL,
        SS_COL,
        "clean_ss",
        LABEL_COL,
        POS_COL,
        PROTEIN_COL,
        WINDOW_COL,
    ]

    optional_cols = [
        "selected_epitope",
        "protein_found",
        "sequence_match",
        STATUS_COL,
    ]

    metadata_cols += [col for col in optional_cols if col in df.columns]

    df[metadata_cols].to_csv(metadata_path, index=False)

    index_df = pd.DataFrame({
        PROTEIN_GROUP_COL: protein_group_ids,
        GROUP_COL: group_ids,
        "row_idx_in_X_ss": np.arange(len(group_ids), dtype=np.int64),
    })

    index_df.to_csv(index_path, index=False)

    np.save(protein_group_ids_path, protein_group_ids)
    np.save(group_ids_path, group_ids)

    logging.info("Saved metadata: %s", metadata_path)
    logging.info("Saved index: %s", index_path)
    logging.info("Saved all protein group IDs: %s", protein_group_ids_path)
    logging.info("Saved all group IDs: %s", group_ids_path)


def chunk_is_valid(
    x_path: Path,
    protein_gid_path: Path,
    gid_path: Path,
    expected_rows: int,
) -> bool:
    """Check whether an existing chunk is complete and has the expected shape."""

    if not x_path.exists() or not protein_gid_path.exists() or not gid_path.exists():
        return False

    try:
        x = np.load(x_path, mmap_mode="r")
        protein_gids = np.load(protein_gid_path, mmap_mode="r")
        gids = np.load(gid_path, mmap_mode="r")

        return (
            x.shape == (expected_rows, WINDOW_SIZE, HIDDEN_SIZE)
            and protein_gids.shape[0] == expected_rows
            and gids.shape[0] == expected_rows
        )

    except Exception as exc:
        logging.warning("Invalid existing chunk detected: %s", exc)
        return False


def write_embedding_config(
    config_path: Path,
    ss_csv_path: Path,
    total: int,
    batch_size: int,
    chunk_size: int,
) -> None:
    """Save embedding configuration for reproducibility."""

    config = {
        "source_csv": str(ss_csv_path),
        "total_rows": int(total),
        "batch_size": int(batch_size),
        "chunk_size": int(chunk_size),
        "window_size": int(WINDOW_SIZE),
        "max_len": int(MAX_LEN),
        "hidden_size": int(HIDDEN_SIZE),
        "dtype": "float32",
        "output_shape_per_row": [WINDOW_SIZE, HIDDEN_SIZE],
        "model_dir": str(MODEL_DIR),
        "id_columns": [PROTEIN_GROUP_COL, GROUP_COL],
    }

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def remove_previous_outputs(out_dir: Path) -> None:
    """Remove previous outputs when --force is enabled."""

    logging.info("Force enabled. Removing previous outputs in: %s", out_dir)

    for item in [
        out_dir / "protein_group_id.npy",
        out_dir / "group_id.npy",
        out_dir / "group_ids.npy",
        out_dir / "metadata.csv",
        out_dir / "index.csv",
        out_dir / "group_index.csv",
        out_dir / "chunks_manifest.csv",
        out_dir / "embedding_config.json",
        out_dir / "done.flag",
    ]:
        if item.exists():
            item.unlink()

    chunks_dir = out_dir / "chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)


# ======================================================
# MAIN PROCESSING
# ======================================================

def validate_required_columns(df: pd.DataFrame, required_cols: set, csv_path: Path) -> None:
    """Validate required input columns."""

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")


def process_ss_csv(
    ss_csv_path: Path,
    tokenizer: BertTokenizerFast,
    model: BertForMaskedLM,
    device: torch.device,
    batch_size: int,
    chunk_size: int,
) -> None:
    """Generate SecondaryBERT embeddings for one classification_*_ss.csv file."""

    logging.info("=" * 80)
    logging.info("Processing: %s", ss_csv_path)
    logging.info("=" * 80)

    df = pd.read_csv(ss_csv_path)

    required_cols = {
        PROTEIN_GROUP_COL,
        GROUP_COL,
        SEQ_COL,
        SS_COL,
        LABEL_COL,
        POS_COL,
        PROTEIN_COL,
        WINDOW_COL,
    }

    validate_required_columns(df, required_cols, ss_csv_path)

    if STATUS_COL in df.columns:
        before = len(df)
        df = df[df[STATUS_COL] == "ok"].copy()
        logging.info("Kept ss_status == ok: %d/%d", len(df), before)

    df = df.copy()
    df["clean_ss"] = df[SS_COL].apply(clean_secondary_structure)
    df = df[df["clean_ss"] != ""].copy()

    # Keep one row per group_id so secondary embeddings are aligned by group.
    df = df.sort_values(GROUP_COL).drop_duplicates(GROUP_COL, keep="first")
    df = df.reset_index(drop=True)

    out_dir = build_output_dir(ss_csv_path)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / "done.flag"
    manifest_path = out_dir / "chunks_manifest.csv"
    config_path = out_dir / "embedding_config.json"

    protein_group_ids = df[PROTEIN_GROUP_COL].to_numpy(dtype=np.int64)
    group_ids = df[GROUP_COL].to_numpy(dtype=np.int64)
    ss_list = df["clean_ss"].tolist()
    total = len(df)

    logging.info("Rows to embed: %d", total)
    logging.info("Batch size: %d", batch_size)
    logging.info("Chunk size: %d", chunk_size)
    logging.info("Output dir: %s", out_dir)

    save_metadata_and_index(df, protein_group_ids, group_ids, out_dir)
    write_embedding_config(config_path, ss_csv_path, total, batch_size, chunk_size)

    manifest_rows = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_id = start // chunk_size
        expected_rows = end - start

        x_chunk_path = chunks_dir / f"chunk_{chunk_id:06d}_X_ss.npy"
        protein_gid_chunk_path = chunks_dir / f"chunk_{chunk_id:06d}_protein_group_id.npy"
        gid_chunk_path = chunks_dir / f"chunk_{chunk_id:06d}_group_id.npy"

        if chunk_is_valid(
            x_path=x_chunk_path,
            protein_gid_path=protein_gid_chunk_path,
            gid_path=gid_chunk_path,
            expected_rows=expected_rows,
        ):
            logging.info(
                "Skipping chunk %06d because it already exists (%d/%d)",
                chunk_id,
                end,
                total,
            )

            manifest_rows.append({
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "n_rows": expected_rows,
                "x_file": str(x_chunk_path),
                "protein_group_id_file": str(protein_gid_chunk_path),
                "group_id_file": str(gid_chunk_path),
            })
            continue

        tmp_x_path = x_chunk_path.with_suffix(".tmp.npy")
        tmp_protein_gid_path = protein_gid_chunk_path.with_suffix(".tmp.npy")
        tmp_gid_path = gid_chunk_path.with_suffix(".tmp.npy")

        if tmp_x_path.exists():
            tmp_x_path.unlink()
        if tmp_protein_gid_path.exists():
            tmp_protein_gid_path.unlink()
        if tmp_gid_path.exists():
            tmp_gid_path.unlink()

        ss_chunk = ss_list[start:end]
        protein_gid_chunk = protein_group_ids[start:end]
        gid_chunk = group_ids[start:end]

        x_parts = []

        for local_start in range(0, expected_rows, batch_size):
            local_end = min(local_start + batch_size, expected_rows)

            batch = ss_chunk[local_start:local_end]

            x_batch = embed_batch_secondary_structures(
                batch=batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            x_parts.append(x_batch)

            global_done = start + local_end

            if ((local_start // batch_size) % 20 == 0) or (global_done >= total):
                logging.info("Embedded %d/%d", global_done, total)

        x_chunk = np.concatenate(x_parts, axis=0).astype(SAVE_DTYPE, copy=False)

        expected_shape = (expected_rows, WINDOW_SIZE, HIDDEN_SIZE)

        if x_chunk.shape != expected_shape:
            raise RuntimeError(
                f"Bad chunk shape for chunk {chunk_id}: "
                f"{x_chunk.shape}, expected {expected_shape}"
            )

        np.save(tmp_x_path, x_chunk)
        np.save(tmp_protein_gid_path, protein_gid_chunk)
        np.save(tmp_gid_path, gid_chunk)

        tmp_x_path.rename(x_chunk_path)
        tmp_protein_gid_path.rename(protein_gid_chunk_path)
        tmp_gid_path.rename(gid_chunk_path)

        logging.info("Saved chunk %06d: rows %d-%d", chunk_id, start, end)

        manifest_rows.append({
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "n_rows": expected_rows,
            "x_file": str(x_chunk_path),
            "protein_group_id_file": str(protein_gid_chunk_path),
            "group_id_file": str(gid_chunk_path),
        })

        del x_parts
        del x_chunk

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    with done_flag.open("w", encoding="utf-8") as f:
        f.write("done\n")

    logging.info("Saved manifest: %s", manifest_path)
    logging.info("Finished: %s", out_dir)


# ======================================================
# FILE DISCOVERY
# ======================================================

def find_ss_files(species: Optional[str] = None) -> List[Path]:
    """Find classification files with secondary structure annotations."""

    pattern = "**/classification_*_ss.csv"
    files = sorted(DATA_INTERMEDIATE.glob(pattern))

    if species is not None:
        files = [file for file in files if f"/{species}/" in str(file)]

    return files


# ======================================================
# CLI
# ======================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate SecondaryBERT embeddings from classification_*_ss.csv files."
    )

    parser.add_argument(
        "--csv",
        default=None,
        help="Specific classification_*_ss.csv file to process.",
    )

    parser.add_argument(
        "--species",
        choices=["human", "mouse"],
        default=None,
        help="Process only one species if --csv is not provided.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Rows saved per chunk.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing outputs and recompute.",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging()

    args = parse_args()

    DATA_INPUT_NN.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR)

    if args.csv is not None:
        files = [Path(args.csv)]
    else:
        files = find_ss_files(species=args.species)

    logging.info("Files to process: %d", len(files))

    for file_path in files:
        out_dir = build_output_dir(file_path)

        if args.force and out_dir.exists():
            remove_previous_outputs(out_dir)

        process_ss_csv(
            ss_csv_path=file_path,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )

    logging.info("All SecondaryBERT embeddings finished.")


if __name__ == "__main__":
    main()