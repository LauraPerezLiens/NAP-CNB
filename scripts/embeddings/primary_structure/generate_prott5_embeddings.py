#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer


# ======================================================
# PATHS
# ======================================================

DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INPUT_NN = Path("/data/nap/lperez_nn/data/data_input_nn")

MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"


# ======================================================
# INPUT COLUMNS
# ======================================================

SEQ_COL = "blosum_seq"
LABEL_COL = "contains_epitope"
POS_COL = "epitope_pos_score"
GROUP_COL = "group_id"


# ======================================================
# PARAMETERS
# ======================================================

BATCH_SIZE = 32
CHUNK_SIZE = 100_000
MAX_LENGTH = 30
SAVE_DTYPE = np.float32
EMBEDDING_DIM = 1024


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
# SEQUENCE PREPROCESSING
# ======================================================

def clean_sequence(seq: str) -> str:
    """
    Clean protein sequence before ProtT5 tokenization.

    Non-standard or unexpected characters are converted to X.
    ProtT5 expects rare amino acids such as U, Z, O and B to be mapped to X.
    """

    if pd.isna(seq):
        return ""

    seq = str(seq).strip().upper()

    # Replace unexpected characters by X.
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", "X", seq)

    # Map rare amino acids to X, as commonly done for ProtT5.
    seq = re.sub(r"[UZOB]", "X", seq)

    return seq


def spaced_sequence(seq: str) -> str:
    """Convert a protein sequence into the space-separated format expected by ProtT5."""

    return " ".join(seq)


# ======================================================
# MODEL
# ======================================================

def load_model_and_tokenizer() -> Tuple[T5Tokenizer, T5EncoderModel, torch.device]:
    """Load ProtT5 tokenizer and encoder model."""

    logging.info("Loading model: %s", MODEL_NAME)

    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        do_lower_case=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        model = T5EncoderModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
        )

        logging.info("GPU: %s", torch.cuda.get_device_name(0))

    else:
        device = torch.device("cpu")
        model = T5EncoderModel.from_pretrained(MODEL_NAME)
        model = model.float()

    model = model.to(device)
    model.eval()

    logging.info("Device: %s", device)

    return tokenizer, model, device


# ======================================================
# EMBEDDINGS
# ======================================================

@torch.inference_mode()
def embed_sequences_mean_pool(
    sequences: List[str],
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Generate ProtT5 embeddings and apply mean pooling over valid tokens.

    Each input sequence is embedded with ProtT5, then token-level embeddings are
    averaged using the attention mask. The final output has one vector per sequence.
    """

    all_embeddings = []
    total = len(sequences)

    logging.info("Unique sequences to embed: %d", total)
    logging.info("ProtT5 batch size: %d", batch_size)

    for i in range(0, total, batch_size):
        batch = sequences[i:i + batch_size]
        batch_spaced = [spaced_sequence(seq) for seq in batch]

        tokens = tokenizer(
            batch_spaced,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].to(device, non_blocking=True)
        attention_mask = tokens["attention_mask"].to(device, non_blocking=True)

        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        # Mean pooling over non-padding tokens.
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        pooled = pooled.float().cpu().numpy()
        all_embeddings.append(pooled)

        if ((i // batch_size) % 50 == 0) or (i + batch_size >= total):
            logging.info("Embedded %d/%d", min(i + batch_size, total), total)

    if not all_embeddings:
        return np.empty((0, EMBEDDING_DIM), dtype=SAVE_DTYPE)

    return np.vstack(all_embeddings).astype(SAVE_DTYPE, copy=False)


# ======================================================
# OUTPUT HELPERS
# ======================================================

def build_output_dir(csv_path: Path) -> Path:
    """
    Build the output directory preserving the input dataset hierarchy.

    Example:
        data_intermediate/human/mhc-I/HLA-A/input.csv

    becomes:
        data_input_nn/human/mhc-I/HLA-A/input_ProtT5_chunks/
    """

    relative_parent = csv_path.parent.relative_to(DATA_INTERMEDIATE)

    out_dir = (
        DATA_INPUT_NN
        / relative_parent
        / f"{csv_path.stem}_ProtT5_chunks"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def save_chunk_files(
    out_dir: Path,
    chunk_id: int,
    X: np.ndarray,
    y: np.ndarray,
    pos: np.ndarray,
    group_ids: np.ndarray,
    metadata_df: pd.DataFrame,
    index_df: pd.DataFrame,
) -> None:
    """Save embeddings, labels, metadata and index files for one chunk."""

    prefix = out_dir / f"chunk_{chunk_id:06d}"

    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)
    np.save(f"{prefix}_pos.npy", pos)
    np.save(f"{prefix}_group_id.npy", group_ids)

    metadata_df.to_csv(f"{prefix}_metadata.csv", index=False)
    index_df.to_csv(f"{prefix}_index.csv", index=False)

    logging.info("Saved chunk %06d", chunk_id)


def append_master_index(master_index_path: Path, index_df: pd.DataFrame) -> None:
    """Append chunk index information to the master index file."""

    write_header = not master_index_path.exists()

    index_df.to_csv(
        master_index_path,
        mode="a",
        header=write_header,
        index=False,
    )


# ======================================================
# MAIN PROCESS
# ======================================================

def validate_required_columns(df: pd.DataFrame, required_cols: set) -> None:
    """Validate that all required columns are present in the input DataFrame."""

    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def process_csv_in_chunks(
    csv_path: Path,
    tokenizer: T5Tokenizer,
    model: T5EncoderModel,
    device: torch.device,
) -> None:
    """
    Process a BLOSUM-augmented classification CSV in chunks.

    For each chunk:
        1. Clean sequences.
        2. Factorize unique sequences.
        3. Embed each unique sequence once.
        4. Map embeddings back to all rows.
        5. Save embeddings and metadata.
    """

    logging.info("=" * 60)
    logging.info("Processing %s", csv_path)
    logging.info("=" * 60)

    required_cols = {SEQ_COL, LABEL_COL, POS_COL, GROUP_COL}

    out_dir = build_output_dir(csv_path)
    master_index_path = out_dir / "master_index.csv"

    # Rebuild master index on each run so it reflects the current processed chunks.
    if master_index_path.exists():
        master_index_path.unlink()

    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)

    for chunk_id, df in enumerate(reader):
        prefix = out_dir / f"chunk_{chunk_id:06d}_X.npy"

        # Skip chunks that were already generated. This allows interrupted jobs to resume.
        if prefix.exists():
            logging.info("Skipping chunk_%06d because output already exists", chunk_id)
            continue

        logging.info("-" * 60)
        logging.info("Chunk %06d", chunk_id)
        logging.info("-" * 60)

        validate_required_columns(df, required_cols)

        df = df.copy()
        df["clean_seq"] = df[SEQ_COL].apply(clean_sequence)

        df_valid = df[df["clean_seq"] != ""].copy().reset_index(drop=True)

        if df_valid.empty:
            logging.warning("Skipping empty valid chunk %06d", chunk_id)
            continue

        # Factorization avoids embedding repeated sequences multiple times.
        codes, uniques = pd.factorize(
            df_valid["clean_seq"],
            sort=False,
        )

        unique_sequences = uniques.tolist()

        unique_embeddings = embed_sequences_mean_pool(
            unique_sequences,
            tokenizer,
            model,
            device,
            batch_size=BATCH_SIZE,
        )

        X = unique_embeddings[codes].astype(SAVE_DTYPE, copy=False)

        y = df_valid[LABEL_COL].to_numpy(dtype=np.int64)
        pos = df_valid[POS_COL].to_numpy(dtype=SAVE_DTYPE)
        group_ids = df_valid[GROUP_COL].to_numpy(dtype=np.int64)

        metadata_df = df_valid[
            [GROUP_COL, SEQ_COL, LABEL_COL, POS_COL]
        ].copy()

        index_df = pd.DataFrame({
            "chunk_id": chunk_id,
            "row_idx_in_chunk": np.arange(len(df_valid), dtype=np.int64),
            "y": y,
            "group_id": group_ids,
        })

        save_chunk_files(
            out_dir=out_dir,
            chunk_id=chunk_id,
            X=X,
            y=y,
            pos=pos,
            group_ids=group_ids,
            metadata_df=metadata_df,
            index_df=index_df,
        )

        append_master_index(master_index_path, index_df)

    logging.info("Finished processing %s", csv_path)


# ======================================================
# ENTRYPOINT
# ======================================================

def main() -> None:
    setup_logging()

    DATA_INPUT_NN.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer()

    csv_path = Path(
        "/data/nap/lperez_nn/data/data_intermediate/"
        "human/mhc-I/HLA-A/"
        "classification_human_mhc-I_HLA-A_blosum.csv"
    )

    process_csv_in_chunks(csv_path, tokenizer, model, device)


if __name__ == "__main__":
    main()