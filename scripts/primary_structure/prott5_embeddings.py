#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer


# ======================================================
# PATHS (usar almacenamiento grande)
# ======================================================
DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INPUT_NN = Path("/data/nap/lperez_nn/data/data_input_nn")

MODEL_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"

# ======================================================
# COLUMNAS
# ======================================================
SEQ_COL = "blosum_seq"
LABEL_COL = "contains_epitope"
POS_COL = "epitope_pos_score"
GROUP_COL = "group_id"

# ======================================================
# PARÁMETROS
# ======================================================
BATCH_SIZE = 32          # subir si cabe en GPU
CHUNK_SIZE = 100000
MAX_LENGTH = 30
SAVE_DTYPE = np.float32


# ======================================================
# LIMPIEZA SECUENCIA
# ======================================================
def clean_sequence(seq: str) -> str:
    if pd.isna(seq):
        return ""

    seq = str(seq).strip().upper()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZUOJ]", "X", seq)
    seq = re.sub(r"[UZOB]", "X", seq)

    return seq


def spaced_sequence(seq: str) -> str:
    return " ".join(seq)


# ======================================================
# MODELO
# ======================================================
def load_model_and_tokenizer():
    print(f"[INFO] Loading model: {MODEL_NAME}")

    tokenizer = T5Tokenizer.from_pretrained(
        MODEL_NAME,
        do_lower_case=False
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        model = T5EncoderModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16
        )

        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    else:
        device = torch.device("cpu")

        model = T5EncoderModel.from_pretrained(MODEL_NAME)
        model = model.float()

    model = model.to(device)
    model.eval()

    print(f"[INFO] Device: {device}")

    return tokenizer, model, device


# ======================================================
# EMBEDDINGS
# ======================================================
@torch.inference_mode()
def embed_sequences_mean_pool(
    sequences,
    tokenizer,
    model,
    device,
    batch_size=BATCH_SIZE
):
    all_embeddings = []

    total = len(sequences)

    print(f"[INFO] Unique sequences to embed: {total}")
    print(f"[INFO] ProtT5 batch size: {batch_size}")

    for i in range(0, total, batch_size):
        batch = sequences[i:i + batch_size]
        batch_spaced = [spaced_sequence(s) for s in batch]

        tokens = tokenizer(
            batch_spaced,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(device, non_blocking=True)
        attention_mask = tokens["attention_mask"].to(device, non_blocking=True)

        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        pooled = pooled.float().cpu().numpy()

        all_embeddings.append(pooled)

        if ((i // batch_size) % 50 == 0) or (i + batch_size >= total):
            print(f"[INFO] Embedded {min(i + batch_size, total)}/{total}")

    if not all_embeddings:
        return np.empty((0, 1024), dtype=SAVE_DTYPE)

    return np.vstack(all_embeddings).astype(SAVE_DTYPE, copy=False)


# ======================================================
# OUTPUT DIR
# ======================================================
def build_output_dir(csv_path: Path):
    relative_parent = csv_path.parent.relative_to(DATA_INTERMEDIATE)

    out_dir = (
        DATA_INPUT_NN /
        relative_parent /
        f"{csv_path.stem}_ProtT5_chunks"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


# ======================================================
# SAVE CHUNK
# ======================================================
def save_chunk_files(
    out_dir,
    chunk_id,
    X,
    y,
    pos,
    group_ids,
    metadata_df,
    index_df
):
    prefix = out_dir / f"chunk_{chunk_id:06d}"

    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)
    np.save(f"{prefix}_pos.npy", pos)
    np.save(f"{prefix}_group_id.npy", group_ids)

    metadata_df.to_csv(f"{prefix}_metadata.csv", index=False)
    index_df.to_csv(f"{prefix}_index.csv", index=False)

    print(f"[OK] Saved chunk {chunk_id:06d}")


def append_master_index(master_index_path, index_df):
    write_header = not master_index_path.exists()

    index_df.to_csv(
        master_index_path,
        mode="a",
        header=write_header,
        index=False
    )


# ======================================================
# MAIN PROCESS
# ======================================================
def process_csv_in_chunks(csv_path, tokenizer, model, device):
    print("=" * 60)
    print(f"[INFO] Processing {csv_path}")
    print("=" * 60)

    required = {SEQ_COL, LABEL_COL, POS_COL, GROUP_COL}

    out_dir = build_output_dir(csv_path)
    master_index_path = out_dir / "master_index.csv"

    # reconstruir master index si ya existe carpeta
    if master_index_path.exists():
        master_index_path.unlink()

    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE)

    for chunk_id, df in enumerate(reader):
        prefix = out_dir / f"chunk_{chunk_id:06d}_X.npy"

        # saltar chunks ya existentes
        if prefix.exists():
            print(f"[SKIP] chunk_{chunk_id:06d} already exists")
            continue

        print("-" * 60)
        print(f"[INFO] Chunk {chunk_id:06d}")
        print("-" * 60)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.copy()

        df["clean_seq"] = df[SEQ_COL].apply(clean_sequence)

        df_valid = df[df["clean_seq"] != ""].copy().reset_index(drop=True)

        if df_valid.empty:
            print("[SKIP] Empty valid chunk")
            continue

        # factorize = más rápido que dict python
        codes, uniques = pd.factorize(
            df_valid["clean_seq"],
            sort=False
        )

        unique_sequences = uniques.tolist()

        unique_embeddings = embed_sequences_mean_pool(
            unique_sequences,
            tokenizer,
            model,
            device,
            batch_size=BATCH_SIZE
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
            "group_id": group_ids
        })

        save_chunk_files(
            out_dir,
            chunk_id,
            X,
            y,
            pos,
            group_ids,
            metadata_df,
            index_df
        )

        append_master_index(master_index_path, index_df)

    print("[INFO] Finished")


# ======================================================
# ENTRYPOINT
# ======================================================
def main():
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