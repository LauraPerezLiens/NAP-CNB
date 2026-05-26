#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForMaskedLM


DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INPUT_NN = Path("/data/nap/lperez_nn/data/data_input_nn")
MODEL_DIR = Path("/home/nap/lperez_nn/model/secondary_bert_mlm")

GROUP_COL = "group_id"
SEQ_COL = "25aa_seq"
SS_COL = "secondary_structure"
LABEL_COL = "contains_epitope"
POS_COL = "epitope_pos_score"
PROTEIN_COL = "protein_url"
WINDOW_COL = "window_start"
STATUS_COL = "ss_status"

BATCH_SIZE = 512
CHUNK_SIZE = 102400
MAX_LEN = 30
WINDOW_SIZE = 25
HIDDEN_SIZE = 768
SAVE_DTYPE = np.float32


def clean_secondary_structure(ss: str) -> str:
    if pd.isna(ss):
        return ""

    ss = str(ss).replace(" ", "").strip().upper()
    valid = {"C", "H", "E"}
    ss = "".join([x if x in valid else "C" for x in ss])

    if len(ss) > WINDOW_SIZE:
        ss = ss[:WINDOW_SIZE]

    if len(ss) < WINDOW_SIZE:
        ss = ss + ("C" * (WINDOW_SIZE - len(ss)))

    return ss


def spaced_secondary_structure(ss: str) -> str:
    return " ".join(ss)


def load_model_and_tokenizer(model_dir: Path):
    print(f"[INFO] Loading Secondary BERT model from: {model_dir}", flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(str(model_dir))
    model = BertForMaskedLM.from_pretrained(str(model_dir))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print(f"[INFO] Device: {device}", flush=True)

    return tokenizer, model, device


@torch.inference_mode()
def embed_batch_secondary_structures(batch, tokenizer, model, device):
    batch_spaced = [spaced_secondary_structure(s) for s in batch]

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
    token_embeddings = hidden[:, 1:1 + WINDOW_SIZE, :]

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


def build_output_dir(ss_csv_path: Path):
    relative_parent = ss_csv_path.parent.relative_to(DATA_INTERMEDIATE)
    dataset_name = ss_csv_path.stem.replace("_ss", "")

    out_dir = (
        DATA_INPUT_NN /
        relative_parent /
        f"{dataset_name}_SecondaryBERT_by_group"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_metadata_and_index(df, group_ids, out_dir):
    metadata_path = out_dir / "metadata.csv"
    group_index_path = out_dir / "group_index.csv"
    group_ids_path = out_dir / "group_ids.npy"

    metadata_cols = [
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
        "ss_status",
    ]

    metadata_cols += [c for c in optional_cols if c in df.columns]

    df[metadata_cols].to_csv(metadata_path, index=False)

    group_index_df = pd.DataFrame({
        GROUP_COL: group_ids,
        "row_idx_in_X_ss": np.arange(len(group_ids), dtype=np.int64),
    })

    group_index_df.to_csv(group_index_path, index=False)
    np.save(group_ids_path, group_ids)

    print(f"[OK] Saved metadata: {metadata_path}", flush=True)
    print(f"[OK] Saved group index: {group_index_path}", flush=True)
    print(f"[OK] Saved all group_ids: {group_ids_path}", flush=True)


def chunk_is_valid(x_path: Path, gid_path: Path, expected_rows: int) -> bool:
    if not x_path.exists() or not gid_path.exists():
        return False

    try:
        x = np.load(x_path, mmap_mode="r")
        gids = np.load(gid_path, mmap_mode="r")

        return (
            x.shape == (expected_rows, WINDOW_SIZE, HIDDEN_SIZE)
            and gids.shape[0] == expected_rows
        )
    except Exception:
        return False


def process_ss_csv(ss_csv_path: Path, tokenizer, model, device, batch_size: int, chunk_size: int):
    print("=" * 80, flush=True)
    print(f"[INFO] Processing: {ss_csv_path}", flush=True)
    print("=" * 80, flush=True)

    df = pd.read_csv(ss_csv_path)

    required = {
        GROUP_COL,
        SEQ_COL,
        SS_COL,
        LABEL_COL,
        POS_COL,
        PROTEIN_COL,
        WINDOW_COL,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {ss_csv_path}: {missing}")

    if STATUS_COL in df.columns:
        before = len(df)
        df = df[df[STATUS_COL] == "ok"].copy()
        print(f"[INFO] Kept ss_status == ok: {len(df):,}/{before:,}", flush=True)

    df = df.copy()
    df["clean_ss"] = df[SS_COL].apply(clean_secondary_structure)
    df = df[df["clean_ss"] != ""].copy()

    df = df.sort_values(GROUP_COL).drop_duplicates(GROUP_COL, keep="first")
    df = df.reset_index(drop=True)

    out_dir = build_output_dir(ss_csv_path)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    done_flag = out_dir / "done.flag"
    manifest_path = out_dir / "chunks_manifest.csv"
    config_path = out_dir / "embedding_config.json"

    group_ids = df[GROUP_COL].to_numpy(dtype=np.int64)
    ss_list = df["clean_ss"].tolist()
    total = len(df)

    print(f"[INFO] Rows to embed: {total:,}", flush=True)
    print(f"[INFO] Batch size: {batch_size}", flush=True)
    print(f"[INFO] Chunk size: {chunk_size}", flush=True)
    print(f"[INFO] Output dir: {out_dir}", flush=True)

    save_metadata_and_index(df, group_ids, out_dir)

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
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    manifest_rows = []

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_id = start // chunk_size
        expected_rows = end - start

        x_chunk_path = chunks_dir / f"chunk_{chunk_id:06d}_X_ss.npy"
        gid_chunk_path = chunks_dir / f"chunk_{chunk_id:06d}_group_ids.npy"

        if chunk_is_valid(x_chunk_path, gid_chunk_path, expected_rows):
            print(
                f"[SKIP] Chunk {chunk_id:06d} already exists "
                f"({end:,}/{total:,})",
                flush=True,
            )

            manifest_rows.append({
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "n_rows": expected_rows,
                "x_file": str(x_chunk_path),
                "group_ids_file": str(gid_chunk_path),
            })
            continue

        tmp_x_path = x_chunk_path.with_suffix(".tmp.npy")
        tmp_gid_path = gid_chunk_path.with_suffix(".tmp.npy")

        if tmp_x_path.exists():
            tmp_x_path.unlink()
        if tmp_gid_path.exists():
            tmp_gid_path.unlink()

        ss_chunk = ss_list[start:end]
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
                print(
                    f"[INFO] Embedded {global_done:,}/{total:,}",
                    flush=True,
                )

        x_chunk = np.concatenate(x_parts, axis=0).astype(SAVE_DTYPE, copy=False)

        if x_chunk.shape != (expected_rows, WINDOW_SIZE, HIDDEN_SIZE):
            raise RuntimeError(
                f"Bad chunk shape for chunk {chunk_id}: "
                f"{x_chunk.shape}, expected {(expected_rows, WINDOW_SIZE, HIDDEN_SIZE)}"
            )

        np.save(tmp_x_path, x_chunk)
        np.save(tmp_gid_path, gid_chunk)

        tmp_x_path.rename(x_chunk_path)
        tmp_gid_path.rename(gid_chunk_path)

        print(
            f"[OK] Saved chunk {chunk_id:06d}: rows {start:,}-{end:,}",
            flush=True,
        )

        manifest_rows.append({
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "n_rows": expected_rows,
            "x_file": str(x_chunk_path),
            "group_ids_file": str(gid_chunk_path),
        })

        del x_parts
        del x_chunk

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    with open(done_flag, "w") as f:
        f.write("done\n")

    print(f"[OK] Saved manifest: {manifest_path}", flush=True)
    print(f"[OK] Finished: {out_dir}", flush=True)


def find_ss_files(species=None):
    pattern = "**/classification_*_ss.csv"
    files = sorted(DATA_INTERMEDIATE.glob(pattern))

    if species is not None:
        files = [f for f in files if f"/{species}/" in str(f)]

    return files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default=None,
        help="Specific classification_*_ss.csv file to process",
    )

    parser.add_argument(
        "--species",
        choices=["human", "mouse"],
        default=None,
        help="Process only one species if --csv is not provided",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Embedding batch size",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Rows saved per chunk",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing chunk outputs and recompute",
    )

    args = parser.parse_args()

    DATA_INPUT_NN.mkdir(parents=True, exist_ok=True)

    tokenizer, model, device = load_model_and_tokenizer(MODEL_DIR)

    if args.csv is not None:
        files = [Path(args.csv)]
    else:
        files = find_ss_files(species=args.species)

    print(f"[INFO] Files to process: {len(files)}", flush=True)

    for f in files:
        out_dir = build_output_dir(f)

        if args.force and out_dir.exists():
            print(f"[INFO] Force enabled. Removing previous outputs in: {out_dir}", flush=True)

            for item in [
                out_dir / "group_ids.npy",
                out_dir / "metadata.csv",
                out_dir / "group_index.csv",
                out_dir / "chunks_manifest.csv",
                out_dir / "embedding_config.json",
                out_dir / "done.flag",
            ]:
                if item.exists():
                    item.unlink()

            chunks_dir = out_dir / "chunks"
            if chunks_dir.exists:
                for item in chunks_dir.glob("chunk_*"):
                    item.unlink()

        process_ss_csv(
            ss_csv_path=f,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )

    print("\n[OK] All SecondaryBERT embeddings finished.", flush=True)


if __name__ == "__main__":
    main()