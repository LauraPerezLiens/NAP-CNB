#!/usr/bin/env python3

import os
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUTDIR = os.path.join(BASE_DIR, "features", "embedding")


# ============================================================
# Argumentos por línea de comandos
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ProtT5 sequence embeddings from CSV (primary/secondary reusable)"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Input CSV file"
    )
    parser.add_argument(
        "--sequence_col",
        default="25aa_seq",
        help="Column with sequences to embed (e.g., 25aa_seq, secondary_struct)"
    )
    parser.add_argument(
        "--output_prefix",
        default="primary",
        help="Prefix for output files (e.g., primary, secondary)"
    )
    parser.add_argument(
        "--outdir",
        default=DEFAULT_OUTDIR,
        help="Output directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (reduce if GPU OOM)"
    )
    parser.add_argument(
        "--model",
        default="Rostlab/prot_t5_xl_uniref50",
        help="ProtT5 model name"
    )
    parser.add_argument(
        "--label_col",
        default="contains_epitope",
        help="Optional label column name. If missing, y file is not generated."
    )
    parser.add_argument(
        "--score_col",
        default="epitope_pos_score",
        help="Optional score column to append as extra feature if available."
    )
    parser.add_argument(
        "--append_label_to_X",
        action="store_true",
        help="Append label column to X when label column exists."
    )
    return parser.parse_args()


# ============================================================
# Utilidades
# ============================================================
def space_amino_acids(seq):
    """Insert spaces between amino acids as required by ProtT5."""
    seq = str(seq).replace(" ", "")
    return " ".join(list(seq))


def mean_pooling(hidden_states, attention_mask):
    """
    Mean pooling with attention mask.
    hidden_states: (B, L, D)
    attention_mask: (B, L)
    """
    mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Cargar CSV
    # --------------------------------------------------------
    df = pd.read_csv(args.csv)

    if args.sequence_col not in df.columns:
        raise ValueError(f"Missing required column: {args.sequence_col}")

    sequences = df[args.sequence_col].astype(str).tolist()

    # --------------------------------------------------------
    # Cargar modelo y tokenizer
    # --------------------------------------------------------
    tokenizer = T5Tokenizer.from_pretrained(
        args.model,
        do_lower_case=False
    )

    model = T5EncoderModel.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    embedding_dim = model.config.d_model
    print(f"ProtT5 embedding dim: {embedding_dim}")

    # --------------------------------------------------------
    # Embeddings por batches
    # --------------------------------------------------------
    all_embeddings = []

    for i in tqdm(range(0, len(sequences), args.batch_size), desc="Computing embeddings"):
        batch_seqs = sequences[i:i + args.batch_size]
        batch_seqs = [space_amino_acids(seq) for seq in batch_seqs]

        inputs = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = mean_pooling(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )

        all_embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(all_embeddings)

    # --------------------------------------------------------
    # Features adicionales
    # --------------------------------------------------------
    feature_blocks = [embeddings]

    has_label = args.label_col in df.columns
    has_score = args.score_col in df.columns

    if args.append_label_to_X and has_label:
        feature_blocks.append(df[args.label_col].values.reshape(-1, 1))

    if has_score:
        feature_blocks.append(df[args.score_col].values.reshape(-1, 1))

    X = np.hstack(feature_blocks)

    y = df[args.label_col].values if has_label else None

    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------
    x_out = os.path.join(args.outdir, f"X_{args.output_prefix}_embeddings.npy")
    np.save(x_out, X)
    if y is not None:
        y_out = os.path.join(args.outdir, f"y_{args.output_prefix}_labels.npy")
        np.save(y_out, y)

    meta_cols = [
        args.sequence_col,
        args.label_col,
        args.score_col,
        "protein_id",
        "window_start"
    ]
    existing_meta_cols = [col for col in dict.fromkeys(meta_cols) if col in df.columns]
    df_meta = df[existing_meta_cols]
    df_meta.to_csv(
        os.path.join(args.outdir, f"metadata_{args.output_prefix}.csv"),
        index=False
    )

    print("\n=== DONE ===")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Final X shape: {X.shape}")
    if y is not None:
        print(f"Labels shape: {y.shape}")
    else:
        print("Labels: not generated (label column not found)")
    print(f"Saved to: {args.outdir}")


# ============================================================
if __name__ == "__main__":
    main()
