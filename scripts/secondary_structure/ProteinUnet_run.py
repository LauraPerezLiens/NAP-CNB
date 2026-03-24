#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from optparse import OptionParser

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

MODELS_FOLDER = "/home/nap/lperez_nn/model"
MODEL_PATH = os.path.join(MODELS_FOLDER, "unet_c_ensemble")

SS_LIST = ["C", "H", "E"]

FASTA_RESIDUE_LIST = [
    "A", "D", "N", "R", "C", "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]
VALID_RESIDUES = set(FASTA_RESIDUE_LIST)

NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))

WINDOW_SIZE = 1024
OVERLAP = 200
STEP = WINDOW_SIZE - OVERLAP

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")

ensemble_c_model = keras.models.load_model(MODEL_PATH)


def fasta_to_sequence(fasta_text: str) -> str:
    if pd.isna(fasta_text) or not fasta_text:
        return ""
    lines = str(fasta_text).splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq.strip().upper()


def get_invalid_residues(seq: str):
    return sorted(set(r for r in seq if r not in VALID_RESIDUES))


def fill_array_with_value(array: np.ndarray, length_limit: int, value):
    array_length = len(array)
    filler = value * np.ones(
        (length_limit - array_length, array.shape[1]),
        dtype=array.dtype
    )
    return np.concatenate((array, filler))


def predict_window_probabilities(seq: str) -> np.ndarray:
    seq = str(seq).strip().upper()
    if not seq:
        return np.empty((0, len(SS_LIST)), dtype=np.float32)

    invalid_residues = get_invalid_residues(seq)
    if invalid_residues:
        raise ValueError(
            f"Secuencia con residuos no estándar: {','.join(invalid_residues)}"
        )

    if len(seq) > WINDOW_SIZE:
        raise ValueError(f"La ventana no puede superar {WINDOW_SIZE} aa")

    sequence = to_categorical(
        [RESIDUE_DICT[r] for r in seq],
        num_classes=NB_RESIDUES
    )

    sequence = fill_array_with_value(sequence, WINDOW_SIZE, 0)
    pred_c = ensemble_c_model.predict(np.array([sequence]), verbose=0)

    if hasattr(pred_c, "numpy"):
        pred_c = pred_c.numpy()
    if isinstance(pred_c, list):
        pred_c = pred_c[0]

    pred_c = np.array(pred_c)

    if pred_c.ndim == 3:
        pred_c = pred_c[0]

    return pred_c[:len(seq)]


def probabilities_to_ss(prob_matrix: np.ndarray) -> str:
    if prob_matrix.size == 0:
        return ""
    indices = np.argmax(prob_matrix, axis=-1)
    return "".join(SS_LIST[int(idx)] for idx in indices)


def predict_secondary_structure(seq: str) -> str:
    seq = str(seq).strip().upper()
    if not seq:
        return ""

    invalid_residues = get_invalid_residues(seq)
    if invalid_residues:
        raise ValueError(
            f"Secuencia con residuos no estándar: {','.join(invalid_residues)}"
        )

    seq_len = len(seq)

    if seq_len <= WINDOW_SIZE:
        prob_matrix = predict_window_probabilities(seq)
        return probabilities_to_ss(prob_matrix)

    sum_probs = np.zeros((seq_len, len(SS_LIST)), dtype=np.float32)
    counts = np.zeros(seq_len, dtype=np.float32)

    starts = list(range(0, seq_len, STEP))
    if starts[-1] + WINDOW_SIZE < seq_len:
        starts.append(seq_len - WINDOW_SIZE)

    starts = sorted(set(min(s, seq_len - WINDOW_SIZE) for s in starts))

    for start in starts:
        end = min(start + WINDOW_SIZE, seq_len)
        window_seq = seq[start:end]
        window_probs = predict_window_probabilities(window_seq)

        win_len = len(window_seq)

        left_trim = 0 if start == 0 else OVERLAP // 2
        right_trim = 0 if end == seq_len else OVERLAP // 2

        usable_start = left_trim
        usable_end = win_len - right_trim

        global_start = start + usable_start
        global_end = start + usable_end

        sum_probs[global_start:global_end] += window_probs[usable_start:usable_end]
        counts[global_start:global_end] += 1.0

    uncovered = np.where(counts == 0)[0]
    if len(uncovered) > 0:
        for start in starts:
            end = min(start + WINDOW_SIZE, seq_len)
            window_seq = seq[start:end]
            window_probs = predict_window_probabilities(window_seq)
            sum_probs[start:end] += window_probs
            counts[start:end] += 1.0

    avg_probs = sum_probs / counts[:, None]
    return probabilities_to_ss(avg_probs)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help="Input CSV file", metavar="FILE")
    parser.add_option("-o", "--output", dest="output", help="Output CSV file", metavar="FILE")
    options, args = parser.parse_args()

    in_path = options.input
    out_path = options.output

    if in_path is None or out_path is None:
        raise ValueError("Debes proporcionar input y output.")

    df = pd.read_csv(in_path)

    required_cols = ["protein_url", "fasta"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"El input debe contener la columna {col}")

    df["protein_url"] = df["protein_url"].astype(str).str.strip()
    df["sequence"] = df["fasta"].apply(fasta_to_sequence)
    df = df[(df["protein_url"] != "") & (df["sequence"] != "")].copy()

    results = []

    total = len(df)
    print(f"[INFO] Total proteins to process: {total}")

    n_long = 0
    n_skipped_invalid = 0
    n_ok = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        protein_url = row.protein_url
        sequence = row.sequence

        invalid_residues = get_invalid_residues(sequence)
        if invalid_residues:
            n_skipped_invalid += 1
            print(
                f"[WARN] Skipping {protein_url} "
                f"(invalid residues: {','.join(invalid_residues)})"
            )
            continue

        if len(sequence) > WINDOW_SIZE:
            n_long += 1

        try:
            secondary_structure = predict_secondary_structure(sequence)
        except Exception as e:
            print(f"[WARN] Failed prediction for {protein_url}: {e}")
            continue

        if not secondary_structure:
            print(f"[WARN] Empty SS output for {protein_url}")
            continue

        if len(secondary_structure) != len(sequence):
            print(
                f"[WARN] Length mismatch for {protein_url}: "
                f"sequence={len(sequence)} ss={len(secondary_structure)}"
            )
            continue

        n_ok += 1

        results.append({
            "id": n_ok,
            "protein_url": protein_url,
            "sequence": sequence,
            "secondary_structure": secondary_structure
        })

        if i % 100 == 0 or i == total:
            print(f"[INFO] Processed {i}/{total}")

    out_df = pd.DataFrame(
        results,
        columns=["id", "protein_url", "sequence", "secondary_structure"]
    )
    out_df.to_csv(out_path, index=False)

    print(f"[INFO] Proteins longer than {WINDOW_SIZE}: {n_long}")
    print(f"[INFO] Proteins skipped due to invalid residues: {n_skipped_invalid}")
    print(f"[INFO] Proteins successfully predicted: {n_ok}")
    print(f"[OK] Done secondary structure prediction -> {out_path}")