#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from optparse import OptionParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =========================
# CONFIG
# =========================

BASE_DIR = "/home/nap/lperez_nn/data/data_uniref50/secondary_structure_prediction"
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

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

os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")

print("[INFO] Loading model...")
ensemble_c_model = keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded")


# =========================
# FASTA PARSER
# =========================

def parse_fasta(fasta_path):
    header = None
    seq_lines = []

    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines).upper()
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)

        if header is not None:
            yield header, "".join(seq_lines).upper()


# =========================
# UTILS
# =========================

def get_invalid_residues(seq: str):
    return sorted(set(r for r in seq if r not in VALID_RESIDUES))


def fill_array_with_value(array: np.ndarray, length_limit: int, value):
    array_length = len(array)
    filler = value * np.ones(
        (length_limit - array_length, array.shape[1]),
        dtype=array.dtype
    )
    return np.concatenate((array, filler))


# =========================
# PREDICTION
# =========================

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


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help="Input FASTA file")
    parser.add_option("-o", "--output", dest="output", help="Output CSV filename")
    options, args = parser.parse_args()

    if not options.input or not options.output:
        raise ValueError("Debes proporcionar --input y --output")

    in_path = options.input
    out_path = os.path.join(OUTPUTS_DIR, options.output)

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Output: {out_path}")

    n_total = 0
    n_long = 0
    n_skipped_invalid = 0
    n_failed = 0
    n_ok = 0

    with open(out_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["id", "protein_id", "sequence", "secondary_structure"])

        for n_total, (header, sequence) in enumerate(parse_fasta(in_path), start=1):
            protein_id = header.split()[0]

            invalid_residues = get_invalid_residues(sequence)
            if invalid_residues:
                n_skipped_invalid += 1
                continue

            if len(sequence) > WINDOW_SIZE:
                n_long += 1

            try:
                secondary_structure = predict_secondary_structure(sequence)
            except Exception as e:
                n_failed += 1
                print(f"[WARN] Failed prediction for {protein_id}: {e}")
                continue

            if not secondary_structure:
                n_failed += 1
                continue

            if len(secondary_structure) != len(sequence):
                n_failed += 1
                print(
                    f"[WARN] Length mismatch for {protein_id}: "
                    f"sequence={len(sequence)} ss={len(secondary_structure)}"
                )
                continue

            n_ok += 1
            writer.writerow([n_ok, protein_id, sequence, secondary_structure])

            if n_total % 100 == 0:
                print(f"[INFO] Processed {n_total} | OK={n_ok}")

    print(f"[INFO] Total proteins read: {n_total}")
    print(f"[INFO] Proteins longer than {WINDOW_SIZE}: {n_long}")
    print(f"[INFO] Proteins skipped due to invalid residues: {n_skipped_invalid}")
    print(f"[INFO] Failed predictions: {n_failed}")
    print(f"[INFO] Proteins successfully predicted: {n_ok}")
    print(f"[OK] Done -> {out_path}")
