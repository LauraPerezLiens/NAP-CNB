# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from optparse import OptionParser
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ⚡ Forzar CPU si se desea, o usar GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'

# ======================================================
# CONSTANTES
# ======================================================
models_folder = "/home/lperez/Escritorio/PhD/NAP/hertz-tres-data/tools/models_proteinUnet/models_proteinUnet"

SS_LIST = ["C", "H", "E", "T", "G", "S", "I", "B"]
FASTA_RESIDUE_LIST = ["A","D","N","R","C","E","Q","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
NB_RESIDUES = len(FASTA_RESIDUE_LIST)
RESIDUE_DICT = dict(zip(FASTA_RESIDUE_LIST, range(NB_RESIDUES)))
UPPER_LENGTH_LIMIT = 1024

# ======================================================
# CARGAR MODELO
# ======================================================
# ⚡ Se asume modelo en SavedModel + Keras 2.x
ensemble_c_model = keras.models.load_model(
    os.path.join(models_folder, "unet_c_ensemble")
)

# ======================================================
# FUNCIONES AUXILIARES
# ======================================================
def split_seq(inpath, outpath):
    with open(outpath, 'w') as file1:
        with open(inpath, 'r') as f:
            for line in f:
                for char in line.strip():
                    file1.write(char + " ")
                file1.write('\n')

def fill_array_with_value(array: np.array, length_limit: int, value):
    array_length = len(array)
    filler = value * np.ones((length_limit - array_length, array.shape[1]), array.dtype)
    return np.concatenate((array, filler))

def read_input(input_data):
    protein_names = [input_data[0]]
    sequences = [input_data[1]]
    return protein_names, sequences

def save_predictions(resnames, pred_c, file2):
    sequence_length = len(resnames)

    # Normaliza salida
    if hasattr(pred_c, "numpy"):
        pred_c = pred_c.numpy()
    if isinstance(pred_c, list):
        pred_c = pred_c[0]
    pred_c = np.array(pred_c)

    # Convierte one-hot a SS
    # ⚡ Aplanar primero si tiene dimensión extra
    if pred_c.ndim == 3:  # (1, seq_len, nb_classes)
        pred_c = pred_c[0]
    indices = np.argmax(pred_c, axis=-1)
    list_predictions = [SS_LIST[int(idx)] for idx in indices][:sequence_length]

    file2.write("".join(list_predictions) + "\n")

# ======================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN
# ======================================================
def main_prediction(input_data, file2):
    protein_names, residue_lists = read_input(input_data)
    for protein_name, resnames in zip(protein_names, residue_lists):
        if len(resnames) > UPPER_LENGTH_LIMIT:
            print(f"Sequence longer than {UPPER_LENGTH_LIMIT} residues!")
            continue
        residue_valid = "".join([r for r in resnames if r in RESIDUE_DICT])
        sequence = to_categorical([RESIDUE_DICT[r] for r in residue_valid], num_classes=NB_RESIDUES)
        sequence = fill_array_with_value(sequence, UPPER_LENGTH_LIMIT, 0)
        pred_c = ensemble_c_model.predict(np.array([sequence]))
        save_predictions(resnames, pred_c, file2)

# ======================================================
# SCRIPT PRINCIPAL
# ======================================================
if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help="Input CSV/TSV file", metavar="FILE")
    parser.add_option("-o", "--output", dest="output", help="Output file", metavar="FILE")
    options, args = parser.parse_args()
    INpath = options.input
    OUTpath = options.output

    if INpath is None or OUTpath is None:
        raise ValueError("Debes proporcionar input y output.")

    df = pd.read_csv(INpath)
    sequences = df['25aa_seq'].tolist()

    file_dir = os.path.dirname(INpath)
    SEQpath = os.path.join(file_dir, 'input_secondary_pred.txt')
    SPpath = os.path.join(file_dir, 'proteins_seq.txt')
    SEC_NOSPpath = os.path.join(file_dir, 'secondary_pred_nospace.txt')

    with open(SEQpath, 'w') as f:
        for seq in sequences:
            f.write(seq + "\n")

    split_seq(SEQpath, SPpath)

    with open(SEQpath, 'r') as f_in, open(SEC_NOSPpath, 'w') as f_out:
        data = f_in.read().splitlines()
        for count, seq in enumerate(data):
            main_prediction([count, seq], f_out)

    split_seq(SEC_NOSPpath, OUTpath)
    print("Done secondary struct prediction")