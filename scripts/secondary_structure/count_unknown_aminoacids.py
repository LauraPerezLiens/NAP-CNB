#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def fasta_to_sequence(fasta_text: str) -> str:
    if pd.isna(fasta_text) or not fasta_text:
        return ""
    lines = str(fasta_text).splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq.strip().upper()

def classify_invalid_residues(seq: str):
    invalid = [aa for aa in seq if aa not in VALID_AA]
    return Counter(invalid)

def analyze_file(csv_path: str):
    df = pd.read_csv(csv_path)

    if "fasta" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'fasta'")

    if "protein_url" not in df.columns:
        raise ValueError("El CSV debe contener la columna 'protein_url'")

    df["sequence"] = df["fasta"].apply(fasta_to_sequence)

    total_proteins = 0
    valid_proteins = 0
    invalid_proteins = 0

    invalid_residue_counts = Counter()          # nº total de residuos inválidos
    proteins_by_invalid_residue = Counter()     # nº de proteínas que contienen cada residuo
    proteins_by_invalid_set = Counter()         # combinaciones, ej. ('X',), ('X','U')
    invalid_examples = defaultdict(list)        # ejemplos por residuo

    for row in df.itertuples(index=False):
        protein_url = str(row.protein_url).strip()
        seq = row.sequence

        if not protein_url or not seq:
            continue

        total_proteins += 1
        invalid_counter = classify_invalid_residues(seq)

        if not invalid_counter:
            valid_proteins += 1
            continue

        invalid_proteins += 1

        invalid_residue_counts.update(invalid_counter)

        invalid_types_in_protein = tuple(sorted(invalid_counter.keys()))
        proteins_by_invalid_set[invalid_types_in_protein] += 1

        for aa in invalid_counter:
            proteins_by_invalid_residue[aa] += 1
            if len(invalid_examples[aa]) < 10:
                invalid_examples[aa].append(protein_url)

    print("=" * 60)
    print(f"Archivo: {csv_path}")
    print("=" * 60)
    print(f"Total proteínas analizadas: {total_proteins}")
    print(f"Proteínas totalmente válidas: {valid_proteins}")
    print(f"Proteínas con residuos no estándar: {invalid_proteins}")
    if total_proteins > 0:
        print(f"% proteínas con residuos no estándar: {100 * invalid_proteins / total_proteins:.2f}%")

    print("\n--- Recuento por residuo no estándar (proteínas que lo contienen) ---")
    if proteins_by_invalid_residue:
        for aa, n in proteins_by_invalid_residue.most_common():
            print(f"{aa}: {n} proteínas")
    else:
        print("No se encontraron residuos no estándar.")

    print("\n--- Recuento total de residuos no estándar ---")
    if invalid_residue_counts:
        for aa, n in invalid_residue_counts.most_common():
            print(f"{aa}: {n} ocurrencias totales")
    else:
        print("No se encontraron residuos no estándar.")

    print("\n--- Combinaciones de residuos no estándar por proteína ---")
    if proteins_by_invalid_set:
        for combo, n in proteins_by_invalid_set.most_common():
            combo_str = ",".join(combo)
            print(f"{combo_str}: {n} proteínas")
    else:
        print("No hay combinaciones problemáticas.")

    print("\n--- Cuántas eliminarías según el criterio ---")
    # 1) eliminar cualquier proteína con cualquier residuo no estándar
    remove_any_invalid = invalid_proteins
    print(f"Eliminar si hay cualquier residuo no estándar: {remove_any_invalid}")

    # 2) eliminar solo proteínas con X
    remove_if_X = proteins_by_invalid_residue.get("X", 0)
    print(f"Eliminar si hay X: {remove_if_X}")

    # 3) eliminar solo proteínas con U
    remove_if_U = proteins_by_invalid_residue.get("U", 0)
    print(f"Eliminar si hay U: {remove_if_U}")

    # 4) eliminar si hay alguno de X,U,O,B,Z,J
    special_set = {"X", "U", "O", "B", "Z", "J"}
    remove_special = 0
    for combo, n in proteins_by_invalid_set.items():
        if any(aa in special_set for aa in combo):
            remove_special += n
    print(f"Eliminar si hay alguno de X/U/O/B/Z/J: {remove_special}")

    print("\n--- Ejemplos por residuo ---")
    for aa, urls in sorted(invalid_examples.items()):
        print(f"{aa}:")
        for u in urls:
            print(f"  - {u}")

if __name__ == "__main__":
    analyze_file("/home/nap/lperez_nn/data/data_intermediate/human_parent_protein_fasta.csv")
    print("\n\n")
    analyze_file("/home/nap/lperez_nn/data/data_intermediate/mouse_parent_protein_fasta.csv")