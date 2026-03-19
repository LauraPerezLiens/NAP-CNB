#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")
DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")

WINDOW_SIZE = {
    "human": 25,
    "mouse": 12,
}


def fasta_to_sequence(fasta_text: str) -> str:
    if pd.isna(fasta_text) or not fasta_text:
        return ""
    lines = str(fasta_text).splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    return seq.strip().upper()


def build_protein_sequence_map(species: str) -> Dict[str, str]:
    fasta_csv = DATA_INTERMEDIATE / f"{species}_parent_protein_fasta.csv"
    if not fasta_csv.exists():
        raise FileNotFoundError(f"No existe {fasta_csv}")

    df = pd.read_csv(fasta_csv)

    required_cols = ["protein_url", "fasta"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{fasta_csv} debe contener la columna {col}")

    df["sequence"] = df["fasta"].apply(fasta_to_sequence)
    df["protein_url"] = df["protein_url"].astype(str).str.strip()
    df["sequence"] = df["sequence"].astype(str).str.strip()

    df = df[(df["protein_url"] != "") & (df["sequence"] != "")].copy()

    return {
        row.protein_url: row.sequence
        for row in df.itertuples()
    }


def compute_epitope_position_score(
    window_start: int,
    window_end: int,
    ep_start: int,
    ep_end: int,
    window_size: int,
) -> float:
    window_center = math.ceil(window_size / 2)
    max_center_dist = window_center - 1

    ep_center = (ep_start + ep_end) / 2.0
    ep_center_in_window = ep_center - window_start + 1
    signed_dist = ep_center_in_window - window_center

    if max_center_dist == 0:
        return 0.0

    score = min(1.0, abs(signed_dist) / max_center_dist)
    signed_score = math.copysign(score, signed_dist)
    return round(signed_score, 3)


def validate_exact_match(protein_seq: str, epitope_seq: str, ep_start: int, ep_end: int) -> bool:
    if not protein_seq or not epitope_seq:
        return False
    if ep_start < 1 or ep_end < 1 or ep_start > ep_end or ep_end > len(protein_seq):
        return False

    observed = protein_seq[ep_start - 1:ep_end]
    return observed == epitope_seq


def classify_one_haplotype(
    species: str,
    class_type: str,
    haplotype: str,
    seq_map: Dict[str, str],
) -> None:
    window_size = WINDOW_SIZE[species]

    epitope_file = DATA_RAW / species / class_type / haplotype / "merged_unique_events.csv"
    if not epitope_file.exists():
        print(f"[WARN] No existe {epitope_file}")
        return

    print(f"[INFO] Procesando {epitope_file}")

    epitopes_df = pd.read_csv(epitope_file)

    required_cols = [
        "epitope__name_canonical",
        "epitope__starting_position",
        "epitope__ending_position",
        "parent_id",
    ]
    for col in required_cols:
        if col not in epitopes_df.columns:
            raise ValueError(f"Falta la columna {col} en {epitope_file}")

    epitopes_df = epitopes_df.dropna(
        subset=[
            "epitope__name_canonical",
            "epitope__starting_position",
            "epitope__ending_position",
            "parent_id",
        ]
    ).copy()

    epitopes_df["parent_id"] = epitopes_df["parent_id"].astype(str).str.strip()
    epitopes_df["epitope__starting_position"] = pd.to_numeric(
        epitopes_df["epitope__starting_position"], errors="coerce"
    )
    epitopes_df["epitope__ending_position"] = pd.to_numeric(
        epitopes_df["epitope__ending_position"], errors="coerce"
    )
    epitopes_df["epitope__name_canonical"] = (
        epitopes_df["epitope__name_canonical"].astype(str).str.strip().str.upper()
    )

    epitopes_df = epitopes_df.dropna(
        subset=["epitope__starting_position", "epitope__ending_position"]
    ).copy()

    epitopes_df["epitope__starting_position"] = epitopes_df["epitope__starting_position"].astype(int)
    epitopes_df["epitope__ending_position"] = epitopes_df["epitope__ending_position"].astype(int)

    epitopes_df = epitopes_df[
        (epitopes_df["parent_id"] != "") &
        (epitopes_df["epitope__name_canonical"] != "")
    ].copy()

    protein_urls = sorted(set(epitopes_df["parent_id"]))
    rows: List[List] = []

    print(f"[INFO] {species} {class_type} {haplotype}: {len(protein_urls)} proteínas con epítopos")

    for idx, protein_url in enumerate(protein_urls, start=1):
        seq = seq_map.get(protein_url, "")
        if not seq:
            continue

        if len(seq) < window_size:
            continue

        prot_epitopes = epitopes_df[epitopes_df["parent_id"] == protein_url].copy()
        if prot_epitopes.empty:
            continue

        # quedarse solo con epítopos que casan exactamente en la proteína
        prot_epitopes["exact_match"] = prot_epitopes.apply(
            lambda r: validate_exact_match(
                protein_seq=seq,
                epitope_seq=r["epitope__name_canonical"],
                ep_start=r["epitope__starting_position"],
                ep_end=r["epitope__ending_position"],
            ),
            axis=1,
        )
        prot_epitopes = prot_epitopes[prot_epitopes["exact_match"]].copy()

        if prot_epitopes.empty:
            continue

        if idx % 100 == 0 or idx == 1:
            print(f"[INFO] {species} {class_type} {haplotype}: proteína {idx}/{len(protein_urls)}")

        for i in range(len(seq) - window_size + 1):
            window_seq = seq[i:i + window_size]
            window_start = i + 1
            window_end = i + window_size

            positive_hits: List[Tuple[str, float]] = []

            for e in prot_epitopes.itertuples():
                ep_start = e.epitope__starting_position
                ep_end = e.epitope__ending_position
                ep_name = e.epitope__name_canonical

                if window_start <= ep_start and window_end >= ep_end:
                    pos_score = compute_epitope_position_score(
                        window_start=window_start,
                        window_end=window_end,
                        ep_start=ep_start,
                        ep_end=ep_end,
                        window_size=window_size,
                    )
                    positive_hits.append((ep_name, pos_score))

            if positive_hits:
                for ep_name, pos_score in positive_hits:
                    rows.append([
                        window_seq,
                        1,
                        ep_name,
                        pos_score,
                        protein_url,
                        window_start,
                    ])
            else:
                rows.append([
                    window_seq,
                    0,
                    "",
                    0.0,
                    protein_url,
                    window_start,
                ])

    output_cols = [
        f"{window_size}aa_seq",
        "contains_epitope",
        "selected_epitope",
        "epitope_pos_score",
        "protein_url",
        "window_start",
    ]

    output_df = pd.DataFrame(rows, columns=output_cols)
    output_df = output_df.drop_duplicates().reset_index(drop=True)

    out_dir = DATA_INTERMEDIATE / species / class_type / haplotype
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"classification_{species}_{class_type}_{haplotype}.csv"
    output_df.to_csv(out_file, index=False, float_format="%.3f")

    print(f"[OK] Guardado: {out_file}")
    print(f"[OK] Total ventanas: {len(output_df)}")


def main() -> None:
    for species in ["human", "mouse"]:
        print(f"[INFO] Cargando secuencias para {species}")
        seq_map = build_protein_sequence_map(species)

        species_dir = DATA_RAW / species
        if not species_dir.exists():
            print(f"[WARN] No existe {species_dir}")
            continue

        for class_dir in sorted(species_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_type = class_dir.name

            for haplo_dir in sorted(class_dir.iterdir()):
                if not haplo_dir.is_dir():
                    continue

                haplotype = haplo_dir.name
                classify_one_haplotype(
                    species=species,
                    class_type=class_type,
                    haplotype=haplotype,
                    seq_map=seq_map,
                )


if __name__ == "__main__":
    main()