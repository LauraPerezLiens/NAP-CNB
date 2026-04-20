#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")

SECONDARY_FILES = {
    "human": DATA_INTERMEDIATE / "secondary_struct_human.csv",
    "mouse": DATA_INTERMEDIATE / "secondary_struct_mouse.csv",
}

SPECIES = ["human", "mouse"]
MHC_CLASSES = ["mhc-I", "mhc-II"]
WINDOW_SIZE = 25


def normalize_url(url: str) -> str:
    if pd.isna(url):
        return ""
    return str(url).strip().replace("https://", "http://")


def build_ss_lookup(ss_csv: Path) -> dict:
    df = pd.read_csv(ss_csv)

    required = {"protein_url", "sequence", "secondary_structure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ss_csv}: faltan columnas {missing}")

    lookup = {}

    for _, row in df.iterrows():
        protein_url = normalize_url(row["protein_url"])
        seq = "" if pd.isna(row["sequence"]) else str(row["sequence"]).strip()
        ss = "" if pd.isna(row["secondary_structure"]) else str(row["secondary_structure"]).strip()

        if not protein_url or not seq or not ss:
            continue

        if len(seq) != len(ss):
            continue

        lookup[protein_url] = {
            "sequence": seq,
            "secondary_structure": ss,
        }

    return lookup


def add_ss_to_normal_classification(class_df: pd.DataFrame, ss_lookup: dict, window_size: int) -> pd.DataFrame:
    required = {"25aa_seq", "protein_url", "window_start"}
    missing = required - set(class_df.columns)
    if missing:
        raise ValueError(f"faltan columnas en classification normal: {missing}")

    out = class_df.copy()

    if "group_id" not in out.columns:
        out.insert(0, "group_id", range(1, len(out) + 1))

    ss_windows = []
    ss_status = []
    protein_found = []
    sequence_match = []

    for _, row in out.iterrows():
        seq_window_expected = "" if pd.isna(row["25aa_seq"]) else str(row["25aa_seq"]).strip()
        protein_url = normalize_url(row["protein_url"])

        try:
            window_start = int(row["window_start"])
        except Exception:
            window_start = None

        if protein_url not in ss_lookup:
            ss_windows.append(None)
            ss_status.append("protein_not_found")
            protein_found.append(0)
            sequence_match.append(None)
            continue

        protein_found.append(1)

        full_seq = ss_lookup[protein_url]["sequence"]
        full_ss = ss_lookup[protein_url]["secondary_structure"]

        if window_start is None or window_start < 1:
            ss_windows.append(None)
            ss_status.append("invalid_window_start")
            sequence_match.append(None)
            continue

        start_idx = window_start - 1
        end_idx = start_idx + window_size

        if end_idx > len(full_seq):
            ss_windows.append(None)
            ss_status.append("window_out_of_range")
            sequence_match.append(None)
            continue

        seq_window_real = full_seq[start_idx:end_idx]
        ss_window = full_ss[start_idx:end_idx]

        if seq_window_real != seq_window_expected:
            ss_windows.append(None)
            ss_status.append("sequence_mismatch")
            sequence_match.append(0)
            continue

        ss_windows.append(ss_window)
        ss_status.append("ok")
        sequence_match.append(1)

    out["secondary_structure"] = ss_windows
    out["protein_found"] = protein_found
    out["sequence_match"] = sequence_match
    out["ss_status"] = ss_status

    return out


def propagate_ss_to_blosum(blosum_df: pd.DataFrame, normal_df: pd.DataFrame) -> pd.DataFrame:
    required_blosum = {"group_id"}
    missing_blosum = required_blosum - set(blosum_df.columns)
    if missing_blosum:
        raise ValueError(f"faltan columnas en blosum: {missing_blosum}")

    required_normal = {"group_id", "secondary_structure", "ss_status"}
    missing_normal = required_normal - set(normal_df.columns)
    if missing_normal:
        raise ValueError(f"faltan columnas en normal_df: {missing_normal}")

    mapping = normal_df[["group_id", "secondary_structure", "ss_status"]].drop_duplicates(subset=["group_id"])
    out = blosum_df.merge(mapping, on="group_id", how="left")
    return out


def summarize_status(df: pd.DataFrame) -> str:
    if "ss_status" not in df.columns:
        return "sin columna ss_status"

    counts = df["ss_status"].value_counts(dropna=False).to_dict()
    return ", ".join(f"{k}={v}" for k, v in counts.items())


def process_one_haplotype(species: str, mhc_class: str, haplotype_dir: Path, ss_lookup: dict):
    normal_csvs = sorted(
        p for p in haplotype_dir.glob("classification_*.csv")
        if "_blosum" not in p.stem and not p.stem.endswith("_ss")
    )

    if not normal_csvs:
        print(f"[SKIP] No normal classification in {haplotype_dir}")
        return

    for normal_csv in normal_csvs:
        print("=====================================")
        print(f"Processing: {normal_csv}")
        print("=====================================")

        try:
            class_df = pd.read_csv(normal_csv)
            out_normal = add_ss_to_normal_classification(class_df, ss_lookup, WINDOW_SIZE)

            out_normal_csv = normal_csv.with_name(normal_csv.stem + "_ss.csv")
            out_normal.to_csv(out_normal_csv, index=False)

            print(f"[OK] Normal saved: {out_normal_csv}")
            print(f"     Status: {summarize_status(out_normal)}")

            blosum_csv = normal_csv.with_name(normal_csv.stem + "_blosum.csv")
            if blosum_csv.exists():
                blosum_df = pd.read_csv(blosum_csv)
                out_blosum = propagate_ss_to_blosum(blosum_df, out_normal)

                out_blosum_csv = blosum_csv.with_name(blosum_csv.stem + "_ss.csv")
                out_blosum.to_csv(out_blosum_csv, index=False)

                print(f"[OK] BLOSUM saved: {out_blosum_csv}")
                print(f"     Rows with SS: {out_blosum['secondary_structure'].notna().sum()} / {len(out_blosum)}")
            else:
                print(f"[INFO] No BLOSUM file found for: {normal_csv.name}")

        except Exception as e:
            print(f"[ERROR] {normal_csv}")
            print(f"        {e}")


def main():
    ss_lookups = {}

    for species in SPECIES:
        ss_csv = SECONDARY_FILES[species]
        if not ss_csv.exists():
            raise FileNotFoundError(f"No existe {ss_csv}")
        print(f"Loading secondary structure for {species}: {ss_csv}")
        ss_lookups[species] = build_ss_lookup(ss_csv)

    for species in SPECIES:
        for mhc_class in MHC_CLASSES:
            class_dir = DATA_INTERMEDIATE / species / mhc_class
            if not class_dir.exists():
                print(f"[SKIP] No existe {class_dir}")
                continue

            haplotype_dirs = sorted([p for p in class_dir.iterdir() if p.is_dir()])
            if not haplotype_dirs:
                print(f"[SKIP] No haplotypes in {class_dir}")
                continue

            for haplotype_dir in haplotype_dirs:
                print("\n#####################################################")
                print(f"Species: {species} | Class: {mhc_class} | Haplotype: {haplotype_dir.name}")
                print("#####################################################")
                process_one_haplotype(
                    species=species,
                    mhc_class=mhc_class,
                    haplotype_dir=haplotype_dir,
                    ss_lookup=ss_lookups[species],
                )


if __name__ == "__main__":
    main()