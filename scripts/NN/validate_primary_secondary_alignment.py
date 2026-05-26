#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


PRIMARY_X_DIM = 1024
SECONDARY_WINDOW = 25
SECONDARY_DIM = 768


def load_secondary_index(secondary_dir: Path):
    group_ids_path = secondary_dir / "group_ids.npy"
    manifest_path = secondary_dir / "chunks_manifest.csv"
    chunks_dir = secondary_dir / "chunks"

    if not group_ids_path.exists():
        raise FileNotFoundError(f"Missing: {group_ids_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing: {manifest_path}")

    if not chunks_dir.exists():
        raise FileNotFoundError(f"Missing: {chunks_dir}")

    secondary_group_ids = np.load(group_ids_path, mmap_mode="r")
    manifest = pd.read_csv(manifest_path)

    print(f"[INFO] Secondary group_ids: {secondary_group_ids.shape}")
    print(f"[INFO] Secondary chunks in manifest: {len(manifest)}")

    gid_to_row = {int(g): i for i, g in enumerate(secondary_group_ids)}

    if len(gid_to_row) != len(secondary_group_ids):
        print("[WARN] Duplicate group_ids found in secondary group_ids.npy")

    return secondary_group_ids, gid_to_row, manifest


def validate_secondary_chunks(secondary_dir: Path, manifest: pd.DataFrame, max_chunks=None):
    print("\n" + "=" * 80)
    print("[CHECK] SecondaryBERT chunks")
    print("=" * 80)

    checked = 0

    for _, row in manifest.iterrows():
        if max_chunks is not None and checked >= max_chunks:
            break

        chunk_id = int(row["chunk_id"])
        expected_rows = int(row["n_rows"])

        x_path = Path(row["x_file"])
        gid_path = Path(row["group_ids_file"])

        if not x_path.exists():
            print(f"[ERROR] Missing secondary X chunk: {x_path}")
            continue

        if not gid_path.exists():
            print(f"[ERROR] Missing secondary group_ids chunk: {gid_path}")
            continue

        x = np.load(x_path, mmap_mode="r")
        gids = np.load(gid_path, mmap_mode="r")

        ok_shape = x.shape == (expected_rows, SECONDARY_WINDOW, SECONDARY_DIM)
        ok_gid = gids.shape == (expected_rows,)

        print(
            f"chunk_{chunk_id:06d} | "
            f"X={x.shape} | gids={gids.shape} | "
            f"shape_ok={ok_shape} | gids_ok={ok_gid}"
        )

        if not ok_shape:
            print(f"[ERROR] Bad secondary X shape in {x_path}")

        if not ok_gid:
            print(f"[ERROR] Bad secondary group_ids shape in {gid_path}")

        checked += 1

    print(f"[INFO] Checked secondary chunks: {checked}")


def primary_chunk_ids(primary_dir: Path):
    files = sorted(primary_dir.glob("chunk_*_X.npy"))
    ids = []

    for f in files:
        name = f.name
        chunk_id = int(name.split("_")[1])
        ids.append(chunk_id)

    return ids


def validate_primary_secondary(primary_dir: Path, secondary_dir: Path, max_chunks=None):
    print("=" * 80)
    print("[INFO] Primary dir:")
    print(primary_dir)
    print("[INFO] Secondary dir:")
    print(secondary_dir)
    print("=" * 80)

    secondary_group_ids, gid_to_row, manifest = load_secondary_index(secondary_dir)

    validate_secondary_chunks(
        secondary_dir=secondary_dir,
        manifest=manifest,
        max_chunks=5,
    )

    chunk_ids = primary_chunk_ids(primary_dir)

    if not chunk_ids:
        raise RuntimeError(f"No primary chunks found in: {primary_dir}")

    print("\n" + "=" * 80)
    print("[CHECK] Primary ProtT5 chunks + alignment")
    print("=" * 80)
    print(f"[INFO] Primary chunks found: {len(chunk_ids)}")
    print(f"[INFO] First chunk id: {chunk_ids[0]}")
    print(f"[INFO] Last chunk id: {chunk_ids[-1]}")

    total_rows = 0
    total_missing_secondary = 0
    total_nan_primary = 0
    total_nan_pos = 0
    total_pos = 0
    total_neg = 0

    checked_chunks = 0

    for chunk_id in chunk_ids:
        if max_chunks is not None and checked_chunks >= max_chunks:
            break

        prefix = primary_dir / f"chunk_{chunk_id:06d}"

        x_path = Path(f"{prefix}_X.npy")
        y_path = Path(f"{prefix}_y.npy")
        pos_path = Path(f"{prefix}_pos.npy")
        gid_path = Path(f"{prefix}_group_id.npy")

        missing_files = [
            p for p in [x_path, y_path, pos_path, gid_path]
            if not p.exists()
        ]

        if missing_files:
            print(f"[ERROR] Missing files for chunk {chunk_id:06d}:")
            for p in missing_files:
                print(f"  - {p}")
            continue

        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
        pos = np.load(pos_path, mmap_mode="r")
        gids = np.load(gid_path, mmap_mode="r")

        n = x.shape[0]

        ok_x = len(x.shape) == 2 and x.shape[1] == PRIMARY_X_DIM
        ok_y = y.shape == (n,)
        ok_pos = pos.shape == (n,)
        ok_gid = gids.shape == (n,)

        if not ok_x or not ok_y or not ok_pos or not ok_gid:
            print(f"[ERROR] Bad shapes in primary chunk {chunk_id:06d}")
            print(f"  X:   {x.shape}")
            print(f"  y:   {y.shape}")
            print(f"  pos: {pos.shape}")
            print(f"  gid: {gids.shape}")
            continue

        missing_secondary = sum(int(g) not in gid_to_row for g in gids)

        y_values, y_counts = np.unique(y, return_counts=True)
        y_count_dict = dict(zip(y_values.tolist(), y_counts.tolist()))

        n_pos = int(y_count_dict.get(1, 0))
        n_neg = int(y_count_dict.get(0, 0))

        nan_primary = int(np.isnan(x).sum())
        nan_pos = int(np.isnan(pos).sum())

        total_rows += n
        total_missing_secondary += missing_secondary
        total_nan_primary += nan_primary
        total_nan_pos += nan_pos
        total_pos += n_pos
        total_neg += n_neg

        print(
            f"chunk_{chunk_id:06d} | "
            f"X={x.shape} | y={y.shape} | pos={pos.shape} | gid={gids.shape} | "
            f"pos={n_pos} neg={n_neg} | "
            f"missing_secondary={missing_secondary} | "
            f"nan_X={nan_primary} nan_pos={nan_pos}"
        )

        checked_chunks += 1

    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print("=" * 80)
    print(f"Checked primary chunks: {checked_chunks}")
    print(f"Checked rows: {total_rows:,}")
    print(f"Labels positive: {total_pos:,}")
    print(f"Labels negative: {total_neg:,}")
    print(f"Missing secondary group_ids: {total_missing_secondary:,}")
    print(f"NaNs in primary X: {total_nan_primary:,}")
    print(f"NaNs in pos: {total_nan_pos:,}")

    if total_rows > 0:
        print(f"Positive ratio: {total_pos / total_rows:.6f}")
        print(f"Missing secondary ratio: {total_missing_secondary / total_rows:.6f}")

    if total_missing_secondary == 0 and total_nan_primary == 0 and total_nan_pos == 0:
        print("\n[OK] Validation passed for checked chunks.")
    else:
        print("\n[WARN] Validation found issues. Review summary above.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--primary-dir",
        required=True,
        help="Directory with ProtT5 chunks.",
    )

    parser.add_argument(
        "--secondary-dir",
        required=True,
        help="Directory with SecondaryBERT_by_group output.",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Only validate the first N primary chunks. Omit to validate all.",
    )

    args = parser.parse_args()

    validate_primary_secondary(
        primary_dir=Path(args.primary_dir),
        secondary_dir=Path(args.secondary_dir),
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
