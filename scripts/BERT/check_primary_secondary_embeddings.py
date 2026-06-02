#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd


def load_primary_chunks(primary_dir):
    rows = []

    group_files = sorted(glob.glob(os.path.join(primary_dir, "chunk_*_group_id.npy")))

    if not group_files:
        raise FileNotFoundError(f"No se encontraron chunk_*_group_id.npy en {primary_dir}")

    for gf in group_files:
        chunk = os.path.basename(gf).replace("_group_id.npy", "")

        x_file = os.path.join(primary_dir, f"{chunk}_X.npy")
        y_file = os.path.join(primary_dir, f"{chunk}_y.npy")
        pos_file = os.path.join(primary_dir, f"{chunk}_pos.npy")
        meta_file = os.path.join(primary_dir, f"{chunk}_metadata.csv")

        group_ids = np.load(gf, allow_pickle=True)

        x_shape = None
        y_shape = None
        pos_shape = None
        n_meta = None

        if os.path.exists(x_file):
            x_shape = np.load(x_file, mmap_mode="r").shape

        if os.path.exists(y_file):
            y_shape = np.load(y_file, mmap_mode="r").shape

        if os.path.exists(pos_file):
            pos_shape = np.load(pos_file, mmap_mode="r").shape

        if os.path.exists(meta_file):
            n_meta = len(pd.read_csv(meta_file))

        for gid in group_ids:
            rows.append({
                "group_id": str(gid),
                "primary_chunk": chunk,
                "primary_X_shape": str(x_shape),
                "primary_y_shape": str(y_shape),
                "primary_pos_shape": str(pos_shape),
                "primary_metadata_rows": n_meta,
            })

    return pd.DataFrame(rows)


def load_secondary_chunks(secondary_dir):
    chunks_dir = os.path.join(secondary_dir, "chunks")

    rows = []

    group_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*_group_ids.npy")))

    if not group_files:
        raise FileNotFoundError(f"No se encontraron chunk_*_group_ids.npy en {chunks_dir}")

    for gf in group_files:
        chunk = os.path.basename(gf).replace("_group_ids.npy", "")

        x_file = os.path.join(chunks_dir, f"{chunk}_X_ss.npy")

        group_ids = np.load(gf, allow_pickle=True)

        x_shape = None
        if os.path.exists(x_file):
            x_shape = np.load(x_file, mmap_mode="r").shape

        for gid in group_ids:
            rows.append({
                "group_id": str(gid),
                "secondary_chunk": chunk,
                "secondary_X_shape": str(x_shape),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare primary ProtT5/BLOSUM embeddings with SecondaryBERT embeddings by group_id."
    )

    parser.add_argument(
        "--primary-dir",
        required=True,
        help="Directory containing primary embedding chunks."
    )

    parser.add_argument(
        "--secondary-dir",
        required=True,
        help="Directory containing SecondaryBERT_by_group output."
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for reports."
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("\nLoading primary embeddings...")
    primary_df = load_primary_chunks(args.primary_dir)

    print("Loading secondary embeddings...")
    secondary_df = load_secondary_chunks(args.secondary_dir)

    primary_ids = set(primary_df["group_id"])
    secondary_ids = set(secondary_df["group_id"])

    common_ids = primary_ids & secondary_ids
    only_primary = primary_ids - secondary_ids
    only_secondary = secondary_ids - primary_ids

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print(f"Primary rows:              {len(primary_df)}")
    print(f"Primary unique group_ids:  {len(primary_ids)}")
    print(f"Secondary rows:            {len(secondary_df)}")
    print(f"Secondary unique group_ids:{len(secondary_ids)}")
    print(f"Common group_ids:          {len(common_ids)}")
    print(f"Only primary:              {len(only_primary)}")
    print(f"Only secondary:            {len(only_secondary)}")

    if len(primary_ids) > 0:
        print(f"Loss vs primary:           {len(only_primary) / len(primary_ids) * 100:.2f}%")

    merged = primary_df.merge(
        secondary_df,
        on="group_id",
        how="outer",
        indicator=True
    )

    common_df = merged[merged["_merge"] == "both"].copy()
    only_primary_df = merged[merged["_merge"] == "left_only"].copy()
    only_secondary_df = merged[merged["_merge"] == "right_only"].copy()

    primary_df.to_csv(os.path.join(args.outdir, "primary_group_ids.csv"), index=False)
    secondary_df.to_csv(os.path.join(args.outdir, "secondary_group_ids.csv"), index=False)
    merged.to_csv(os.path.join(args.outdir, "primary_secondary_merged_status.csv"), index=False)
    common_df.to_csv(os.path.join(args.outdir, "common_group_ids.csv"), index=False)
    only_primary_df.to_csv(os.path.join(args.outdir, "only_primary_missing_secondary.csv"), index=False)
    only_secondary_df.to_csv(os.path.join(args.outdir, "only_secondary_missing_primary.csv"), index=False)

    summary = pd.DataFrame([{
        "primary_rows": len(primary_df),
        "primary_unique_group_ids": len(primary_ids),
        "secondary_rows": len(secondary_df),
        "secondary_unique_group_ids": len(secondary_ids),
        "common_group_ids": len(common_ids),
        "only_primary_missing_secondary": len(only_primary),
        "only_secondary_missing_primary": len(only_secondary),
        "loss_vs_primary_percent": round(len(only_primary) / len(primary_ids) * 100, 4) if primary_ids else None,
    }])

    summary.to_csv(os.path.join(args.outdir, "summary.csv"), index=False)

    print("\nReports saved in:")
    print(args.outdir)


if __name__ == "__main__":
    main()