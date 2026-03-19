#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
from pathlib import Path


DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")

AA_BLOCK_RE = re.compile(r"([A-Z]+)")


def canonical_epitope(ep_name: str) -> str:
    if not ep_name:
        return ""
    ep_name = ep_name.strip().upper()
    m = AA_BLOCK_RE.search(ep_name)
    return m.group(1) if m else ""


def get_value(row, key):
    if key == "source_id":
        return (row.get("epitope__source_molecule_iri") or "").strip()

    if key == "parent_id":
        return (row.get("epitope__molecule_parent_iri") or "").strip()

    if key == "epitope":
        return (row.get("epitope__name") or "").strip()

    if key == "start":
        return (row.get("epitope__starting_position") or "").strip()

    if key == "end":
        return (row.get("epitope__ending_position") or "").strip()

    return ""


def load_rows(path: Path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rebuild_from_raw(out_dir: Path) -> None:
    mhc_file = out_dir / "mhc_export_full.csv"
    tcell_file = out_dir / "tcell_export_full.csv"

    rows = load_rows(mhc_file) + load_rows(tcell_file)

    seen = set()
    result = []

    for row in rows:
        epi = canonical_epitope(get_value(row, "epitope"))
        if not epi:
            continue

        start = get_value(row, "start")
        end = get_value(row, "end")
        if not start or not end:
            continue

        source_id = get_value(row, "source_id")
        parent_id = get_value(row, "parent_id")
        protein_id = parent_id

        key = (epi, start, end, source_id, parent_id)

        if key not in seen:
            seen.add(key)
            result.append((epi, start, end, source_id, parent_id, protein_id))

    out_file = out_dir / "merged_unique_events_rebuilt.csv"

    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epitope__name_canonical",
            "epitope__starting_position",
            "epitope__ending_position",
            "source_id",
            "parent_id",
            "protein_id",
        ])
        w.writerows(result)

    print(f"[OK] {out_file} n={len(result)}")


def main():
    for species_dir in sorted(DATA_RAW.iterdir()):
        if not species_dir.is_dir():
            continue

        for class_dir in sorted(species_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for haplo_dir in sorted(class_dir.iterdir()):
                if not haplo_dir.is_dir():
                    continue

                print(f"[INFO] Procesando {haplo_dir}")
                rebuild_from_raw(haplo_dir)


if __name__ == "__main__":
    main()