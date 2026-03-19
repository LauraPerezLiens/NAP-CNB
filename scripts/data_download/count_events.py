#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from pathlib import Path


DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")


def count_rows(csv_file: Path) -> int:
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # saltar cabecera
        return sum(1 for _ in reader)


def main():
    results = []

    for path in sorted(DATA_RAW.rglob("merged_unique_events.csv")):
        n = count_rows(path)
        results.append((str(path), n))
        print(f"{path} -> {n}")

    total = sum(n for _, n in results)

    print("\n========== TOTAL ==========")
    print(f"Total filas: {total}")


if __name__ == "__main__":
    main()