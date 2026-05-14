#!/usr/bin/env python3
from pathlib import Path
import math
import re

DATA_ROOT = Path("/home/nap/lperez_nn/data/data_intermediate")
CHUNK_SIZE = 100000

# Logs actuales
LOG_TO_NAME = {
    "hlaA_gpu0.log": "human/mhc-I/HLA-A",
    "hlaB_gpu1.log": "human/mhc-I/HLA-B",
    "hlaC_gpu2.log": "human/mhc-I/HLA-C",
    "hlaDP_gpu3.log": "human/mhc-II/HLA-DP",
}

CSV_MAP = {
    "human/mhc-I/HLA-A": DATA_ROOT / "human/mhc-I/HLA-A/classification_human_mhc-I_HLA-A_blosum.csv",
    "human/mhc-I/HLA-B": DATA_ROOT / "human/mhc-I/HLA-B/classification_human_mhc-I_HLA-B_blosum.csv",
    "human/mhc-I/HLA-C": DATA_ROOT / "human/mhc-I/HLA-C/classification_human_mhc-I_HLA-C_blosum.csv",
    "human/mhc-II/HLA-DP": DATA_ROOT / "human/mhc-II/HLA-DP/classification_human_mhc-II_HLA-DP_blosum.csv",
    "human/mhc-II/HLA-DQ": DATA_ROOT / "human/mhc-II/HLA-DQ/classification_human_mhc-II_HLA-DQ_blosum.csv",
    "human/mhc-II/HLA-DR": DATA_ROOT / "human/mhc-II/HLA-DR/classification_human_mhc-II_HLA-DR_blosum.csv",
    "mouse/mhc-I/H2-D": DATA_ROOT / "mouse/mhc-I/H2-D/classification_mouse_mhc-I_H2-D_blosum.csv",
    "mouse/mhc-I/H2-K": DATA_ROOT / "mouse/mhc-I/H2-K/classification_mouse_mhc-I_H2-K_blosum.csv",
    "mouse/mhc-I/H2-L": DATA_ROOT / "mouse/mhc-I/H2-L/classification_mouse_mhc-I_H2-L_blosum.csv",
    "mouse/mhc-II/H2-IA": DATA_ROOT / "mouse/mhc-II/H2-IA/classification_mouse_mhc-II_H2-IA_blosum.csv",
    "mouse/mhc-II/H2-IE": DATA_ROOT / "mouse/mhc-II/H2-IE/classification_mouse_mhc-II_H2-IE_blosum.csv",
}

# Último chunk guardado según tus logs actuales
CURRENT_SAVED = {
    "human/mhc-I/HLA-A": 1883,
    "human/mhc-I/HLA-B": 268,
    "human/mhc-I/HLA-C": 293,
    "human/mhc-II/HLA-DP": 248,
}

def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)

def total_chunks_from_lines(n_lines: int, chunk_size: int = CHUNK_SIZE) -> int:
    data_rows = max(n_lines - 1, 0)  # quitamos cabecera
    return math.ceil(data_rows / chunk_size)

def main():
    rows = []
    total_human = 0
    total_mouse = 0
    remaining_human = 0
    remaining_mouse = 0

    print("=" * 120)
    print(f"{'Dataset':35} {'Lines':>12} {'Data rows':>12} {'Chunks':>8} {'Done now':>10} {'Remain':>10}")
    print("=" * 120)

    for name, csv_path in CSV_MAP.items():
        n_lines = count_lines(csv_path)
        data_rows = max(n_lines - 1, 0)
        chunks = total_chunks_from_lines(n_lines)

        done_now = 0
        if name in CURRENT_SAVED:
            done_now = CURRENT_SAVED[name] + 1  # chunk 000268 => 269 chunks completos

        remain = max(chunks - done_now, 0)

        print(f"{name:35} {n_lines:12d} {data_rows:12d} {chunks:8d} {done_now:10d} {remain:10d}")

        if name.startswith("human/"):
            total_human += chunks
            remaining_human += remain if name in CURRENT_SAVED else chunks
        else:
            total_mouse += chunks
            remaining_mouse += remain if name in CURRENT_SAVED else chunks

    print("=" * 120)
    print(f"Total human chunks:   {total_human}")
    print(f"Remaining human:      {remaining_human}")
    print(f"Total mouse chunks:   {total_mouse}")
    print(f"Remaining mouse:      {remaining_mouse}")
    print(f"Total remaining all:  {remaining_human + remaining_mouse}")
    print("=" * 120)

if __name__ == "__main__":
    main()
