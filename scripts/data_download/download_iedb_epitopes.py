#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_MHC = "https://query-api.iedb.org/mhc_export"
BASE_TCELL = "https://query-api.iedb.org/tcell_export"

HAPLOTYPES_I = {
    "mouse": ["H2-K", "H2-D", "H2-L"],
    "human": ["HLA-A", "HLA-B", "HLA-C"],
}
HAPLOTYPES_II = {
    "mouse": ["H2-IA", "H2-IE"],
    "human": ["HLA-DR", "HLA-DQ", "HLA-DP"],
}

SPECIES_QUERY = {
    "mouse": "Mus musculus",
    "human": "Homo sapiens",
}

AA_BLOCK_RE = re.compile(r"([A-Z]+)")


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_value(row: Dict, key: str) -> str:
    if key == "source_id":
        return str(row.get("epitope__source_molecule_iri", "") or "").strip()

    if key == "parent_id":
        return str(row.get("epitope__molecule_parent_iri", "") or "").strip()

    if key == "protein_id":
        return str(row.get("epitope__molecule_parent_iri", "") or "").strip()

    if key in row and row[key] is not None:
        return str(row[key]).strip()

    return ""


def canonical_epitope(ep_name: str) -> str:
    if not ep_name:
        return ""
    ep_name = ep_name.strip().upper()
    m = AA_BLOCK_RE.search(ep_name)
    return m.group(1) if m else ""


def fetch_all(base_url: str, params: Dict, timeout: int = 120) -> List[Dict]:
    all_rows: List[Dict] = []
    page = 0
    limit = int(params.get("limit", 1000))
    session = make_session()

    while True:
        p = dict(params)
        p["offset"] = page * limit

        try:
            r = session.get(base_url, params=p, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logging.error("Error en %s offset=%d: %s", base_url, p["offset"], e)
            break

        logging.info(
            "GET %s page=%d offset=%d n=%d",
            base_url.split("/")[-1],
            page,
            p["offset"],
            len(data),
        )

        if not data:
            break

        all_rows.extend(data)
        page += 1

    return all_rows


def write_full_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return

    keys = sorted({k for r in rows for k in r.keys()})

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_merged_unique_events(rows: List[Dict], out_dir: Path) -> None:
    seen = set()
    result = []

    for row in rows:
        epi = canonical_epitope(get_value(row, "epitope__name"))
        if not epi:
            continue

        start = get_value(row, "epitope__starting_position")
        end = get_value(row, "epitope__ending_position")
        if not start or not end:
            continue

        source_id = get_value(row, "source_id")
        parent_id = get_value(row, "parent_id")
        protein_id = parent_id

        key = (epi, start, end, source_id, parent_id)

        if key not in seen:
            seen.add(key)
            result.append((epi, start, end, source_id, parent_id, protein_id))

    out_file = out_dir / "merged_unique_events.csv"

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

    logging.info("MERGED -> %s (n=%d)", out_file, len(result))


def fetch_pipeline(species, class_type, haplotype, mode, class_num):
    base_url = BASE_MHC if mode == "mhc" else BASE_TCELL

    params = {
        "epitope__object_type": "eq.Linear peptide",
        "assay__qualitative_measurement": "ilike.*Positive*",
        "mhc_restriction__class": f"eq.{class_num}",
        "host__name": f"ilike.*{SPECIES_QUERY[species]}*",
        "mhc_restriction__name": f"ilike.*{haplotype}*",
        "epitope__source_organism": f"ilike.*{SPECIES_QUERY[species]}*",
        "limit": 500 if mode == "mhc" else 1000,
        "offset": 0,
    }

    logging.info("=== %s %s %s %s ===", species, class_type, haplotype, mode)
    return fetch_all(base_url, params)


def main():
    setup_logging()
    root = Path("/home/nap/lperez_nn/data/data_raw")

    for species in ["mouse", "human"]:
        for class_type, class_num, haps in [
            ("mhc-I", "I", HAPLOTYPES_I),
            ("mhc-II", "II", HAPLOTYPES_II),
        ]:
            for haplotype in haps[species]:

                out_dir = root / species / class_type / haplotype
                out_dir.mkdir(parents=True, exist_ok=True)

                rows_mhc = fetch_pipeline(species, class_type, haplotype, "mhc", class_num)
                rows_tcell = fetch_pipeline(species, class_type, haplotype, "tcell", class_num)

                write_full_csv(rows_mhc, out_dir / "mhc_export_full.csv")
                write_full_csv(rows_tcell, out_dir / "tcell_export_full.csv")

                build_merged_unique_events(rows_mhc + rows_tcell, out_dir)

    logging.info("DONE")


if __name__ == "__main__":
    main()