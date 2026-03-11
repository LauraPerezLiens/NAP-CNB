#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import logging
import re
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


# Regex: primer bloque de aminoácidos (A-Z). Si hay modificaciones tipo "+ MCM(K4)" se ignora.
AA_BLOCK_RE = re.compile(r"([A-Z]+)")

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
    if key in row and row[key] is not None:
        return str(row[key])
    if key == "antigen_id":
        return str(row.get("epitope__source_molecule_iri", "") or "")
    if key == "protein_id":
        return str(row.get("epitope__molecule_parent_iri", "") or "")
    return ""

def canonical_epitope(ep_name: str) -> str:
    """
    Convierte 'YILKPLPL + MCM(K4)' -> 'YILKPLPL'
    Convierte ' SIINFEKL ' -> 'SIINFEKL'
    Si no encuentra nada, devuelve "".
    """
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

        success = False
        data = None

        for attempt in range(5):
            try:
                r = session.get(base_url, params=p, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                success = True
                break
            except Exception as e:
                wait_s = 2 ** attempt
                logging.warning(
                    "Error en %s page=%d offset=%d intento=%d: %s. Reintentando en %ds...",
                    base_url, page, p["offset"], attempt + 1, e, wait_s
                )
                time.sleep(wait_s)

        if not success:
            logging.error("Fallo definitivo en %s page=%d offset=%d", base_url, page, p["offset"])
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
    all_keys = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

def write_filtered_outputs(rows: List[Dict], out_dir: Path, prefix: str) -> None:
    """
    Genera:
      - <prefix>_unique_epitopes.csv  (1 fila por epítopo canonical)
      - <prefix>_unique_events.csv    (dedupe por epitope canonical + start/end + antigen_id + protein_id)
      - <prefix>_epitope_counts.csv   (conteo por epítopo canonical)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}

    seen_epi = set()
    unique_epitopes: List[Tuple[str, str, str, str, str]] = []

    seen_evt = set()
    unique_events: List[Tuple[str, str, str, str, str]] = []

    for row in rows:
        raw = get_value(row, "epitope__name")
        epi = canonical_epitope(raw)
        if not epi:
            continue

        start = get_value(row, "epitope__starting_position")
        end = get_value(row, "epitope__ending_position")
        antigen = get_value(row, "antigen_id")
        protein = get_value(row, "protein_id")

        counts[epi] = counts.get(epi, 0) + 1

        epi_key = epi
        evt_key = (epi, start, end, antigen, protein)

        if epi_key not in seen_epi:
            seen_epi.add(epi_key)
            unique_epitopes.append((epi, start, end, antigen, protein))

        if evt_key not in seen_evt:
            seen_evt.add(evt_key)
            unique_events.append((epi, start, end, antigen, protein))

    header = [
        "epitope__name_canonical",
        "epitope__starting_position",
        "epitope__ending_position",
        "antigen_id",
        "protein_id",
    ]

    with (out_dir / f"{prefix}_unique_epitopes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(unique_epitopes)

    with (out_dir / f"{prefix}_unique_events.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(unique_events)

    with (out_dir / f"{prefix}_epitope_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epitope__name_canonical", "n_rows_in_export"])
        for epi in sorted(counts.keys()):
            w.writerow([epi, counts[epi]])

    logging.info(
        "%s: unique_epitopes=%d | unique_events=%d | total_rows=%d",
        prefix,
        len(unique_epitopes),
        len(unique_events),
        len(rows),
    )



def fetch_pipeline(species: str, class_type: str, haplotype: str, mode: str, class_num: str, root: Path) -> None:
    base_url = BASE_MHC if mode == "mhc" else BASE_TCELL
    page_limit = 500 if mode == "mhc" else 1000

    params = {
        "epitope__object_type": "eq.Linear peptide",
        "assay__qualitative_measurement": "ilike.*Positive*",
        "mhc_restriction__class": f"eq.{class_num}",
        "host__name": f"ilike.*{SPECIES_QUERY[species]}*",
        "mhc_restriction__name": f"ilike.*{haplotype}*",
        "epitope__source_organism": f"ilike.*{SPECIES_QUERY[species]}*",
        "order": "assay_id.asc",
        "limit": page_limit,
        "offset": 0,
    }

    out_dir = root / species / class_type / haplotype
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=== %s %s %s %s ===", species, class_type, haplotype, mode)
    rows = fetch_all(base_url, params=params)

    full_path = out_dir / f"{mode}_export_full.csv"
    write_full_csv(rows, full_path)

    write_filtered_outputs(rows, out_dir, mode)

def merge_unique_epitopes(out_dir: Path) -> None:
    mhc_u = out_dir / "mhc_unique_epitopes.csv"
    tcell_u = out_dir / "tcell_unique_epitopes.csv"

    merged = {}
    header = [
        "epitope__name_canonical",
        "epitope__starting_position",
        "epitope__ending_position",
        "antigen_id",
        "protein_id",
    ]

    def load_unique(path: Path):
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                epi = row["epitope__name_canonical"]
                merged.setdefault(epi, row)

    load_unique(mhc_u)
    load_unique(tcell_u)

    merged_path = out_dir / "merged_unique_epitopes.csv"
    with merged_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for epi in sorted(merged.keys()):
            w.writerow(merged[epi])

    logging.info("MERGED unique epitopes -> %s (n=%d)", merged_path, len(merged))

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    root = Path("/home/nap/lperez_nn/data/data_raw")

    for species in ["mouse", "human"]:
        for class_type, class_num, haps in [
            ("mhc-I", "I", HAPLOTYPES_I),
            ("mhc-II", "II", HAPLOTYPES_II),
        ]:
            for haplotype in haps[species]:
                fetch_pipeline(species, class_type, haplotype, "mhc", class_num, root)
                fetch_pipeline(species, class_type, haplotype, "tcell", class_num, root)

                out_dir = root / species / class_type / haplotype
                merge_unique_epitopes(out_dir)

    logging.info("DONE")


if __name__ == "__main__":
    main()