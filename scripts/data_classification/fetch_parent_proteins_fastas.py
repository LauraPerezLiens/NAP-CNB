#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import time
from pathlib import Path
from typing import Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")
DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

SPECIES = ["human", "mouse"]

UNIPROT_RE = re.compile(
    r"https?://(?:www\.)?uniprot\.org/uniprot/([A-Z0-9]+(?:[-.][A-Z0-9]+)?)/*$",
    re.IGNORECASE,
)

NCBI_PROTEIN_RE = re.compile(
    r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/protein/([A-Z0-9_.-]+)/*$",
    re.IGNORECASE,
)


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


def parse_protein_url(url: str) -> Tuple[str, str, str]:
    """
    Devuelve:
    - db: uniprot / ncbi_protein / unknown
    - id_full
    - id_base

    Ejemplos:
    https://www.uniprot.org/uniprot/P47974.3 -> ("uniprot", "P47974.3", "P47974")
    http://www.ncbi.nlm.nih.gov/protein/NP_008818.3 -> ("ncbi_protein", "NP_008818.3", "NP_008818")
    """
    url = str(url).strip()
    if not url:
        return "", "", ""

    m = UNIPROT_RE.match(url)
    if m:
        id_full = m.group(1).upper().strip()
        id_base = re.split(r"[-.]", id_full)[0]
        return "uniprot", id_full, id_base

    m = NCBI_PROTEIN_RE.match(url)
    if m:
        id_full = m.group(1).strip()
        id_base = id_full.split(".")[0]
        return "ncbi_protein", id_full, id_base

    return "unknown", "", ""


def collect_unique_parent_urls(species: str) -> Set[str]:
    urls: Set[str] = set()
    species_dir = DATA_RAW / species

    if not species_dir.exists():
        print(f"[WARN] No existe {species_dir}")
        return urls

    for class_dir in sorted(species_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for haplo_dir in sorted(class_dir.iterdir()):
            if not haplo_dir.is_dir():
                continue

            csv_file = haplo_dir / "merged_unique_events.csv"
            if not csv_file.exists():
                continue

            print(f"Leyendo {csv_file}")

            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    parent_url = str(row.get("parent_id") or "").strip()
                    if parent_url:
                        urls.add(parent_url)

    return urls


def fetch_uniprot_fasta(session: requests.Session, protein_id: str) -> str:
    if not protein_id:
        return ""

    fasta_url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"

    try:
        r = session.get(fasta_url, timeout=60)
        if r.status_code != 200:
            return ""

        text = r.text.strip()
        if not text.startswith(">"):
            return ""

        return text
    except Exception:
        return ""


def fetch_ncbi_fasta(session: requests.Session, protein_id: str) -> str:
    if not protein_id:
        return ""

    fasta_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=protein&id={protein_id}&rettype=fasta&retmode=text"
    )

    try:
        r = session.get(fasta_url, timeout=60)
        if r.status_code != 200:
            return ""

        text = r.text.strip()
        if not text.startswith(">"):
            return ""

        return text
    except Exception:
        return ""


def fetch_fasta_with_fallback(session: requests.Session, parent_url: str) -> Tuple[str, str, str, str, str]:
    """
    Devuelve:
    - db
    - id_full
    - id_base
    - resolved_id
    - fasta
    """
    db, id_full, id_base = parse_protein_url(parent_url)

    if db == "uniprot":
        fasta = fetch_uniprot_fasta(session, id_full)
        if fasta:
            return db, id_full, id_base, id_full, fasta

        if id_base and id_base != id_full:
            fasta = fetch_uniprot_fasta(session, id_base)
            if fasta:
                return db, id_full, id_base, id_base, fasta

        return db, id_full, id_base, "", ""

    if db == "ncbi_protein":
        fasta = fetch_ncbi_fasta(session, id_full)
        if fasta:
            return db, id_full, id_base, id_full, fasta

        if id_base and id_base != id_full:
            fasta = fetch_ncbi_fasta(session, id_base)
            if fasta:
                return db, id_full, id_base, id_base, fasta

        return db, id_full, id_base, "", ""

    return db, id_full, id_base, "", ""


def save_fasta_csv(species: str, parent_urls: Set[str]) -> None:
    out_file = DATA_INTERMEDIATE / f"{species}_parent_protein_fasta.csv"
    session = make_session()

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_url", "db", "id_full", "id_base", "resolved_id", "fasta"])

        total = len(parent_urls)
        for i, parent_url in enumerate(sorted(parent_urls), start=1):
            db, id_full, id_base, resolved_id, fasta = fetch_fasta_with_fallback(session, parent_url)
            writer.writerow([parent_url, db, id_full, id_base, resolved_id, fasta])

            if not fasta:
                print(f"[WARN] {species} | sin FASTA | {parent_url}")

            if i % 200 == 0 or i == total:
                print(f"[{species}] proteínas procesadas {i}/{total}")

            time.sleep(0.1)

    print(f"[{species}] FASTA guardado en: {out_file}")


def main() -> None:
    for species in SPECIES:
        parent_urls = collect_unique_parent_urls(species)
        print(f"[{species}] proteínas únicas (parent_id): {len(parent_urls)}")
        save_fasta_csv(species, parent_urls)


if __name__ == "__main__":
    main()