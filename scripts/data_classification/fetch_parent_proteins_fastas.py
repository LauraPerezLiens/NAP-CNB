#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================
# CONFIGURATION
# =========================

DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")
DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

SPECIES = ["human", "mouse"]

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.1
PROGRESS_EVERY = 200

UNIPROT_RE = re.compile(
    r"https?://(?:www\.)?uniprot\.org/uniprot/([A-Z0-9]+(?:[-.][A-Z0-9]+)?)/*$",
    re.IGNORECASE,
)

NCBI_PROTEIN_RE = re.compile(
    r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/protein/([A-Z0-9_.-]+)/*$",
    re.IGNORECASE,
)


# =========================
# LOGGING
# =========================

class MaxLevelFilter(logging.Filter):
    """Filter log records up to a maximum logging level."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def setup_logging() -> None:
    """Configure logging: INFO goes to stdout, WARNING/ERROR goes to stderr."""

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)


# =========================
# HTTP SESSION
# =========================

def make_session() -> requests.Session:
    """Create a requests session with retry strategy."""

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


# =========================
# URL PARSING
# =========================

def parse_protein_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a protein URL and extract database and accession identifiers.

    Returns:
        db:
            Protein database name: "uniprot", "ncbi_protein", or "unknown".
        id_full:
            Full accession identifier, including version if present.
        id_base:
            Base accession identifier without version suffix.

    Examples:
        https://www.uniprot.org/uniprot/P47974.3
        -> ("uniprot", "P47974.3", "P47974")

        http://www.ncbi.nlm.nih.gov/protein/NP_008818.3
        -> ("ncbi_protein", "NP_008818.3", "NP_008818")
    """

    url = str(url).strip()
    if not url:
        return "", "", ""

    match = UNIPROT_RE.match(url)
    if match:
        id_full = match.group(1).upper().strip()
        id_base = re.split(r"[-.]", id_full)[0]
        return "uniprot", id_full, id_base

    match = NCBI_PROTEIN_RE.match(url)
    if match:
        id_full = match.group(1).strip()
        id_base = id_full.split(".")[0]
        return "ncbi_protein", id_full, id_base

    return "unknown", "", ""


# =========================
# INPUT COLLECTION
# =========================

def collect_unique_parent_urls(species: str) -> Set[str]:
    """Collect unique parent protein URLs from merged_unique_events.csv files."""

    urls: Set[str] = set()
    species_dir = DATA_RAW / species

    if not species_dir.exists():
        logging.warning("Species directory does not exist: %s", species_dir)
        return urls

    for class_dir in sorted(species_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for haplo_dir in sorted(class_dir.iterdir()):
            if not haplo_dir.is_dir():
                continue

            csv_file = haplo_dir / "merged_unique_events.csv"
            if not csv_file.exists():
                logging.warning("Missing file: %s", csv_file)
                continue

            logging.info("Reading %s", csv_file)

            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    parent_url = str(row.get("parent_id") or "").strip()
                    if parent_url:
                        urls.add(parent_url)

    return urls


# =========================
# FASTA FETCHING
# =========================

def fetch_uniprot_fasta(session: requests.Session, protein_id: str) -> str:
    """Fetch FASTA sequence from UniProt."""

    if not protein_id:
        return ""

    fasta_url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"

    try:
        response = session.get(fasta_url, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            logging.debug("UniProt request failed for %s with status %s", protein_id, response.status_code)
            return ""

        text = response.text.strip()
        if not text.startswith(">"):
            logging.debug("Invalid UniProt FASTA response for %s", protein_id)
            return ""

        return text

    except requests.RequestException as exc:
        logging.debug("UniProt request exception for %s: %s", protein_id, exc)
        return ""


def fetch_ncbi_fasta(session: requests.Session, protein_id: str) -> str:
    """Fetch FASTA sequence from NCBI Protein."""

    if not protein_id:
        return ""

    fasta_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=protein&id={protein_id}&rettype=fasta&retmode=text"
    )

    try:
        response = session.get(fasta_url, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            logging.debug("NCBI request failed for %s with status %s", protein_id, response.status_code)
            return ""

        text = response.text.strip()
        if not text.startswith(">"):
            logging.debug("Invalid NCBI FASTA response for %s", protein_id)
            return ""

        return text

    except requests.RequestException as exc:
        logging.debug("NCBI request exception for %s: %s", protein_id, exc)
        return ""


def fetch_fasta_with_fallback(session: requests.Session, parent_url: str) -> Tuple[str, str, str, str, str]:
    """
    Fetch FASTA sequence using full identifier first and base identifier as fallback.

    Returns:
        db:
            Source database.
        id_full:
            Full accession identifier.
        id_base:
            Base accession identifier.
        resolved_id:
            Identifier that successfully retrieved the FASTA sequence.
        fasta:
            FASTA sequence text. Empty string if retrieval failed.
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

    logging.warning("Unknown protein URL format: %s", parent_url)
    return db, id_full, id_base, "", ""


# =========================
# OUTPUT
# =========================

def save_fasta_csv(species: str, parent_urls: Set[str]) -> None:
    """Fetch FASTA sequences and save them into a species-specific CSV file."""

    out_file = DATA_INTERMEDIATE / f"{species}_parent_protein_fasta.csv"
    session = make_session()

    total = len(parent_urls)
    found = 0
    missing = 0

    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["protein_url", "db", "id_full", "id_base", "resolved_id", "fasta"])

        for i, parent_url in enumerate(sorted(parent_urls), start=1):
            db, id_full, id_base, resolved_id, fasta = fetch_fasta_with_fallback(session, parent_url)

            writer.writerow([parent_url, db, id_full, id_base, resolved_id, fasta])

            if fasta:
                found += 1
            else:
                missing += 1
                logging.warning("%s | FASTA not found | %s", species, parent_url)

            if i % PROGRESS_EVERY == 0 or i == total:
                logging.info("[%s] processed proteins %d/%d", species, i, total)

            time.sleep(SLEEP_BETWEEN_REQUESTS)

    logging.info("[%s] FASTA saved to: %s", species, out_file)
    logging.info("[%s] FASTA found: %d", species, found)
    logging.info("[%s] FASTA missing: %d", species, missing)


# =========================
# MAIN
# =========================

def main() -> None:
    setup_logging()

    for species in SPECIES:
        parent_urls = collect_unique_parent_urls(species)

        logging.info("[%s] unique parent proteins: %d", species, len(parent_urls))

        if not parent_urls:
            logging.warning("[%s] no parent protein URLs found. Skipping.", species)
            continue

        save_fasta_csv(species, parent_urls)

    logging.info("DONE")


if __name__ == "__main__":
    main()