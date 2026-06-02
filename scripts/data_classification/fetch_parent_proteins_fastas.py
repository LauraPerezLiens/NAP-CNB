#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ======================================================
# CONFIGURATION
# ======================================================

DATA_RAW = Path("/home/nap/lperez_nn/data/data_raw")
DATA_INTERMEDIATE = Path("/home/nap/lperez_nn/data/data_intermediate")
DATA_INTERMEDIATE.mkdir(parents=True, exist_ok=True)

SPECIES = ["human", "mouse"]

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.1
PROGRESS_EVERY = 200

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

UNIPROT_RE = re.compile(
    r"https?://(?:www\.)?uniprot\.org/uniprot/([A-Z0-9]+(?:[-.][A-Z0-9]+)?)/*$",
    re.IGNORECASE,
)

NCBI_PROTEIN_RE = re.compile(
    r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/protein/([A-Z0-9_.-]+)/*$",
    re.IGNORECASE,
)


# ======================================================
# LOGGING
# ======================================================

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


# ======================================================
# HTTP SESSION
# ======================================================

def make_session() -> requests.Session:
    """Create a requests session with retry logic."""

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


# ======================================================
# SEQUENCE HELPERS
# ======================================================

def normalize_url(url: str) -> str:
    """Normalize protein URLs for consistent matching."""

    if pd.isna(url):
        return ""

    return str(url).strip().replace("https://", "http://")


def fasta_to_sequence(fasta_text: str) -> str:
    """Convert FASTA text into a plain amino acid sequence."""

    if not fasta_text:
        return ""

    return "".join(
        line.strip()
        for line in str(fasta_text).splitlines()
        if not line.startswith(">")
    ).upper()


def get_invalid_residues(seq: str) -> str:
    """Return non-standard amino acid residues found in a sequence."""

    invalid = sorted(set(seq) - VALID_AA)

    return ",".join(invalid)


def extract_protein_name(fasta_text: str) -> str:
    """Extract protein name from FASTA header."""

    if not fasta_text:
        return ""

    first_line = str(fasta_text).splitlines()[0].strip()

    if not first_line.startswith(">"):
        return ""

    header = first_line[1:]

    parts = header.split(" ", 1)

    if len(parts) < 2:
        return ""

    description = parts[1]

    return re.split(r"\sOS=", description)[0].strip()


# ======================================================
# URL PARSING
# ======================================================

def parse_protein_url(url: str) -> Tuple[str, str, str]:
    """
    Parse protein URL and extract database identifiers.

    Returns:
        db:
            Database name: uniprot, ncbi_protein or unknown.
        id_full:
            Full protein accession.
        id_base:
            Base accession without isoform/version suffix.
    """

    url = normalize_url(url)

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


# ======================================================
# INPUT COLLECTION
# ======================================================

def collect_unique_parent_urls(species: str) -> Set[str]:
    """Collect unique parent protein URLs from merged_unique_events.csv files."""

    urls: Set[str] = set()
    species_dir = DATA_RAW / species

    if not species_dir.exists():
        logging.warning("[%s] species directory does not exist: %s", species, species_dir)
        return urls

    for class_dir in sorted(species_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for haplo_dir in sorted(class_dir.iterdir()):
            if not haplo_dir.is_dir():
                continue

            csv_file = haplo_dir / "merged_unique_events.csv"

            if not csv_file.exists():
                logging.warning("[%s] missing file: %s", species, csv_file)
                continue

            logging.info("[%s] reading %s", species, csv_file)

            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    parent_url = normalize_url(row.get("parent_id") or "")

                    if parent_url:
                        urls.add(parent_url)

    return urls


# ======================================================
# FASTA FETCHING
# ======================================================

def fetch_uniprot_fasta(session: requests.Session, protein_id: str) -> str:
    """Fetch FASTA from UniProt."""

    if not protein_id:
        return ""

    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            return ""

        text = response.text.strip()

        if not text.startswith(">"):
            return ""

        return text

    except requests.RequestException as exc:
        logging.debug("UniProt request failed for %s: %s", protein_id, exc)
        return ""


def fetch_ncbi_fasta(session: requests.Session, protein_id: str) -> str:
    """Fetch FASTA from NCBI Protein."""

    if not protein_id:
        return ""

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=protein&id={protein_id}&rettype=fasta&retmode=text"
    )

    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)

        if response.status_code != 200:
            return ""

        text = response.text.strip()

        if not text.startswith(">"):
            return ""

        return text

    except requests.RequestException as exc:
        logging.debug("NCBI request failed for %s: %s", protein_id, exc)
        return ""


def fetch_fasta_with_fallback(
    session: requests.Session,
    parent_url: str,
) -> Tuple[str, str, str, str, str]:
    """
    Fetch FASTA using full accession first and base accession as fallback.

    Returns:
        db, id_full, id_base, resolved_id, fasta
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


# ======================================================
# OUTPUT
# ======================================================

def save_species_outputs(
    species: str,
    valid_rows: List[Dict],
    master_rows: List[Dict],
) -> None:
    """Save FASTA CSV and protein master index for one species."""

    fasta_out = DATA_INTERMEDIATE / f"{species}_parent_protein_fasta.csv"
    master_out = DATA_INTERMEDIATE / f"{species}_protein_master_index.csv"

    fasta_df = pd.DataFrame(
        valid_rows,
        columns=[
            "protein_group_id",
            "protein_url",
            "db",
            "id_full",
            "id_base",
            "resolved_id",
            "fasta",
        ],
    )

    master_df = pd.DataFrame(
        master_rows,
        columns=[
            "protein_group_id",
            "protein_url",
            "protein_name",
            "resolved_id",
        ],
    )

    fasta_df.to_csv(fasta_out, index=False)
    master_df.to_csv(master_out, index=False)

    logging.info("[%s] FASTA CSV saved: %s", species, fasta_out)
    logging.info("[%s] protein master index saved: %s", species, master_out)


def process_species(species: str) -> None:
    """Fetch and filter parent protein FASTA sequences for one species."""

    parent_urls = collect_unique_parent_urls(species)

    logging.info("[%s] unique parent proteins: %d", species, len(parent_urls))

    if not parent_urls:
        logging.warning("[%s] no parent protein URLs found. Skipping.", species)
        return

    session = make_session()

    valid_rows: List[Dict] = []
    master_rows: List[Dict] = []

    total = len(parent_urls)

    missing = 0
    skipped_invalid = 0
    unknown_url = 0
    written = 0

    invalid_residue_counter: Dict[str, int] = {}

    protein_group_id = 1

    for i, parent_url in enumerate(sorted(parent_urls), start=1):
        parent_url = normalize_url(parent_url)

        db, id_full, id_base, resolved_id, fasta = fetch_fasta_with_fallback(
            session=session,
            parent_url=parent_url,
        )

        if db == "unknown":
            unknown_url += 1

        if not fasta:
            missing += 1
            logging.warning("[%s] FASTA not found: %s", species, parent_url)
            continue

        sequence = fasta_to_sequence(fasta)
        invalid_residues = get_invalid_residues(sequence)

        if invalid_residues:
            skipped_invalid += 1

            for residue in invalid_residues.split(","):
                invalid_residue_counter[residue] = invalid_residue_counter.get(residue, 0) + 1

            logging.warning(
                "[%s] skipped non-standard residues (%s): %s",
                species,
                invalid_residues,
                parent_url,
            )
            continue

        protein_name = extract_protein_name(fasta)

        valid_rows.append({
            "protein_group_id": protein_group_id,
            "protein_url": parent_url,
            "db": db,
            "id_full": id_full,
            "id_base": id_base,
            "resolved_id": resolved_id,
            "fasta": fasta,
        })

        master_rows.append({
            "protein_group_id": protein_group_id,
            "protein_url": parent_url,
            "protein_name": protein_name,
            "resolved_id": resolved_id,
        })

        written += 1
        protein_group_id += 1

        if i % PROGRESS_EVERY == 0 or i == total:
            logging.info("[%s] processed proteins %d/%d", species, i, total)

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    save_species_outputs(
        species=species,
        valid_rows=valid_rows,
        master_rows=master_rows,
    )

    logging.info("[%s] total parent proteins: %d", species, total)
    logging.info("[%s] written valid FASTA: %d", species, written)
    logging.info("[%s] missing FASTA: %d", species, missing)
    logging.info("[%s] skipped due to non-standard residues: %d", species, skipped_invalid)
    logging.info("[%s] unknown URL format: %d", species, unknown_url)
    logging.info("[%s] invalid residue types: %s", species, invalid_residue_counter)


# ======================================================
# MAIN
# ======================================================

def main() -> None:
    setup_logging()

    for species in SPECIES:
        process_species(species)

    logging.info("DONE")


if __name__ == "__main__":
    main()