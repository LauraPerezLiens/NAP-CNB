# data_download

## Overview

This module retrieves epitope data from the **IEDB API** and generates a **processed, deduplicated dataset of epitope events**.

It downloads both **MHC binding assays** and **T-cell assays**, merges them, and produces a clean dataset per **species**, **MHC class**, and **haplotype**.

---

## Pipeline role

This module corresponds to the **data ingestion step** of the pipeline:

```text
external data (IEDB) → data_download → data_classification
```

- Input → IEDB API
- Output → `merged_unique_events.csv`

---

## Script

### `build_iedb_dataset.py`

Downloads and processes epitope data from IEDB.

#### Data sources

- MHC assays → https://query-api.iedb.org/mhc_export
- T-cell assays → https://query-api.iedb.org/tcell_export

#### Output

Data is stored in:

```text
/home/nap/lperez_nn/data/data_raw/
```

Organized as:

```text
species / mhc-class / haplotype
```

#### Example directory structure

```text
data_raw/
└── human/
    └── mhc-I/
        └── HLA-A/
            ├── mhc_export_full.csv
            ├── tcell_export_full.csv
            └── merged_unique_events.csv
```

#### Output files

For each haplotype:

- `mhc_export_full.csv` → raw MHC data from IEDB
- `tcell_export_full.csv` → raw T-cell data from IEDB
- `merged_unique_events.csv` → final processed dataset

#### Filtering criteria

The script applies the following filters when querying IEDB:

- Linear peptides only
- Positive assays only
- Species-specific filtering:
   - human → *Homo sapiens*
   - mouse → *Mus musculus*
- MHC class I and II
- Haplotype-specific filtering (e.g., HLA-A, HLA-B, H2-K, etc.)

#### Processing pipeline

1. Query IEDB API using paginated requests
2. Retrieve MHC and T-cell datasets
3. Save raw datasets (`*_export_full.csv`)
4. Merge both datasets
5. Extract canonical epitope sequence
6. Filter invalid entries:
   - Missing epitope
   - Missing start/end positions
7. Remove duplicates
8. Save final dataset (`merged_unique_events.csv`)

#### Canonical epitope extraction

Epitope sequences are normalized using:

```text
([A-Z]+)
```

This extracts the amino acid sequence from the epitope name.

#### Deduplication criteria

Entries are considered duplicates if they share:

```text
(epitope sequence, start, end, source_id, parent_id)
```

#### Error handling

If an API request fails, the script **raises an exception and stops execution**.

This prevents generating incomplete or corrupted datasets.

#### Execution

Run:

```bash
python3 build_iedb_dataset.py
```

---

## Notes

- Output size depends on IEDB content and filters
- Some entries are discarded due to missing or invalid data
- `protein_id` is derived from `parent_id`
- Dataset quality depends on IEDB consistency

---

## Summary

This module retrieves and processes raw epitope data from IEDB, generating the base dataset used in downstream steps of the pipeline.