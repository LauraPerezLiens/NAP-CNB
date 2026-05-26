# build_iedb_dataset.py

## Overview

This script downloads epitope data from the IEDB API and generates a **processed, deduplicated dataset of epitope events**.

It retrieves both **MHC binding assays** and **T-cell assays**, merges them, and produces a clean dataset per species, MHC class, and haplotype.

---

## Data sources

- MHC assays: https://query-api.iedb.org/mhc_export  
- T-cell assays: https://query-api.iedb.org/tcell_export  

---

## Output structure

Data is stored in:
```bash
/home/nap/lperez_nn/data/data_raw/
```

Organized as:

```bash
species / mhc-class / haplotype
```

Example:

```bash
data_raw/
├── human/
│ ├── mhc-I/
│ │ ├── HLA-A/
│ │ │ ├── mhc_export_full.csv
│ │ │ ├── tcell_export_full.csv
│ │ │ └── merged_unique_events.csv
```

---

## Output files

For each haplotype:

- `mhc_export_full.csv` → raw MHC data from IEDB  
- `tcell_export_full.csv` → raw T-cell data from IEDB  
- `merged_unique_events.csv` → **final processed dataset**

---

## Filtering criteria

The script applies the following filters when querying IEDB:

- Linear peptides only  
- Positive assays only  
- Species-specific filtering:
  - human → *Homo sapiens*
  - mouse → *Mus musculus*
- MHC class I and II  
- Specific haplotypes (e.g., HLA-A, HLA-B, H2-K, etc.)

---

## Processing pipeline

1. Query IEDB API with pagination  
2. Retrieve MHC and T-cell datasets  
3. Save raw datasets (`*_export_full.csv`)  
4. Merge both datasets  
5. Extract canonical epitope sequence  
6. Filter invalid entries:
   - Missing epitope
   - Missing start/end positions  
7. Remove duplicates  
8. Save final dataset (`merged_unique_events.csv`)  

---

## Canonical epitope extraction

Epitope sequences are normalized using:

```python
([A-Z]+)
```

This extracts the amino acid sequence from the epitope name.

---

## Deduplication criteria

Entries are considered duplicates if they share:

```python
(epitope sequence, start, end, source_id, parent_id)
```

---

## Key functions
- fetch_all() → API pagination with retry logic
- fetch_pipeline() → builds query parameters
- canonical_epitope() → extracts amino acid sequence
- build_merged_unique_events() → filtering and deduplication
- write_full_csv() → saves raw data

---

## Technical details
- Uses retry logic for robust API calls
- Pagination:
  - limit = 1000 (applied to both MHC and T-cell endpoints)
- Logging:
  - INFO → progress tracking
  - WARNING/ERROR → API issues

---
## Error handling

If an API request fails, the script **raises an exception and stops execution**.

This prevents generating incomplete or corrupted datasets.

--- 

## Usage

Run:
```bash
python3 build_iedb_dataset.py
```

--- 

## Notes
- Output size depends on IEDB content and filters
- Some entries are discarded due to missing or invalid data
- *protein_id* is derived from *parent_id*
- Dataset quality depends on IEDB consistency

---

## Summary

This script is the **data ingestion step of the pipeline**, responsible for generating the base dataset used in downstream processing and model training.