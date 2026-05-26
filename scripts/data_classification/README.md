# data_classification

## Overview

This module generates the **epitope classification dataset** used for downstream modeling.

It takes the deduplicated epitope events from `data_download` and:

1. Retrieves the **parent protein sequences (FASTA)**
2. Generates **fixed-length sliding windows (25 aa)**
3. Labels each window as **epitope / non-epitope**

---

## Pipeline role

This module connects:

```text
data_download → data_classification → embeddings
```

- Input → epitope events (`merged_unique_events.csv`)
- Output → labeled sequence windows for model training

---

## Scripts
### 1. `fetch_parent_proteins_fastas.py`

Fetches protein sequences from external databases (UniProt / NCBI).

#### Input

```text
data_raw/*/*/*/merged_unique_events.csv
```

#### Output

```text
data_intermediate/
├── human_parent_protein_fasta.csv
├── mouse_parent_protein_fasta.csv
```

#### Description

- Extracts unique parent_id values
- Identifies database:
  - UniProt
  - NCBI Protein
- Retrieves FASTA sequences via API
- Uses fallback strategy:
  - full accession → base accession
- Stores sequences in CSV format

#### Output columns

```text
protein_url | db | id_full | id_base | resolved_id | fasta
```

### 2. `epitope_classification.py`

Builds the final classification dataset.

#### Input

- `merged_unique_events.csv` (from `data_download`)
- `*_parent_protein_fasta.csv` (from previous step)

#### Output

```text
data_intermediate/{species}/{mhc_class}/{haplotype}/
└── classification_{species}_{class}_{haplotype}.csv
```

#### Description

For each protein:

1. Loads full protein sequence
2. Validates epitope positions (exact match with sequence)
3. Generates sliding windows of 25 amino acids
4. Labels each window:
   - 1 → contains epitope
   - 0 → does not contain epitope

#### Window representation

Each row corresponds to a 25 amino acid window:
```text
25aa_seq | contains_epitope | selected_epitope | epitope_pos_score | protein_url | window_start
```

#### Epitope position score

The epitope_pos_score represents the relative position of the epitope inside the window:

- `0.0` → epitope centered
- `< 0` → shifted to the left
- `> 0` → shifted to the right
- Range: [-1, 1]

#### Data filtering

During processing:

- Rows with missing values are removed
- Epitope coordinates are validated
- Epitopes must exactly match the protein sequence
- Proteins without valid epitopes are discarded

---

## Execution order

Run scripts in this order:

```bash
python3 fetch_parent_proteins_fastas.py
python3 epitope_classification.py
```
---

## Notes

- Window size is fixed to 25 amino acids
- Multiple windows may correspond to the same epitope
- Dataset size can be very large due to sliding window generation
- Exact-match validation ensures data consistency

---

## Summary

This module transforms raw epitope annotations into a structured supervised dataset, ready for embedding generation and neural network training.

