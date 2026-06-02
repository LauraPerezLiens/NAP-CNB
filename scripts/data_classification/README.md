# data_classification

## Overview

This module generates the **epitope classification dataset** used for downstream modeling.

It takes the deduplicated epitope events from `data_download` and:

1. Retrieves and validates parent protein sequences (FASTA)
2. Removes proteins containing non-standard amino acids
3. Generates fixed-length sliding windows (25 aa)
4. Labels each window as epitope / non-epitope
5. Assigns stable dataset identifiers for downstream modules

---

## Pipeline role

This module connects:

```text
data_download → data_classification → embeddings
```

- Input → epitope events (`merged_unique_events.csv`)
- Output → labeled 25-aa classification windows with stable dataset identifiers

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
- Converts FASTA records into plain amino acid sequences
- Removes proteins containing non-standard amino acids
  (e.g. X, U, O, B, Z, J)

#### Protein validation

Only proteins composed exclusively of the 20 standard amino acids are retained:

```text
ACDEFGHIKLMNPQRSTVWY
```

Proteins containing any non-standard residue are discarded before being written to the output FASTA dataset.

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
5. Assigns protein_group_id and group_id identifiers

#### Window representation

Each row corresponds to a 25 amino acid window:
```text
protein_group_id | group_id | 25aa_seq | contains_epitope | selected_epitope | epitope_pos_score | protein_url | window_start
```

#### Dataset identifiers

Each generated window receives two identifiers:

```text
protein_group_id
```

Unique identifier assigned to each protein within the classification dataset.

```text
group_id
```

Unique identifier assigned to every 25-aa window.

These identifiers are later propagated to:

- BLOSUM augmented datasets
- Secondary structure datasets
- Embedding datasets
- Neural-network inputs

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
- Proteins without FASTA sequences are discarded
- Proteins shorter than the window size are discarded
- Proteins without valid epitopes after exact-match validation are discarded
- Duplicate windows are removed

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
- Duplicate classification windows are removed before export

---

## Important

Protein validation is performed during FASTA retrieval by
`fetch_parent_proteins_fastas.py`.

As a consequence:

- Proteins containing non-standard amino acids are excluded
- All classification windows originate from validated protein sequences
- `protein_group_id` and `group_id` remain consistent across downstream modules

---

## Summary

This module retrieves and validates parent protein sequences, removes invalid proteins, generates labeled 25-aa classification windows, and assigns stable identifiers used throughout downstream embedding and neural-network workflows.

