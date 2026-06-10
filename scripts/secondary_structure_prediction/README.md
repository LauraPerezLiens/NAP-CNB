# secondary_structure

## Overview

This module handles the **prediction and integration of protein secondary structure information** within the pipeline.

It uses a ProteinUnet-based model to predict secondary structure labels from protein sequences and maps those predictions to the 25 amino acid windows used for downstream modeling.

The generated secondary structure annotations are later used to compute SecondaryBERT embeddings.

---

## Pipeline role

This module connects:

```text
data_classification → secondary_structure → secondary_structure_embeddings
```

- Input → parent protein FASTA sequences and classification windows
- Output → classification datasets enriched with secondary structure

---

## Structure

```text
secondary_structure/
├── predict_secondary_structure.py
└── add_secondary_structure.py
```

---

## Scripts

### 1. `predict_secondary_structure.py`

#### Description

Predicts protein secondary structure using a ProteinUnet model.

For each parent protein sequence, the script predicts:

- `C` → coil
- `H` → alpha helix
- `E` → beta strand

Predictions are generated for the full protein sequence.

#### Input

`*_parent_protein_fasta.csv`

Expected columns:

```text
protein_url | fasta
```

The FASTA file is expected to originate from `fetch_parent_proteins_fastas.py` and therefore already contains only validated protein sequences.


#### Output

```text
secondary_struct_human.csv
secondary_struct_mouse.csv
```

Output columns:

```text
protein_url | sequence | secondary_structure
```

#### Key features

- Predicts full-length protein secondary structure
- Handles long proteins using overlapping windows:
   - window size → 1024 aa
   - overlap → 200 aa
- Reconstructs full secondary structure prediction
- Skips proteins with invalid amino acids
- Validates sequence and output lengths

For proteins longer than 1024 residues:

- Predictions are generated independently for overlapping windows
- Window probabilities are averaged in overlapping regions
- The final secondary structure is reconstructed residue-by-residue

#### Protein validation

Only proteins containing standard amino acids are used:

```text
ACDEFGHIKLMNPQRSTVWY
```

Proteins containing any other residue are skipped before prediction.

#### Model details

- Architecture → ProteinUnet
- Output labels → C / H / E
- Prediction granularity → residue-level

#### Execution

Run:

```bash
python3 predict_secondary_structure.py \
    -i input.csv \
    -o output.csv
```

### 2. `add_secondary_structure.py`

#### Description

Maps predicted secondary structure to the 25 amino acid windows used in the classification datasets.

The script aligns each classification window with its corresponding region in the full protein secondary structure.

Secondary structure annotations are added only to the original classification datasets. BLOSUM-augmented datasets are not modified in this step.

#### Input

**Secondary structure predictions**

```text
secondary_struct_human.csv
secondary_struct_mouse.csv
```

**Classification datasets**

```text
classification_*.csv
```

#### Required classification columns

```text
protein_group_id
group_id
25aa_seq
protein_url
window_start
```

#### Output

```text
classification_*_ss.csv
```

#### Key processing steps
1. Load predicted secondary structures
2. Match proteins using normalized URLs
3. Validate window coordinates
4. Validate exact sequence match
5. Extract secondary structure windows
6. Save classification datasets enriched with secondary structure

#### Added columns

The output dataset preserves all original classification columns and appends:

```text
secondary_structure
protein_found
sequence_match
ss_status
```

- `secondary_structure` → predicted SS3 sequence for the 25-aa window
- `protein_found` → indicates whether the parent protein was found in the secondary structure database
- `sequence_match` → indicates whether the extracted protein window exactly matches the classification sequence
- `ss_status` → mapping status

#### Dataset identifiers

The module preserves:

```text
protein_group_id
group_id
```

so that secondary-structure embeddings can later be aligned with primary-structure embeddings.

#### Secondary structure status

Possible values for `ss_status`:

| Status                   | Meaning                                           |
|---------------------------|---------------------------------------------------|
| `ok`                      | successful mapping                                |
| `protein_not_found`       | protein missing in secondary structure predictions |
| `invalid_window_start`    | invalid window coordinates                        |
| `window_out_of_range`     | window exceeds protein length                     |
| `sequence_mismatch`       | classification window does not match protein sequence |


#### Important notes

Secondary structure windows have fixed length:

```text
25 residues
```

- BLOSUM variants do not receive secondary-structure annotations in this module, because secondary structure is predicted from the original protein sequence, not from artificial BLOSUM-mutated windows.
- Exact sequence validation ensures mapping consistency
- Secondary structure is assigned only when the classification window exactly matches the parent protein sequence

#### Execution

Run:

```bash
python3 add_secondary_structure.py
```

---

## Important

Because proteins containing non-standard amino acids are removed during parent-protein FASTA generation (`fetch_parent_proteins_fastas.py`), secondary structure prediction is performed only on validated protein sequences.

As a consequence:

- All predicted structures correspond to valid protein sequences
- Secondary structure mapping is deterministic
- `protein_group_id` and `group_id` remain consistent across classification, BLOSUM and secondary-structure datasets

---

## Summary

This module predicts residue-level SS3 secondary structure for validated parent proteins and maps those predictions to the 25-aa classification windows, generating classification datasets enriched with structural information for downstream SecondaryBERT embedding generation.