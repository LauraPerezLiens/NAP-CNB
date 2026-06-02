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

It also propagates secondary structure annotations to BLOSUM-augmented datasets.

#### Input

**Secondary structure predictions**

```text
secondary_struct_human.csv
secondary_struct_mouse.csv
```

**Classification datasets**

```text
classification_*.csv
classification_*_blosum.csv
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
classification_*_blosum_ss.csv
```

#### Key processing steps
1. Load predicted secondary structures
2. Match proteins using normalized URLs
3. Validate window coordinates
4. Validate exact sequence match
5. Extract secondary structure windows
6. Propagate annotations to BLOSUM variants using `protein_group_id` and `group_id`

#### Added columns

```text
secondary_structure
protein_found
sequence_match
ss_status
```

#### Dataset identifiers

The module preserves:

```text
protein_group_id
group_id
```

and uses them to propagate secondary structure annotations from the original classification windows to all BLOSUM variants.

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

- BLOSUM variants inherit secondary structure from the original sequence using both `protein_group_id` and `group_id`.
- Exact sequence validation ensures mapping consistency
- Secondary structure is assigned only when the classification window exactly matches the parent protein sequence

#### Execution

Run:

```bash
python3 add_secondary_structure.py
```

---

## Important

Because proteins containing non-standard amino acids are removed during `data_classification`, secondary structure prediction is performed only on validated protein sequences.

As a consequence:

- All predicted structures correspond to valid protein sequences
- Secondary structure mapping is deterministic
- `protein_group_id` and `group_id` remain consistent across classification, BLOSUM and secondary-structure datasets

---

## Summary

This module predicts secondary structure for validated parent proteins and integrates residue-level structural annotations into classification and BLOSUM datasets while preserving dataset identifiers required by downstream embedding workflows.