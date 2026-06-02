# data_augmentation

## Overview

This module performs **data augmentation** on the epitope classification dataset using **BLOSUM62-based amino acid substitutions**.

It generates **single-point mutated variants** of each 25 amino acid sequence to increase dataset diversity while preserving biological plausibility.

---

## Pipeline role

This module connects:

```text
data_classification → data_augmentation → embeddings
```

- Input → labeled 25-aa windows
- Output → augmented sequences with BLOSUM-based mutations and preserved dataset identifiers

---

## Script

### `generate_blosum62_variants.py`

Generates sequence variants using the `BLOSUM62` substitution matrix.

#### Input

`classification_{species}_{class}_{haplotype}.csv`

Each row contains a 25-aa classification window together with its identifiers and labels.

#### Required input columns

```text
protein_group_id
group_id
25aa_seq
contains_epitope
selected_epitope
epitope_pos_score
```

#### Output

`classification_{species}_{class}_{haplotype}_blosum.csv`

#### Description

For each input sequence:

1. Keeps the original sequence
2. Generates one variant per position
   - Each variant contains a single amino acid substitution
   - The substitution is chosen using `BLOSUM62` (most probable change)

#### Variant generation

```text
1 original sequence
+ N mutated variants (one per position)
= N + 1 sequences
```

For the current pipeline:

```text
N = 25
→ 26 sequences per input row
```

Each mutation:

- Changes only one amino acid
- Uses the highest-scoring substitution from `BLOSUM62`

#### Output format

Each row in the output dataset:

```text
protein_group_id | group_id | original_seq | blosum_seq | contains_epitope | selected_epitope | epitope_pos_score
```

#### Data validation

- Empty sequences are skipped
- Sequences containing non-standard amino acids are skipped
- Rows with invalid protein_group_id or group_id values are skipped

#### Columns

- `protein_group_id` → identifier linking all variants to the same parent protein
- `group_id` → identifier linking all variants generated from the same original 25-aa window
- `original_seq` → original 25-aa sequence
- `blosum_seq` → mutated (or original) sequence
- `contains_epitope` → label (`0`/`1`)
- `selected_epitope` → epitope sequence (if any)
- `epitope_pos_score` → relative epitope position

#### Dataset identifiers

The identifiers generated during `data_classification`
are preserved:

```text
protein_group_id
```

Links every augmented sequence to its parent protein.

```text
group_id
```

Links every augmented sequence to the original 25-aa classification window.

All BLOSUM variants generated from the same window share the same identifiers.

#### Important notes

- The **first row of each group** corresponds to the original sequence
- Mutations are **conservative substitutions** (biologically plausible)
- Only **standard amino acids** are considered:

```text
ACDEFGHIKLMNPQRSTVWY
```
- Sequences with non-standard amino acids are skipped

#### Execution

Run:

```bash
python3 generate_blosum62_variants.py input.csv [output.csv]
```

If no output file is provided:

```text
input.csv → input_blosum.csv
```

---

## Important

This module preserves both
`protein_group_id`
and
`group_id`
from the classification dataset.

These identifiers allow downstream modules
(secondary structure, embeddings and neural-network training)
to trace every augmented sequence back to:

- its parent protein
- its original classification window

---

## Summary

This module expands the classification dataset by generating BLOSUM62-based conservative amino acid substitutions while preserving protein and window identifiers required by downstream modules.