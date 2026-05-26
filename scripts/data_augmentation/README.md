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
- Output → augmented sequences with BLOSUM-based mutations

---

## Script

### `generate_blosum62_variants.py`

Generates sequence variants using the `BLOSUM62` substitution matrix.

#### Input

`classification_{species}_{class}_{haplotype}.csv`

Each row contains a 25 amino acid sequence and its label.

#### Output

`classification_{species}_{class}_{haplotype}_blosum.csv`

#### Description

For each input sequence:

1. Keeps the original sequence
2. Generates one variant per position
   - Each variant contains a single amino acid substitution
   - The substitution is chosen using `BLOSUM62` (most probable change)

#### Variant generation

Given a sequence of length 25:

```text
1 original sequence
+ 25 mutated variants (1 per position)
= 26 total sequences per input row
```

Each mutation:

- Changes only one amino acid
- Uses the highest-scoring substitution from `BLOSUM62`

#### Output format

Each row in the output dataset:

```text
original_seq | blosum_seq | group_id | contains_epitope | selected_epitope | epitope_pos_score
```

#### Columns

- `original_seq` → original 25-aa sequence
- `blosum_seq` → mutated (or original) sequence
- `group_id` → links all variants from the same original sequence
- `contains_epitope` → label (`0`/`1`)
- `selected_epitope` → epitope sequence (if any)
- `epitope_pos_score` → relative epitope position

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

## Summary

This module expands the dataset by generating biologically meaningful sequence variants, improving model generalization and robustness.