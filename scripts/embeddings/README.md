# embeddings

## Overview

This module generates **feature representations of protein sequences** for downstream neural network models.

Two complementary embedding strategies are implemented:

- **Primary structure embeddings** → based on amino acid sequence (ProtT5)
- **Secondary structure embeddings** → based on predicted structural states (SecondaryBERT)

These embeddings capture different biological properties of proteins.

---

## Pipeline role

This module connects:

```text
data_augmentation → embeddings → model_training
```

- Input → augmented sequences (`*_blosum.csv`)
- Output → numerical embeddings (model input features)

---

## Structure

```text
embeddings/
├── primary_structure/
│   └── generate_prott5_embeddings.py
└── secondary_structure/
    └── generate_secondary_bert_embeddings.py
```
---

## Primary structure embeddings

### Script

#### `generate_prott5_embeddings.py`

##### Description

Generates embeddings from amino acid sequences using the ProtT5 model.

##### Input

`classification_{species}_{class}_{haplotype}_blosum.csv`

##### Key steps

1. Clean protein sequences
2. Convert sequences to ProtT5 format (space-separated amino acids)
3. Compute embeddings using ProtT5
4. Apply mean pooling over tokens
5. Save embeddings in chunks

##### Output

```text
chunk_XXXXXX_X.npy       → embeddings (float32)
chunk_XXXXXX_y.npy       → labels
chunk_XXXXXX_pos.npy     → epitope position score
chunk_XXXXXX_group_id.npy
chunk_XXXXXX_metadata.csv
chunk_XXXXXX_index.csv
master_index.csv
```

##### Embedding details

- Model → `Rostlab/prot_t5_xl_half_uniref50-enc`
- Output dimension → 1024
- Pooling → mean pooling

---

## Secondary structure embeddings

### Script

#### `generate_secondary_bert_embeddings.py`

##### Description

Generates embeddings from predicted secondary structure sequences (C/H/E) using a BERT-based model.

##### Input

`classification_{species}_{class}_{haplotype}_ss.csv`

##### Key steps

1. Clean secondary structure sequences
2. Ensure fixed length (25 positions)
3. Convert to BERT format (space-separated tokens)
4. Compute token-level embeddings
5. Save embeddings grouped by group_id

##### Output

```text
chunks/
├── chunk_XXXXXX_X_ss.npy        → embeddings (25 x 768)
├── chunk_XXXXXX_group_ids.npy
metadata.csv
group_index.csv
chunks_manifest.csv
embedding_config.json
done.flag
```

##### Embedding details

- Model → custom SecondaryBERT
- Input tokens → C / H / E
- Output shape → (25, 768) per sequence
- No pooling → preserves positional information

---

## Key differences


| Feature               | Primary (ProtT5)        | Secondary (BERT)          |
|----------------------|-------------------------|----------------------------|
| Input                | Amino acid sequence     | Secondary structure (C/H/E) |
| Model                | ProtT5                  | BERT                       |
| Output shape         | (1024)                  | (25, 768)                  |
| Pooling              | Mean pooling            | No pooling                 |
| Information captured | Sequence-level          | Structural pattern         |

---

## Important notes

- Primary and secondary embeddings are **complementary**
- Primary embeddings capture **biochemical properties**
- Secondary embeddings capture **structural patterns**
- Both can be combined as input features in downstream models

---

## Execution

Primary structure:

```bash
python3 generate_prott5_embeddings.py
```

Secondary structure:

```bash
python3 generate_secondary_bert_embeddings.py
```
---

## Summary

This module transforms protein data into high-dimensional numerical representations, enabling machine learning models to learn from both sequence and structural information.