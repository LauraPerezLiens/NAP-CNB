# embeddings

## Overview

This module generates **feature representations of protein sequences** for downstream neural network models.

Two complementary embedding strategies are implemented:

- **Primary structure embeddings** → amino acid sequence embeddings generated with ProtT5
- **Secondary structure embeddings** → secondary structure embeddings generated with SecondaryBERT

Both embedding types preserve the relationship between proteins, windows and labels through `protein_group_id` and `group_id`, allowing complete traceability throughout the pipeline.

---

## Pipeline role

```text
data_augmentation
        │
        ▼
embeddings
        │
        ▼
model_training
```

**Input**

```text
classification_*_blosum.csv
classification_*_ss.csv
```

**Output**

```text
ProtT5 embeddings
SecondaryBERT embeddings
```

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

# Primary structure embeddings

## Script

### `generate_prott5_embeddings.py`

### Description

Generates sequence-level embeddings from amino acid windows using the ProtT5 protein language model.

The script:

1. Loads augmented sequence windows from classification datasets.
2. Cleans protein sequences and converts them to the ProtT5 token format.
3. Computes ProtT5 token embeddings.
4. Applies mean pooling across sequence positions.
5. Saves embeddings, labels and metadata in chunked files.

To reduce computation, identical sequences within a chunk are embedded only once and reused through sequence factorization.

---

### Input

```text
classification_{species}_{class}_{haplotype}_blosum.csv
```

Required columns:

```text
protein_group_id
group_id
blosum_seq
contains_epitope
epitope_pos_score
```

---

### Embedding details

| Parameter | Value |
|------------|---------|
| Model | Rostlab/prot_t5_xl_half_uniref50-enc |
| Embedding size | 1024 |
| Pooling | Mean pooling |
| Maximum length | 30 |
| Chunk size | 100,000 rows |
| Batch size | 32 |

Output shape:

```text
(N, 1024)
```

One embedding vector per sequence window.

---

### Output

For each chunk:

```text
chunk_000000_X.npy
chunk_000000_y.npy
chunk_000000_pos.npy
chunk_000000_group_id.npy
chunk_000000_protein_group_id.npy
chunk_000000_metadata.csv
chunk_000000_index.csv
```

Global index:

```text
master_index.csv
```

---

### Output files

#### Embeddings

```text
chunk_XXXXXX_X.npy
```

Shape:

```text
(N, 1024)
```

---

#### Labels

```text
chunk_XXXXXX_y.npy
```

Binary epitope labels.

---

#### Positional score

```text
chunk_XXXXXX_pos.npy
```

Epitope position score.

---

#### Group identifiers

```text
chunk_XXXXXX_group_id.npy
chunk_XXXXXX_protein_group_id.npy
```

Used to track windows and proteins across the pipeline.

---

#### Metadata

```text
chunk_XXXXXX_metadata.csv
```

Contains:

```text
protein_group_id
group_id
blosum_seq
contains_epitope
epitope_pos_score
```

---

#### Index

```text
chunk_XXXXXX_index.csv
```

Maps:

```text
chunk_id
row_idx_in_chunk
protein_group_id
group_id
y
```

---

# Secondary structure embeddings

## Script

### `generate_secondary_bert_embeddings.py`

### Description

Generates contextual embeddings from predicted secondary structure sequences using the pretrained SecondaryBERT model.

The script:

1. Loads secondary structure annotations.
2. Keeps only windows with valid secondary structure predictions (`ss_status == ok`).
3. Cleans and normalizes C/H/E sequences.
4. Converts secondary structure strings into BERT token format.
5. Computes token-level contextual embeddings.
6. Saves embeddings, metadata and identifiers grouped by `group_id`.

Unlike ProtT5 embeddings, no pooling is applied because positional information is preserved.

---

### Input

```text
classification_{species}_{class}_{haplotype}_ss.csv
```

Required columns:

```text
protein_group_id
group_id
25aa_seq
secondary_structure
contains_epitope
epitope_pos_score
protein_url
window_start
```

Optional columns:

```text
selected_epitope
protein_found
sequence_match
ss_status
```

---

### Filtering

If present:

```text
ss_status == "ok"
```

Rows without valid secondary structure predictions are excluded before embedding generation.

---

### Embedding details

| Parameter | Value |
|------------|---------|
| Model | SecondaryBERT |
| Hidden size | 768 |
| Window size | 25 |
| Maximum length | 30 |
| Chunk size | 102,400 rows |
| Batch size | 512 |
| Pooling | None |

Output shape:

```text
(N, 25, 768)
```

Each residue position retains its own embedding vector.

---

### Output structure

```text
classification_*_SecondaryBERT_by_group/
│
├── metadata.csv
├── index.csv
├── protein_group_id.npy
├── group_id.npy
├── chunks_manifest.csv
├── embedding_config.json
├── done.flag
│
└── chunks/
    ├── chunk_000000_X_ss.npy
    ├── chunk_000000_protein_group_id.npy
    ├── chunk_000000_group_id.npy
    ├── chunk_000001_X_ss.npy
    └── ...
```

---

### Metadata

```text
metadata.csv
```

Contains:

```text
protein_group_id
group_id
25aa_seq
secondary_structure
clean_ss
contains_epitope
epitope_pos_score
protein_url
window_start
```

plus any optional columns present in the original dataset.

---

### Index

```text
index.csv
```

Maps:

```text
protein_group_id
group_id
row_idx_in_X_ss
```

allowing reconstruction of the relationship between embeddings and original windows.

---

### Chunk manifest

```text
chunks_manifest.csv
```

Contains:

```text
chunk_id
start
end
n_rows
x_file
protein_group_id_file
group_id_file
```

and allows efficient loading of large embedding collections.

---

### Configuration

```text
embedding_config.json
```

Stores:

```text
source_csv
chunk_size
batch_size
hidden_size
window_size
model_dir
```

for reproducibility.

---

## Key differences

| Feature | Primary (ProtT5) | Secondary (SecondaryBERT) |
|----------|----------|----------|
| Input | Amino acid sequence | Secondary structure sequence |
| Tokens | Amino acids | C / H / E |
| Model | ProtT5 | SecondaryBERT |
| Output shape | (1024) | (25, 768) |
| Pooling | Mean pooling | None |
| Preserves residue positions | No | Yes |
| Information captured | Sequence semantics | Structural patterns |

---

## Important notes

- Primary and secondary embeddings are complementary.
- Both embedding types preserve `protein_group_id` and `group_id`.
- SecondaryBERT embeddings are generated only for proteins with valid secondary structure predictions.
- ProtT5 embeddings are sequence-level representations.
- SecondaryBERT embeddings retain residue-level positional information.
- Both embedding sets are used as input features for downstream neural network training.

---

## Execution

### ProtT5

```bash
python3 generate_prott5_embeddings.py \
    --input classification_human_mhc-I_HLA-A_blosum.csv
```

### SecondaryBERT

Process a single file:

```bash
python3 generate_secondary_bert_embeddings.py \
    --csv classification_human_mhc-I_HLA-A_ss.csv
```

Process all human datasets:

```bash
python3 generate_secondary_bert_embeddings.py \
    --species human
```

Force regeneration:

```bash
python3 generate_secondary_bert_embeddings.py \
    --species human \
    --force
```

---

## Summary

This module transforms protein sequence and secondary structure information into high-dimensional numerical representations suitable for deep learning. ProtT5 provides sequence-level biochemical representations, while SecondaryBERT provides residue-level structural representations. Together they form the feature space used by downstream neural network models.