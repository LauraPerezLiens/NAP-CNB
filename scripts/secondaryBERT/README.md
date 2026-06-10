# secondaryBERT

## Overview

This module builds and trains a BERT masked-language model using predicted protein secondary-structure sequences.

The pipeline starts from the UniRef50 protein database, reduces sequence redundancy by clustering representatives at 30% sequence identity, predicts secondary structure with ProteinUnet, prepares a C/H/E masked-language-modeling dataset, and finally trains a BERT model.

Although the original input database is UniRef50, sequences are clustered at 30% sequence identity using MMseqs2. The resulting representative sequence set is referred to throughout this documentation as **"UniRef30"**, although it is not the official UniRef30 database.

For the UniRef30 dataset used in this project:

```text
Representative sequences: 23,127,616
Chunk size: 100,000 sequences
Generated chunks: 232
```

---

## Pipeline role

```text
UniRef50 FASTA
    ↓
MMseqs2 clustering at 30% identity
    ↓
UniRef30 representative sequences
    ↓
FASTA chunking
    ↓
ProteinUnet secondary-structure prediction
    ↓
C/H/E secondary-structure dataset
    ↓
BERT MLM pretraining
```

---

## Directory structure

Expected data structure:

```text
/home/nap/lperez_nn/data/data_uniref50/
├── bert_mlm_dataset/
│   ├── all_cleaned.parquet
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   └── vocab.txt
│
├── clustering/
│   ├── uniref50_c30_cluster.tsv
│   └── uniref50_c30_rep_seq.fasta
│
├── mmseqs2/
│   └── mmseqs/
│
└── proteinunet_prediction/
    ├── chunks/
    │   ├── chunk_00001.fasta
    │   ├── chunk_00002.fasta
    │   └── ...
    │
    ├── outputs/
    │   ├── chunk_00001.csv
    │   ├── chunk_00001_skipped.csv
    │   ├── chunk_00002.csv
    │   ├── chunk_00002_skipped.csv
    │   └── ...
    │
    └── missing_secondary_structure_audit.csv
```

---

## Scripts

### 1. `cluster_uniref50_to_uniref30_mmseqs.py`

Clusters UniRef50 sequences with MMseqs2 to obtain a representative sequence set at 30% sequence identity.

#### Input

```text
/data_uniref50/uniref50.fasta
```

#### Output

```text
/data_uniref50/clustering/uniref50_c30_cluster.tsv
/data_uniref50/clustering/uniref50_c30_rep_seq.fasta
```

#### Main parameters

```text
--min-seq-id 0.3
--coverage 0.8
--cov-mode 1
--threads 16
```

#### Run

```bash
python3 cluster_uniref50_to_uniref30_mmseqs.py
```

---

### 2. `create_uniref30_fasta_chunks.py`

Splits the UniRef30 representative FASTA file into smaller FASTA chunks.

This step is kept separate from secondary-structure prediction so that chunks can be processed independently and failed chunks can be rerun without repeating the clustering or splitting steps.

#### Input

```text
/data_uniref50/clustering/uniref50_c30_rep_seq.fasta
```

#### Output

```text
/data_uniref50/proteinunet_prediction/chunks/chunk_00001.fasta
/data_uniref50/proteinunet_prediction/chunks/chunk_00002.fasta
...
```

#### Default chunk size

```text
100,000 sequences per chunk
```

#### Run

```bash
python3 create_uniref30_fasta_chunks.py
```

Optional:

```bash
python3 create_uniref30_fasta_chunks.py \
  --sequences-per-chunk 100000
```

---

### 3. `predict_uniref30_secondary_structure.py`

Predicts secondary structure for one FASTA chunk using ProteinUnet.

The model outputs secondary structure as a string over the alphabet:

```text
C H E
```

where:

```text
C = coil
H = helix
E = beta strand
```

ProteinUnet expects sequences containing only the 20 standard amino acids. Sequences containing non-standard residues such as `X`, `U`, `B`, `Z`, `J` or `O` are skipped and written to a separate skipped file.

The prediction script writes both the successful predictions and skipped/failed proteins. The output CSV contains `id`, `protein_id`, `sequence` and `secondary_structure`, while the skipped CSV records `protein_id`, `reason` and `details`.

#### Input

```text
/data_uniref50/proteinunet_prediction/chunks/chunk_XXXXX.fasta
```

#### Output

```text
/data_uniref50/proteinunet_prediction/outputs/chunk_XXXXX.csv
/data_uniref50/proteinunet_prediction/outputs/chunk_XXXXX_skipped.csv
```

#### Run

```bash
python3 predict_uniref30_secondary_structure.py \
  -i /home/nap/lperez_nn/data/data_uniref50/proteinunet_prediction/chunks/chunk_00001.fasta
```

Optional custom output name:

```bash
python3 predict_uniref30_secondary_structure.py \
  -i /home/nap/lperez_nn/data/data_uniref50/proteinunet_prediction/chunks/chunk_00001.fasta \
  -o chunk_00001.csv
```

---

### 4. `audit_uniref30_secondary_structure_predictions.py`

Audits the ProteinUnet prediction step by comparing every input FASTA chunk against its corresponding output CSV.

For each protein present in the FASTA but absent from the prediction output, the script checks whether the protein contains non-standard amino acids.

#### Input

```text
/data_uniref50/proteinunet_prediction/chunks/chunk_XXXXX.fasta
/data_uniref50/proteinunet_prediction/outputs/chunk_XXXXX.csv
```

#### Output

```text
/data_uniref50/proteinunet_prediction/missing_secondary_structure_audit.csv
```

#### Output columns

```text
chunk
protein_id
sequence_length
missing_reason
invalid_residues
sequence
```

Possible `missing_reason` values include:

```text
non_standard_residues
missing_output_csv
missing_unknown_reason
```

#### Run

```bash
python3 audit_uniref30_secondary_structure_predictions.py
```

---

### 5. `build_secondary_mlm_dataset.py`

Builds the final secondary-structure MLM dataset for BERT pretraining.

This script merges all ProteinUnet prediction CSV files, removes invalid rows, checks that sequence length and secondary-structure length match, converts C/H/E strings into space-separated token sequences, and splits the dataset into train, validation and test sets.

#### Input

```text
/data_uniref50/proteinunet_prediction/outputs/chunk_XXXXX.csv
```

Skipped files are ignored:

```text
chunk_XXXXX_skipped.csv
```

#### Output

```text
/data_uniref50/bert_mlm_dataset/all_cleaned.parquet
/data_uniref50/bert_mlm_dataset/train.parquet
/data_uniref50/bert_mlm_dataset/val.parquet
/data_uniref50/bert_mlm_dataset/test.parquet
/data_uniref50/bert_mlm_dataset/vocab.txt
```

#### Vocabulary

```text
[PAD]
[UNK]
[CLS]
[SEP]
[MASK]
C
H
E
```

#### Default split

```text
train: 80%
validation: 10%
test: 10%
random seed: 42
```

The validation size is interpreted as a fraction of the full cleaned dataset, not as a fraction of the remaining train/validation subset.

#### Run

```bash
python3 build_secondary_mlm_dataset.py
```

---

### 6. `train_secondary_mlm.py`

Trains a BERT masked-language model using the C/H/E secondary-structure dataset.

This script does not generate the parquet files. It expects them to already exist in:

```text
/data_uniref50/bert_mlm_dataset/
```

Proteins without valid ProteinUnet predictions are already excluded before this step, because only proteins present in the parquet dataset are used for training.

#### Input

```text
/data_uniref50/bert_mlm_dataset/train.parquet
/data_uniref50/bert_mlm_dataset/val.parquet
/data_uniref50/bert_mlm_dataset/vocab.txt
```

#### Output

```text
/home/nap/lperez_nn/model/secondary_bert_mlm
```

#### Training dataset size

```text
Train sequences:      18,374,944
Validation sequences: 2,296,868
Test sequences:       2,296,869
Total sequences:      22,968,681
```

#### Model configuration

```text
hidden_size: 768
num_hidden_layers: 12
num_attention_heads: 12
intermediate_size: 3072
max_position_embeddings: 30
```

Parameter description:

- **hidden_size (768)**  
  Dimensionality of the transformer hidden representations. Each token
  (C, H or E) is encoded into a 768-dimensional embedding vector.

- **num_hidden_layers (12)**  
  Number of stacked transformer encoder layers. Deeper models can capture
  more complex structural patterns and long-range dependencies.

- **num_attention_heads (12)**  
  Number of self-attention heads per transformer layer. Multi-head attention
  allows the model to learn different contextual relationships between
  secondary-structure positions simultaneously.

- **intermediate_size (3072)**  
  Size of the feed-forward network inside each transformer block.
  Following the original BERT-base architecture, this value is four times
  the hidden size (4 × 768).

- **max_position_embeddings (30)**  
  Maximum sequence length supported by the model, including special tokens
  such as `[CLS]` and `[SEP]`.

The resulting architecture contains approximately 85 million trainable
parameters.

#### Training parameters

```text
window_size: 25
max_len: 30
mlm_probability: 0.15
batch_size: 32
num_epochs: 1
learning_rate: 1e-4
weight_decay: 0.01
warmup_ratio: 0.05
```

Parameter description:

- **window_size (25)**  
    Maximum number of secondary-structure positions retained from each sequence
    during MLM pretraining. Sequences longer than 25 positions are truncated
    before tokenization.

- **max_len (30)**  
  Maximum tokenized sequence length accepted by BERT. The value includes
  room for special tokens (`[CLS]`, `[SEP]`) and padding.

- **mlm_probability (0.15)**  
  Fraction of tokens randomly selected for masked-language modeling (MLM)
  training. This follows the standard BERT masking strategy where 15% of
  tokens are chosen as prediction targets.

- **batch_size (32)**  
  Per-device batch size used during training. The effective batch size
  scales with the number of GPUs available.

- **num_epochs (1)**  
  Number of complete passes through the training dataset. Given the size
  of the UniRef30-derived dataset (>22 million sequences), a single epoch
  corresponds to hundreds of thousands of optimization steps.

- **learning_rate (1e-4)**  
  Initial learning rate used by the AdamW optimizer.

- **weight_decay (0.01)**  
  L2 regularization applied to model weights to improve generalization and
  reduce overfitting.

- **warmup_ratio (0.05)**  
  Fraction of total training steps used for learning-rate warmup before
  reaching the target learning rate.

#### Run

```bash
python3 train_secondary_mlm.py
```

Optional:

```bash
python3 train_secondary_mlm.py \
  --batch-size 32 \
  --num-epochs 1 \
  --learning-rate 1e-4
```

---

## Complete execution order

```bash
python3 cluster_uniref50_to_uniref30_mmseqs.py

python3 create_uniref30_fasta_chunks.py

python3 predict_uniref30_secondary_structure.py \
  -i /home/nap/lperez_nn/data/data_uniref50/proteinunet_prediction/chunks/chunk_00001.fasta

python3 audit_uniref30_secondary_structure_predictions.py

python3 build_secondary_mlm_dataset.py

python3 train_secondary_mlm.py
```

For all chunks, the ProteinUnet prediction step should be launched once per FASTA chunk.

Example loop:

```bash
for fasta in /home/nap/lperez_nn/data/data_uniref50/proteinunet_prediction/chunks/chunk_*.fasta
do
    python3 predict_uniref30_secondary_structure.py -i "$fasta"
done
```

For parallel execution, each chunk can be submitted as an independent job.

---

## Handling of non-standard amino acids

ProteinUnet uses one-hot encoding over the 20 standard amino acids only:

```text
A D N R C E Q G H I L K M F P S T W Y V
```

Proteins containing residues outside this set are not passed to the model. They are written to the corresponding skipped file:

```text
proteinunet_prediction/outputs/chunk_XXXXX_skipped.csv
```

This avoids generating unreliable secondary-structure predictions for sequences that the model cannot encode.

---

## Dataset cleaning

Before BERT pretraining, the merged ProteinUnet outputs are cleaned.

Rows are retained only if:

```text
secondary_structure contains only C/H/E
sequence length == secondary_structure length
secondary_structure is not empty
```

The cleaned dataset is saved as:

```text
bert_mlm_dataset/all_cleaned.parquet
```

Then it is split into:

```text
train.parquet
val.parquet
test.parquet
```

---

## Notes

* The pipeline uses UniRef50 as the original source database.
* After MMseqs2 clustering at 30% sequence identity, the working representative set is referred to as UniRef30.
* ProteinUnet prediction is the main computational bottleneck.
* FASTA chunking is intentionally separated from prediction to simplify parallel execution and reruns.
* BERT is trained only on valid C/H/E secondary-structure strings.
* Sequences skipped by ProteinUnet are not included in the BERT pretraining dataset.
