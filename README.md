# NAP-CNB

NAP-CNB is a research project focused on the development of machine learning models for antigen and epitope prediction.

The project integrates public immunological datasets, protein sequence processing, secondary structure prediction, protein language models and neural networks for large-scale immunoinformatics analyses.

---

## Project goals

- Build large-scale epitope datasets from public repositories
- Generate classification datasets for MHC-I and MHC-II molecules
- Augment protein sequence datasets using biologically informed substitutions
- Extract representations from protein primary and secondary structure
- Train neural networks for epitope prediction
- Support future applications in cancer immunotherapy design

---

## Pipeline overview

```text
data_download
      ↓
data_classification
      ↓
data_augmentation
      ↓
 ┌───────────────┬────────────────┐
 │               │                │
 ▼               ▼                ▼
ProtT5      ProteinUnet      SecondaryBERT
(primary)   prediction       embeddings
 │               │                │
 └───────────────┴────────────────┘
                 ↓
                NN
```

---

## Repository structure

```text
NAP-CNB/
│
├── scripts/
│   │
│   ├── data_download/
│   │   Download and filter epitope datasets from IEDB.
│   │
│   ├── data_classification/
│   │   Generate labeled epitope and non-epitope windows.
│   │
│   ├── data_augmentation/
│   │   Generate BLOSUM62-based sequence variants.
│   │
│   ├── embeddings/
│   │   Generate ProtT5 and SecondaryBERT embeddings.
│   │
│   ├── secondary_structure_prediction/
│   │   UniRef clustering, ProteinUnet prediction,
│   │   dataset generation and BERT pretraining.
│   │
│   ├── secondaryBERT/
│   │   SecondaryBERT utilities and model scripts.
│   │
│   └── NN/
│       Neural network training and evaluation.
│
├── .gitignore
│
└── README.md
```

---

## Modules

### data_download

Downloads and filters epitope datasets from IEDB and generates standardized event tables for downstream processing.

### data_classification

Retrieves parent protein sequences and generates fixed-length classification windows labeled as epitope or non-epitope.

### data_augmentation

Expands datasets using BLOSUM62-based amino acid substitutions.

### embeddings

Generates numerical representations of proteins using:

- ProtT5 embeddings (primary structure)
- SecondaryBERT embeddings (secondary structure)

### secondary_structure_prediction

Builds the secondary-structure pretraining dataset:

- UniRef50 clustering with MMseqs2
- ProteinUnet secondary-structure prediction
- Dataset cleaning and validation
- SecondaryBERT MLM pretraining

### secondaryBERT

Contains SecondaryBERT-related scripts used for secondary-structure language modeling and embedding generation.

### NN

Contains downstream neural network training and evaluation pipelines.

---

## Technologies

- Python
- PyTorch
- TensorFlow / Keras
- Hugging Face Transformers
- ProtT5
- ProteinUnet
- MMseqs2
- NumPy
- Pandas
- Scikit-learn

---

## Current status

Implemented:

- Data download pipeline
- Classification dataset generation
- Data augmentation pipeline
- ProtT5 embedding generation
- UniRef50 clustering
- ProteinUnet secondary-structure prediction
- SecondaryBERT MLM pretraining
- SecondaryBERT embedding generation

In development:

- Neural network training and evaluation framework