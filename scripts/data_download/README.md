# Data download – IEDB epitopes

This module downloads epitope data from the IEDB API and prepares the raw dataset used by the NAP-CNB pipeline.

## Source database

Data are retrieved from the Immune Epitope Database (IEDB):

https://www.iedb.org

Using the public API:

- https://query-api.iedb.org/mhc_export
- https://query-api.iedb.org/tcell_export

## Downloaded data

The script downloads epitopes for:

Species:
- Homo sapiens
- Mus musculus

MHC classes:
- MHC class I
- MHC class II

Haplotypes:

Human  
- HLA-A
- HLA-B
- HLA-C
- HLA-DR
- HLA-DQ
- HLA-DP

Mouse  
- H2-K
- H2-D
- H2-L
- H2-IA
- H2-IE

Only **positive assays** are retrieved.

## Output structure

The data are stored in:
data/data_raw/


Each folder contains:

### Full export
mhc_export_full.csv
tcell_export_full.csv


Raw data returned by the API.

### Processed outputs
mhc_unique_epitopes.csv
tcell_unique_epitopes.csv
merged_unique_epitopes.csv
mhc_epitope_counts.csv
tcell_epitope_counts.csv


These files contain:

- canonical epitope sequences
- start and end positions
- antigen identifiers
- protein identifiers
- epitope frequency counts


## Role in the NAP-CNB pipeline

This script represents **Step 1 of the pipeline**:

1. Download epitope data from IEDB  
2. Generate curated epitope tables  
3. Provide input data for downstream feature extraction and model training
