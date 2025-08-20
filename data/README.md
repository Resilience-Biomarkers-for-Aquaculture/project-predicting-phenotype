# Data Directory

This folder contains all data used in the project, organized by processing stage and source.

## Directory Structure

- `external/`: Raw gene count matrices and metadata from external sources
- `processed/`: Analysis outputs, corrected data, and trained models
- `raw/`: Unmodified raw data files

## Data Overview

**Input Data:**
- **5 experiments** with gene expression data
- **38,828 genes** analyzed across **217 samples**
- **219 samples** with trait metadata (sensitive/tolerant)

**Processed Outputs:**
- Batch-corrected gene expression matrices
- PCA analysis results and visualizations
- Top predictive gene rankings
- Trained classification models

## Key Files

### Analysis Results
- `processed/pca_*.png`: PCA plots before/after batch correction
- `processed/top_predictive_genes_*.csv`: Gene importance rankings
- `processed/gene_counts_*_corrected.tsv`: Batch-corrected expression data

### Classification Model
- `processed/best_trait_classifier_gradient_boosting.joblib`: Trained model
- `processed/top_50_predictive_genes_for_classification.csv`: Required genes
- `processed/model_*.png`: Performance evaluation plots

### Metadata
- `external/merged_metadata.csv`: Sample trait annotations
- `processed/sample_metadata.csv`: Processed sample information

See subdirectory README files for detailed descriptions of each data type.