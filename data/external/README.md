# External Data

This directory contains raw gene count matrices and metadata from external data sources used as input for the analysis pipeline.

## Gene Count Matrices

**Files:** `01-salmon.merged.gene_counts.tsv` through `05-salmon.merged.gene_counts.tsv`
- **5 experiment files** containing raw gene expression counts
- Generated from Salmon transcript quantification 
- Contains 38,828 genes across multiple samples per experiment
- File sizes range from 2.5MB to 10MB

**Format:**
- Tab-separated values (TSV)
- Columns: `gene_id`, `gene_name`, sample columns
- Values: Raw read counts per gene per sample

## Sample Metadata

**File:** `merged_metadata.csv` (18KB, 219 lines)
- Sample trait annotations (sensitive/tolerant phenotypes)
- Experiment assignments for each sample
- Used to map samples to their known phenotypes for model training

**Format:**
- CSV with columns for experiment IDs and trait assignments
- 219 samples total with trait information
- Used as ground truth for classification model training

## Data Processing

These raw files are processed by:
1. `../../scripts/batch_effect_removal_and_analysis.py` - Merges all 5 experiments and applies batch correction
2. Sample metadata is used to assign known traits for supervised learning

## Quality Information

- **Total genes**: 38,828 across all experiments
- **Total samples**: 217 (after merging and filtering)
- **Trait distribution**: 120 tolerant, 97 sensitive samples
- **Experiments**: 5 independent datasets requiring batch effect removal

See `../processed` for batch-corrected and analysis-ready data.