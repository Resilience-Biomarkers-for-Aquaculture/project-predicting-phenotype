# Scripts

This directory contains scripts for comprehensive gene expression analysis and phenotype prediction.

## Analysis Scripts

### `batch_effect_removal_and_analysis.py`
**Main analysis pipeline** that performs:
- Loads and merges gene count data from 5 experiments (38,828 genes, 217 samples)
- Removes batch effects using 3 methods (centered, quantile, z-score normalization)
- Performs PCA before and after batch correction
- Identifies top predictive genes for trait classification using Random Forest
- Generates visualization plots and saves corrected data

**Usage:**
```bash
python batch_effect_removal_and_analysis.py
```

**Outputs:**
- PCA plots: `../data/processed/pca_*.png`
- Predictive genes: `../data/processed/top_predictive_genes_*.csv`
- Corrected data: `../data/processed/gene_counts_*_corrected.tsv`

### `trait_classification_model.py`
**Machine learning model builder** that:
- Trains multiple classification models (Random Forest, Gradient Boosting, Logistic Regression, SVM)
- Compares model performance with cross-validation
- Saves the best performing model for production use
- Creates prediction functions with confidence scoring

**Usage:**
```bash
python trait_classification_model.py
```

**Outputs:**
- Best model: `../data/processed/best_trait_classifier_gradient_boosting.joblib`
- Required genes: `../data/processed/top_50_predictive_genes_for_classification.csv`
- Performance plots: `../data/processed/model_*.png`

### `predict_new_sample.py`
**Prediction script** for classifying new samples:
- Loads the trained classification model
- Demonstrates prediction on sample data
- Provides confidence scores and prediction status
- Example of how to use the model for new samples

**Usage:**
```bash
python predict_new_sample.py
```

### `normalize_and_group.py` (Legacy)
Original script for basic normalization and clustering.

**Usage:**
```bash
python normalize_and_group.py
```

## Workflow

1. **Run analysis pipeline**:
   ```bash
   python batch_effect_removal_and_analysis.py
   ```

2. **Build classification model**:
   ```bash
   python trait_classification_model.py
   ```

3. **Predict new samples**:
   ```bash
   python predict_new_sample.py
   ```

## Model Performance

The classification pipeline achieves:
- **87.9% cross-validation accuracy** with Gradient Boosting
- **50 predictive genes** identified from 38,828 total genes
- **High confidence predictions** with probability scores

## Documentation

See `../docs/classification_model_usage_guide.md` for detailed usage instructions and examples.
