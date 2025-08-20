# Project: Predicting Phenotype

This repository contains comprehensive gene expression analysis tools for predicting phenotypes (sensitive/tolerant traits) using machine learning models based on predictive gene expression profiles.

## Overview

The project analyzes gene expression data from 5 experiments to:
1. Remove batch effects using multiple normalization methods
2. Identify predictive genes for trait classification
3. Build machine learning models for phenotype prediction
4. Provide tools for classifying new samples with confidence scores

## Key Results

- **38,828 genes** analyzed across **217 samples**
- **Top 50 predictive genes** identified for trait classification
- **87.9% accuracy** achieved with Gradient Boosting classifier
- **Production-ready model** for predicting traits on new samples

## Structure

- `data/`: All project data (raw, external, processed)
- `scripts/`: Analysis and modeling scripts
- `docs/`: Documentation and usage guides

## Quick Start

### 1. Batch Effect Removal and Analysis
```bash
python scripts/batch_effect_removal_and_analysis.py
```

### 2. Build Classification Model
```bash
python scripts/trait_classification_model.py
```

### 3. Predict New Samples
```bash
python scripts/predict_new_sample.py
```

## Model Performance

The best performing model (Gradient Boosting) achieves:
- **Cross-validation accuracy**: 87.9% (±5.3%)
- **Test accuracy**: 79.5%
- **Uses only 50 genes** for prediction

## Files Generated

### Analysis Results
- PCA plots before/after batch correction
- Top predictive genes rankings
- Gene importance visualizations
- Corrected gene expression matrices

### Classification Model
- Trained classifier: `data/processed/best_trait_classifier_gradient_boosting.joblib`
- Required genes: `data/processed/top_50_predictive_genes_for_classification.csv`
- Usage guide: `docs/classification_model_usage_guide.md`

## Documentation

- **Complete usage guide**: `docs/classification_model_usage_guide.md`
- **Model performance**: ROC curves and confusion matrices
- **Prediction examples**: Sample scripts and functions

## Dependencies

- pandas, numpy, matplotlib, seaborn
- scikit-learn, joblib
- Standard Python data science stack

See subdirectory README files for detailed information.