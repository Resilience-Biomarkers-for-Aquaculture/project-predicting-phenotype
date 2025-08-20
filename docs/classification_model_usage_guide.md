# Trait Classification Model Usage Guide

## Overview

This guide explains how to use the trained classification model to predict traits (sensitive/tolerant) for new samples based on their gene expression profiles. The model uses the top 50 most predictive genes identified from the batch effect removal analysis.

## Model Performance

The **Gradient Boosting** model achieved the best performance:
- **Cross-validation accuracy**: 87.9% (±5.3%)
- **Test accuracy**: 79.5%
- **Number of predictive genes used**: 50

## Files Generated

The following files were created during model training:

### Model Files
- `best_trait_classifier_gradient_boosting.joblib` - The trained classification model
- `top_50_predictive_genes_for_classification.csv` - List of genes required for prediction

### Documentation
- `prediction_function_documentation.txt` - Detailed usage instructions
- `model_roc_curves.png` - ROC curves comparing all models
- `model_confusion_matrices.png` - Confusion matrices for all models

## How to Use the Model

### 1. Basic Prediction

```python
import joblib
import pandas as pd

# Load the model and required genes
model = joblib.load('data/processed/best_trait_classifier_gradient_boosting.joblib')
top_genes = pd.read_csv('data/processed/top_50_predictive_genes_for_classification.csv')
required_genes = top_genes['gene_id'].tolist()

# Prepare your new sample data
# Your data should have genes as rows and samples as columns
new_sample_data = your_gene_expression_data  # pandas DataFrame

# Ensure you have all required genes
missing_genes = set(required_genes) - set(new_sample_data.index)
if missing_genes:
    print(f"Missing genes: {missing_genes}")
    # Handle missing genes appropriately

# Extract only the required genes and transpose
X_new = new_sample_data.loc[required_genes].T

# Make prediction
y_pred_proba = model.predict_proba(X_new)

# Get probabilities
proba_tolerant = y_pred_proba[0, 0]
proba_sensitive = y_pred_proba[0, 1]

# Determine prediction
if proba_sensitive > proba_tolerant:
    predicted_trait = 'sensitive'
    confidence = proba_sensitive
else:
    predicted_trait = 'tolerant'
    confidence = proba_tolerant

print(f"Predicted trait: {predicted_trait}")
print(f"Confidence: {confidence:.3f}")
```

### 2. Using the Prediction Function

For convenience, you can use the provided prediction function:

```python
from scripts.trait_classification_model import predict_trait

# Load the saved model and create prediction function
# (This requires running the classification model script first)

result = predict_trait(new_sample_data, confidence_threshold=0.6)

print(f"Predicted trait: {result['predicted_trait']}")
print(f"Confidence: {result['confidence']}")
print(f"Status: {result['prediction_status']}")
```

### 3. Batch Prediction

To predict traits for multiple samples at once:

```python
def predict_multiple_samples(model, required_genes, samples_data, confidence_threshold=0.6):
    """Predict traits for multiple samples"""
    results = []
    
    for sample_id in samples_data.columns:
        sample_data = samples_data[[sample_id]]
        
        try:
            # Extract required genes and transpose
            X_sample = sample_data.loc[required_genes].T
            
            # Make prediction
            y_pred_proba = model.predict_proba(X_sample)
            
            proba_tolerant = y_pred_proba[0, 0]
            proba_sensitive = y_pred_proba[0, 1]
            
            if proba_sensitive > proba_tolerant:
                predicted_trait = 'sensitive'
                confidence = proba_sensitive
            else:
                predicted_trait = 'tolerant'
                confidence = proba_tolerant
            
            prediction_status = 'high_confidence' if confidence >= confidence_threshold else 'low_confidence'
            
            results.append({
                'sample_id': sample_id,
                'predicted_trait': predicted_trait,
                'confidence': confidence,
                'prediction_status': prediction_status,
                'probability_tolerant': proba_tolerant,
                'probability_sensitive': proba_sensitive
            })
            
        except Exception as e:
            results.append({
                'sample_id': sample_id,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

# Usage
results_df = predict_multiple_samples(model, required_genes, your_samples_data)
results_df.to_csv('prediction_results.csv', index=False)
```

## Data Requirements

### Input Format
Your gene expression data should be:
- **Format**: pandas DataFrame
- **Rows**: Gene IDs (must include all 50 required genes)
- **Columns**: Sample IDs
- **Values**: Gene expression values (log2 transformed counts recommended)

### Required Genes
The model requires exactly these 50 genes (see `top_50_predictive_genes_for_classification.csv`):
- LOC111114951, LOC111126363, LOC111113400, LOC111128086, LOC111114036
- LOC111133668, LOC111121995, LOC111107835, LOC111134436, LOC111100783
- LOC111100642, LOC111118969, LOC111105635, LOC111105685, LOC111101754
- LOC111123040, LOC111118133, LOC111115212, LOC111104438, LOC111115308
- LOC111100157, LOC111105221, LOC111137339, LOC111124620, LOC111111014
- LOC111134390, LOC111102291, LOC111109744, LOC111110616, LOC111128611
- LOC111133765, LOC111104351, LOC111138160, LOC111107764, LOC111102362
- LOC111114987, LOC111138174, LOC111112321, LOC111109496, LOC111129143
- LOC111123295, LOC111130573, LOC111116380, LOC111134486, LOC111123472
- LOC111102533, LOC111132369, LOC111113809, LOC111123529, LOC111129034

## Confidence Thresholds

The model provides confidence scores that help assess prediction reliability:

- **High confidence (≥0.6)**: Reliable prediction
- **Medium confidence (0.5-0.6)**: Moderate reliability
- **Low confidence (<0.5)**: Unreliable prediction

You can adjust the confidence threshold based on your needs:
- **Lower threshold (e.g., 0.5)**: More predictions, lower reliability
- **Higher threshold (e.g., 0.7)**: Fewer predictions, higher reliability

## Model Limitations

1. **Gene Requirements**: All 50 predictive genes must be present
2. **Data Quality**: Expression values should be properly normalized
3. **Batch Effects**: Data should be processed similarly to training data
4. **Sample Similarity**: New samples should be from similar experimental conditions

## Troubleshooting

### Common Issues

1. **Missing Genes**
   ```
   Error: Missing required genes: [LOC111114951, LOC111126363]
   Solution: Ensure your data contains all required genes
   ```

2. **Data Format Issues**
   ```
   Error: Columns must be same length as key
   Solution: Check that your data has genes as rows and samples as columns
   ```

3. **Low Confidence Predictions**
   ```
   Warning: Low confidence prediction
   Solution: Check data quality or adjust confidence threshold
   ```

### Getting Help

If you encounter issues:
1. Check the data format matches requirements
2. Verify all required genes are present
3. Ensure expression values are properly normalized
4. Review the model performance metrics

## Example Workflow

1. **Prepare your data**:
   ```python
   # Load your gene expression data
   your_data = pd.read_csv('your_gene_expression.csv', index_col=0)
   
   # Ensure it has the required format
   print(f"Data shape: {your_data.shape}")
   print(f"Genes as rows: {your_data.index.name == 'gene_id'}")
   ```

2. **Load the model**:
   ```python
   model = joblib.load('data/processed/best_trait_classifier_gradient_boosting.joblib')
   ```

3. **Make predictions**:
   ```python
   results = predict_trait_for_new_sample(model, required_genes, your_data)
   ```

4. **Interpret results**:
   ```python
   if results['prediction_status'] == 'high_confidence':
       print(f"High confidence prediction: {results['predicted_trait']}")
   else:
       print("Low confidence - consider additional data")
   ```

## Performance Expectations

Based on the model's cross-validation performance:
- **Expected accuracy**: ~88% on similar data
- **Confidence correlation**: Higher confidence generally means higher accuracy
- **Error patterns**: Model may be more accurate for certain trait types

## Future Improvements

The model can be enhanced by:
1. **More training data**: Additional samples with known traits
2. **Feature selection**: Optimizing the number of predictive genes
3. **Ensemble methods**: Combining multiple model predictions
4. **Domain adaptation**: Adapting to different experimental conditions
