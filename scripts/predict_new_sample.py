#!/usr/bin/env python3
"""
Predict Trait for New Sample

This script demonstrates how to use the saved classification model to predict
traits for new samples with unknown phenotypes.
"""

import pandas as pd
import joblib
import numpy as np

def load_model_and_genes():
    """Load the saved model and required genes"""
    print("Loading saved classification model...")
    
    # Load the best model (Gradient Boosting)
    model = joblib.load('data/processed/best_trait_classifier_gradient_boosting.joblib')
    
    # Load the top genes required for prediction
    top_genes = pd.read_csv('data/processed/top_50_predictive_genes_for_classification.csv')
    required_genes = top_genes['gene_id'].tolist()
    
    print(f"Model loaded successfully")
    print(f"Required genes: {len(required_genes)}")
    
    return model, required_genes

def predict_trait_for_new_sample(model, required_genes, new_sample_data, confidence_threshold=0.6):
    """
    Predict trait for a new sample
    
    Parameters:
    -----------
    model : trained classifier
        The saved classification model
    required_genes : list
        List of gene IDs required for prediction
    new_sample_data : pandas.DataFrame
        Gene expression data for the new sample (genes as rows, samples as columns)
    confidence_threshold : float
        Minimum confidence threshold for making a prediction (default: 0.6)
    
    Returns:
    --------
    dict : Prediction results with trait, confidence, and probability scores
    """
    print(f"\nPredicting trait for new sample...")
    
    # Ensure the new sample has the required genes
    missing_genes = set(required_genes) - set(new_sample_data.index)
    if missing_genes:
        raise ValueError(f"Missing required genes: {missing_genes}")
    
    # Extract only the required genes and transpose
    X_new = new_sample_data.loc[required_genes].T
    
    # Make prediction
    y_pred_proba = model.predict_proba(X_new)
    
    # Get prediction probabilities
    proba_tolerant = y_pred_proba[0, 0]
    proba_sensitive = y_pred_proba[0, 1]
    
    # Determine prediction and confidence
    if proba_sensitive > proba_tolerant:
        predicted_trait = 'sensitive'
        confidence = proba_sensitive
    else:
        predicted_trait = 'tolerant'
        confidence = proba_tolerant
    
    # Check if confidence meets threshold
    if confidence < confidence_threshold:
        prediction_status = 'low_confidence'
    else:
        prediction_status = 'high_confidence'
    
    return {
        'predicted_trait': predicted_trait,
        'confidence': confidence,
        'prediction_status': prediction_status,
        'probability_tolerant': proba_tolerant,
        'probability_sensitive': proba_sensitive,
        'genes_used': required_genes
    }

def create_sample_new_sample_data(required_genes):
    """Create a sample new sample data for demonstration"""
    print("Creating sample new sample data for demonstration...")
    
    # Create random gene expression data for demonstration
    np.random.seed(42)  # For reproducibility
    
    # Generate random expression values (log2 transformed counts)
    expression_values = np.random.normal(8, 2, len(required_genes))
    
    # Create DataFrame
    new_sample_data = pd.DataFrame({
        'new_sample_001': expression_values
    }, index=required_genes)
    
    print(f"Created sample data with {len(required_genes)} genes")
    return new_sample_data

def main():
    """Main function to demonstrate prediction on new samples"""
    print("Trait Prediction for New Samples")
    print("=" * 50)
    
    # Load model and required genes
    model, required_genes = load_model_and_genes()
    
    # Create sample new sample data for demonstration
    new_sample_data = create_sample_new_sample_data(required_genes)
    
    # Make prediction
    try:
        result = predict_trait_for_new_sample(model, required_genes, new_sample_data, confidence_threshold=0.6)
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Trait: {result['predicted_trait'].upper()}")
        print(f"Confidence: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        print(f"Prediction Status: {result['prediction_status']}")
        print(f"Probability Tolerant: {result['probability_tolerant']:.3f} ({result['probability_tolerant']*100:.1f}%)")
        print(f"Probability Sensitive: {result['probability_sensitive']:.3f} ({result['probability_sensitive']*100:.1f}%)")
        
        # Interpretation
        print(f"\nInterpretation:")
        if result['prediction_status'] == 'high_confidence':
            print(f"  ✓ High confidence prediction: {result['predicted_trait']}")
        else:
            print(f"  ⚠ Low confidence prediction - consider additional data or different threshold")
        
        print(f"  ✓ Model used {len(result['genes_used'])} predictive genes")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    print("\n" + "="*50)
    print("PREDICTION COMPLETE")
    print("="*50)
    print("To use this for your own samples:")
    print("1. Prepare gene expression data with the required genes")
    print("2. Call predict_trait_for_new_sample() function")
    print("3. Adjust confidence_threshold as needed")

if __name__ == "__main__":
    main()
