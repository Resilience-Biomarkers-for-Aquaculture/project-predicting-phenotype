#!/usr/bin/env python3
"""
Trait Classification Model using Predictive Genes

This script builds a classification model using the top predictive genes identified
in the batch effect removal analysis to classify samples with unknown traits.
The model provides confidence scores and can be used for prediction on new samples.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_data_and_predictive_genes():
    """Load the corrected gene expression data and top predictive genes"""
    print("Loading data and predictive genes...")
    
    # Load the corrected gene expression data (using centered method as it had best CV score)
    gene_data = pd.read_csv('data/processed/gene_counts_centered_corrected.tsv', sep='\t', index_col=0)
    
    # Load the top predictive genes
    top_genes = pd.read_csv('data/processed/top_predictive_genes_centered.csv')
    
    # Load sample metadata
    sample_metadata = pd.read_csv('data/processed/sample_metadata.csv', index_col=0)
    
    print(f"Loaded {gene_data.shape[0]} genes and {gene_data.shape[1]} samples")
    print(f"Using top {len(top_genes)} predictive genes")
    
    return gene_data, top_genes, sample_metadata

def prepare_training_data(gene_data, top_genes, sample_metadata, n_top_genes=50):
    """Prepare training data using top predictive genes"""
    print(f"Preparing training data with top {n_top_genes} genes...")
    
    # Get the top N most predictive genes
    top_n_genes = top_genes.head(n_top_genes)['gene_id'].tolist()
    
    # Filter gene data to only include these genes
    X = gene_data.loc[top_n_genes].T  # Transpose so samples are rows
    
    # Get trait labels
    y = sample_metadata.loc[X.index, 'trait']
    
    # Filter out any samples with unknown traits
    valid_mask = (y != 'unknown') & (y.notna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Convert traits to binary (sensitive=1, tolerant=0)
    y_binary = (y == 'sensitive').astype(int)
    
    print(f"Training data shape: {X.shape}")
    print(f"Trait distribution: {y_binary.value_counts().to_dict()}")
    
    return X, y_binary, top_n_genes

def train_multiple_models(X, y):
    """Train multiple classification models and compare performance"""
    print("Training multiple classification models...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Evaluate performance
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'scaler': scaler if name == 'SVM' else None,
            'test_accuracy': (y_pred == y_test).mean(),
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
        
        print(f"  Test Accuracy: {results[name]['test_accuracy']:.3f}")
        print(f"  CV Accuracy: {results[name]['cv_accuracy']:.3f} (+/- {results[name]['cv_std'] * 2:.3f})")
    
    return results, X_train, X_test, y_train, y_test

def evaluate_models(results):
    """Evaluate and compare model performance"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*60)
    
    # Create comparison table
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Model': name,
            'Test Accuracy': f"{result['test_accuracy']:.3f}",
            'CV Accuracy': f"{result['cv_accuracy']:.3f}",
            'CV Std': f"{result['cv_std']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Plot ROC curves
    plt.figure(figsize=(12, 8))
    
    for name, result in results.items():
        y_test = result['y_test']
        y_pred_proba = result['y_pred_proba']
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/processed/model_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Tolerant', 'Sensitive'],
                   yticklabels=['Tolerant', 'Sensitive'],
                   ax=axes[i])
        axes[i].set_title(f'{name}\nAccuracy: {result["test_accuracy"]:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('data/processed/model_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_best_model(results, top_genes, n_top_genes=50):
    """Save the best performing model for future use"""
    print("\nSaving best model...")
    
    # Find best model based on CV accuracy
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_accuracy'])
    best_result = results[best_model_name]
    
    print(f"Best model: {best_model_name}")
    print(f"CV Accuracy: {best_result['cv_accuracy']:.3f}")
    
    # Save the model
    model_filename = f'data/processed/best_trait_classifier_{best_model_name.lower().replace(" ", "_")}.joblib'
    scaler_filename = f'data/processed/best_trait_classifier_scaler_{best_model_name.lower().replace(" ", "_")}.joblib'
    
    joblib.dump(best_result['model'], model_filename)
    if best_result['scaler']:
        joblib.dump(best_result['scaler'], scaler_filename)
    
    # Save the top genes used for training
    top_genes_filename = f'data/processed/top_{n_top_genes}_predictive_genes_for_classification.csv'
    top_genes.head(n_top_genes).to_csv(top_genes_filename, index=False)
    
    print(f"Model saved to: {model_filename}")
    if best_result['scaler']:
        print(f"Scaler saved to: {scaler_filename}")
    print(f"Top genes saved to: {top_genes_filename}")
    
    return best_model_name, best_result

def create_prediction_function(best_model_name, best_result, top_genes, n_top_genes=50):
    """Create a function for predicting traits on new samples"""
    print("\nCreating prediction function...")
    
    # Get the top genes used for training
    top_n_genes = top_genes.head(n_top_genes)['gene_id'].tolist()
    
    def predict_trait(new_sample_data, confidence_threshold=0.6):
        """
        Predict trait for a new sample
        
        Parameters:
        -----------
        new_sample_data : pandas.DataFrame
            Gene expression data for the new sample (genes as rows, samples as columns)
        confidence_threshold : float
            Minimum confidence threshold for making a prediction (default: 0.6)
        
        Returns:
        --------
        dict : Prediction results with trait, confidence, and probability scores
        """
        # Ensure the new sample has the required genes
        missing_genes = set(top_n_genes) - set(new_sample_data.index)
        if missing_genes:
            raise ValueError(f"Missing required genes: {missing_genes}")
        
        # Extract only the required genes and transpose
        X_new = new_sample_data.loc[top_n_genes].T
        
        # Scale the data if the model requires it
        if best_result['scaler']:
            X_new_scaled = best_result['scaler'].transform(X_new)
            y_pred_proba = best_result['model'].predict_proba(X_new_scaled)
        else:
            y_pred_proba = best_result['model'].predict_proba(X_new)
        
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
            'genes_used': top_n_genes
        }
    
    # Save the prediction function documentation
    doc_filename = 'data/processed/prediction_function_documentation.txt'
    with open(doc_filename, 'w') as f:
        f.write("TRAIT PREDICTION FUNCTION DOCUMENTATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"CV Accuracy: {best_result['cv_accuracy']:.3f}\n")
        f.write(f"Number of genes used: {n_top_genes}\n\n")
        f.write("Usage:\n")
        f.write("from scripts.trait_classification_model import predict_trait\n")
        f.write("result = predict_trait(new_sample_data, confidence_threshold=0.6)\n\n")
        f.write("Returns:\n")
        f.write("- predicted_trait: 'sensitive' or 'tolerant'\n")
        f.write("- confidence: confidence score (0-1)\n")
        f.write("- prediction_status: 'high_confidence' or 'low_confidence'\n")
        f.write("- probability_tolerant: probability of being tolerant\n")
        f.write("- probability_sensitive: probability of being sensitive\n")
        f.write("- genes_used: list of genes used for prediction\n")
    
    print(f"Documentation saved to: {doc_filename}")
    
    return predict_trait

def demonstrate_prediction_on_test_data(best_model_name, best_result, X_test, y_test, top_genes, n_top_genes=50):
    """Demonstrate prediction on test data to show confidence scores"""
    print("\nDemonstrating prediction on test data...")
    
    # Get the top genes used for training
    top_n_genes = top_genes.head(n_top_genes)['gene_id'].tolist()
    
    # Create a sample from test data
    test_sample_idx = 0
    test_sample_data = X_test.iloc[[test_sample_idx]]
    test_sample_trait = 'sensitive' if y_test.iloc[test_sample_idx] == 1 else 'tolerant'
    
    print(f"Test sample trait: {test_sample_trait}")
    
    # Make prediction
    if best_result['scaler']:
        test_sample_scaled = best_result['scaler'].transform(test_sample_data)
        y_pred_proba = best_result['model'].predict_proba(test_sample_scaled)
    else:
        y_pred_proba = best_result['model'].predict_proba(test_sample_data)
    
    proba_tolerant = y_pred_proba[0, 0]
    proba_sensitive = y_pred_proba[0, 1]
    
    if proba_sensitive > proba_tolerant:
        predicted_trait = 'sensitive'
        confidence = proba_sensitive
    else:
        predicted_trait = 'tolerant'
        confidence = proba_tolerant
    
    print(f"Predicted trait: {predicted_trait}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probability tolerant: {proba_tolerant:.3f}")
    print(f"Probability sensitive: {proba_sensitive:.3f}")
    print(f"Prediction correct: {predicted_trait == test_sample_trait}")

def main():
    """Main function to build and evaluate the classification model"""
    print("Building Trait Classification Model using Predictive Genes")
    print("=" * 60)
    
    # Load data
    gene_data, top_genes, sample_metadata = load_data_and_predictive_genes()
    
    # Prepare training data
    n_top_genes = 50  # Use top 50 genes for classification
    X, y, top_n_genes = prepare_training_data(gene_data, top_genes, sample_metadata, n_top_genes)
    
    # Train multiple models
    results, X_train, X_test, y_train, y_test = train_multiple_models(X, y)
    
    # Evaluate models
    evaluate_models(results)
    
    # Save best model
    best_model_name, best_result = save_best_model(results, top_genes, n_top_genes)
    
    # Create prediction function
    predict_trait = create_prediction_function(best_model_name, best_result, top_genes, n_top_genes)
    
    # Demonstrate prediction
    demonstrate_prediction_on_test_data(best_model_name, best_result, X_test, y_test, top_genes, n_top_genes)
    
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL BUILDING COMPLETE")
    print("="*60)
    print("The model can now be used to predict traits for new samples!")
    print("Use the saved model files and prediction function for new predictions.")

if __name__ == "__main__":
    main()
