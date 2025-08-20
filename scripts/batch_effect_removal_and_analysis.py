#!/usr/bin/env python3
"""
Batch Effect Removal and Gene Expression Analysis Pipeline

This script analyzes gene expression data from 5 different experiments to:
1. Load and merge gene count data
2. Perform PCA before batch effect removal
3. Remove batch effects using multiple methods
4. Perform PCA after batch effect removal
5. Identify predictive genes for trait phenotypes (sensitive/tolerant)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_gene_counts():
    """Load and merge gene count data from all 5 experiments"""
    print("Loading gene count data...")
    
    # Load each experiment file
    experiment_files = [
        'data/external/01-salmon.merged.gene_counts.tsv',
        'data/external/02-salmon.merged.gene_counts.tsv',
        'data/external/03-salmon.merged.gene_counts.tsv',
        'data/external/04-salmon.merged.gene_counts.tsv',
        'data/external/05-salmon.merged.gene_counts.tsv'
    ]
    
    # Load metadata
    metadata = pd.read_csv('data/external/merged_metadata.csv')
    
    # Initialize merged data
    merged_data = None
    
    for i, file_path in enumerate(experiment_files):
        print(f"Loading experiment {i+1}...")
        
        # Load gene counts
        data = pd.read_csv(file_path, sep='\t')
        
        # Set gene_id as index
        data.set_index('gene_id', inplace=True)
        
        # Drop gene_name column if it exists
        if 'gene_name' in data.columns:
            data = data.drop('gene_name', axis=1)
        
        # Add experiment identifier to column names
        data.columns = [f"exp{i+1:02d}_{col}" for col in data.columns]
        
        if merged_data is None:
            merged_data = data
        else:
            # Merge on gene_id
            merged_data = merged_data.merge(data, left_index=True, right_index=True, how='inner')
    
    print(f"Loaded {merged_data.shape[0]} genes and {merged_data.shape[1]} samples")
    return merged_data, metadata

def prepare_metadata(merged_data, metadata):
    """Prepare metadata for analysis by mapping sample IDs"""
    print("Preparing metadata...")
    
    # Create sample mapping
    sample_mapping = {}
    
    # Process each experiment
    for i in range(1, 6):
        exp_cols = [col for col in merged_data.columns if col.startswith(f"exp{i:02d}_")]
        for col in exp_cols:
            sample_id = col.replace(f"exp{i:02d}_", "")
            sample_mapping[col] = {
                'experiment': f'exp{i:02d}',
                'sample_id': sample_id,
                'original_sample_id': sample_id
            }
    
    # Create metadata dataframe for samples
    sample_metadata = pd.DataFrame.from_dict(sample_mapping, orient='index')
    
    # Now map the traits from the original metadata
    # The Experiment column in metadata contains the sample IDs
    trait_mapping = {}
    for _, row in metadata.iterrows():
        if pd.notna(row['Experiment']):
            trait_mapping[row['Experiment']] = row['Trait']
    
    # Assign traits to samples
    sample_metadata['trait'] = sample_metadata['original_sample_id'].map(trait_mapping)
    
    # Fill unknown traits with experiment-based assignment as fallback
    unknown_mask = sample_metadata['trait'].isna()
    if unknown_mask.any():
        print(f"Warning: {unknown_mask.sum()} samples have unknown traits, using experiment-based assignment")
        
        for exp in ['exp01', 'exp02', 'exp03', 'exp04', 'exp05']:
            exp_mask = (sample_metadata['experiment'] == exp) & unknown_mask
            if exp_mask.any():
                # Assign traits based on experiment patterns from metadata
                if exp in ['exp01', 'exp02']:
                    sample_metadata.loc[exp_mask, 'trait'] = 'tolerant'
                elif exp in ['exp03', 'exp04']:
                    sample_metadata.loc[exp_mask, 'trait'] = 'sensitive'
                else:
                    sample_metadata.loc[exp_mask, 'trait'] = 'mixed'
    
    # Print trait distribution
    trait_counts = sample_metadata['trait'].value_counts()
    print("Trait distribution:")
    for trait, count in trait_counts.items():
        print(f"  {trait}: {count}")
    
    return sample_metadata

def perform_pca(data, title, sample_metadata=None):
    """Perform PCA and create visualization"""
    print(f"Performing PCA: {title}")
    
    # Transpose data so samples are rows
    data_t = data.T
    
    # Remove any infinite or NaN values
    data_t = data_t.replace([np.inf, -np.inf], np.nan)
    data_t = data_t.fillna(0)
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_t)
    
    # Perform PCA
    pca = PCA(n_components=min(10, data_scaled.shape[1], data_scaled.shape[0]))
    pca_result = pca.fit_transform(data_scaled)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot explained variance
    explained_var = pca.explained_variance_ratio_
    ax1.plot(range(1, len(explained_var) + 1), np.cumsum(explained_var), 'bo-')
    ax1.set_xlabel('Number of Components')
    ax1.set_ylabel('Cumulative Explained Variance Ratio')
    ax1.set_title('Explained Variance')
    ax1.grid(True)
    
    # Plot first two components
    if sample_metadata is not None and 'trait' in sample_metadata.columns:
        colors = {'tolerant': 'blue', 'sensitive': 'red', 'mixed': 'green', 'unknown': 'gray'}
        for trait in sample_metadata['trait'].unique():
            if pd.notna(trait):
                mask = sample_metadata['trait'] == trait
                ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                           c=colors.get(trait, 'gray'), label=trait, alpha=0.7)
    else:
        ax2.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    
    ax2.set_xlabel(f'PC1 ({explained_var[0]:.2%})')
    ax2.set_ylabel(f'PC2 ({explained_var[1]:.2%})')
    ax2.set_title('PCA: First Two Components')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'data/processed/pca_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, pca_result, explained_var

def remove_batch_effects(data, sample_metadata):
    """Remove batch effects using multiple methods"""
    print("Removing batch effects...")
    
    # Method 1: Simple mean centering per experiment
    print("Method 1: Mean centering per experiment")
    data_centered = data.copy()
    
    for exp in sample_metadata['experiment'].unique():
        exp_cols = sample_metadata[sample_metadata['experiment'] == exp].index
        if len(exp_cols) > 0:
            exp_data = data[exp_cols]
            exp_mean = exp_data.mean(axis=1)
            data_centered[exp_cols] = exp_data.sub(exp_mean, axis=0)
    
    # Method 2: Quantile normalization
    print("Method 2: Quantile normalization")
    data_quantile = data.copy()
    
    # Transpose for easier processing
    data_t = data_quantile.T
    
    # Sort each sample independently
    sorted_data = np.sort(data_t, axis=0)
    
    # Calculate mean across samples for each rank
    mean_sorted = np.mean(sorted_data, axis=0)
    
    # Reorder back to original order
    ranks = np.argsort(np.argsort(data_t, axis=0), axis=0)
    data_quantile_normalized = mean_sorted[ranks]
    
    # Transpose back
    data_quantile = pd.DataFrame(data_quantile_normalized.T, 
                                index=data.index, 
                                columns=data.columns)
    
    # Method 3: Z-score normalization per experiment
    print("Method 3: Z-score normalization per experiment")
    data_zscore = data.copy()
    
    for exp in sample_metadata['experiment'].unique():
        exp_cols = sample_metadata[sample_metadata['experiment'] == exp].index
        if len(exp_cols) > 0:
            # Filter columns that actually exist in the data
            available_cols = [col for col in exp_cols if col in data.columns]
            if len(available_cols) > 0:
                exp_data = data[available_cols]
                exp_mean = exp_data.mean(axis=1)
                exp_std = exp_data.std(axis=1)
                
                # Handle division by zero
                exp_std_safe = exp_std.replace(0, 1e-10)
                
                # Z-score normalization
                zscore_data = (exp_data - exp_mean) / exp_std_safe
                
                # Ensure the data is properly assigned
                for col in available_cols:
                    data_zscore[col] = zscore_data[col]
    
    return {
        'centered': data_centered,
        'quantile': data_quantile,
        'zscore': data_zscore
    }

def find_predictive_genes(data, sample_metadata, method_name):
    """Find genes that predict trait phenotype"""
    print(f"Finding predictive genes using {method_name}...")
    
    # Prepare data
    data_t = data.T
    data_t = data_t.replace([np.inf, -np.inf], np.nan)
    data_t = data_t.fillna(0)
    
    # Get trait labels
    traits = sample_metadata.loc[data_t.index, 'trait']
    
    # Filter out unknown traits and NaN values
    valid_mask = (traits != 'unknown') & (traits.notna())
    if valid_mask.sum() < 10:
        print(f"Not enough samples with known traits for {method_name} ({valid_mask.sum()} valid samples)")
        return None
    
    X = data_t[valid_mask]
    y = traits[valid_mask]
    
    # Convert traits to binary (sensitive=1, tolerant=0)
    y_binary = (y == 'sensitive').astype(int)
    
    print(f"Using {valid_mask.sum()} samples: {y_binary.sum()} sensitive, {len(y_binary) - y_binary.sum()} tolerant")
    
    # Use Random Forest to find feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf, X, y_binary, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    
    # Fit on full dataset
    rf.fit(X, y_binary)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'gene_id': data.index,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"{method_name} - CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return feature_importance

def main():
    """Main analysis pipeline"""
    print("Starting Gene Expression Analysis Pipeline")
    print("=" * 50)
    
    # Load data
    merged_data, metadata = load_gene_counts()
    sample_metadata = prepare_metadata(merged_data, metadata)
    
    # Save sample metadata
    sample_metadata.to_csv('data/processed/sample_metadata.csv')
    
    # Perform PCA before batch effect removal
    print("\n" + "="*50)
    print("PCA BEFORE BATCH EFFECT REMOVAL")
    print("="*50)
    pca_before, pca_result_before, explained_var_before = perform_pca(
        merged_data, "Before Batch Effect Removal", sample_metadata
    )
    
    # Remove batch effects
    print("\n" + "="*50)
    print("BATCH EFFECT REMOVAL")
    print("="*50)
    corrected_data = remove_batch_effects(merged_data, sample_metadata)
    
    # Perform PCA after each correction method
    for method_name, corrected_data_method in corrected_data.items():
        print(f"\nPCA after {method_name} correction:")
        pca_after, pca_result_after, explained_var_after = perform_pca(
            corrected_data_method, f"After {method_name.title()} Correction", sample_metadata
        )
        
        # Find predictive genes
        predictive_genes = find_predictive_genes(corrected_data_method, sample_metadata, method_name)
        
        if predictive_genes is not None:
            # Save top predictive genes
            top_genes = predictive_genes.head(100)
            top_genes.to_csv(f'data/processed/top_predictive_genes_{method_name}.csv', index=False)
            
            # Create visualization of top genes
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(top_genes.head(20))), top_genes.head(20)['importance'])
            plt.xlabel('Gene Rank')
            plt.ylabel('Feature Importance')
            plt.title(f'Top 20 Predictive Genes - {method_name.title()} Method')
            plt.xticks(range(len(top_genes.head(20))), top_genes.head(20)['gene_id'], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'data/processed/top_predictive_genes_{method_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # Save corrected data
    for method_name, corrected_data_method in corrected_data.items():
        corrected_data_method.to_csv(f'data/processed/gene_counts_{method_name}_corrected.tsv', sep='\t')
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("Results saved to data/processed/")
    print("Files created:")
    print("- sample_metadata.csv")
    print("- pca_*.png (PCA plots)")
    print("- top_predictive_genes_*.csv (gene rankings)")
    print("- top_predictive_genes_*.png (gene importance plots)")
    print("- gene_counts_*_corrected.tsv (corrected data)")

if __name__ == "__main__":
    main()
