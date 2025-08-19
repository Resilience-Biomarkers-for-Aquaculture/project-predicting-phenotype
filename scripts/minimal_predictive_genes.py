import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

genes_df = pd.read_csv('predictive_genes.tsv', sep='\t')
metadata = pd.read_csv('data/external/merged_metadata.csv')
trait_map = dict(zip(metadata['Experiment'], metadata['Trait']))

# Aggregate all expression data
gene_expr = {}
gene_names = {}
import os
for fname in os.listdir('data/processed'):
    if fname.endswith('.tsv'):
        df = pd.read_csv(os.path.join('data/processed', fname), sep='\t')
        for idx, row in df.iterrows():
            gene_id = row['gene_id']
            if gene_id not in gene_expr:
                gene_expr[gene_id] = {}
            for sample in df.columns[2:]:
                gene_expr[gene_id][sample] = row[sample]

# Build sample x gene matrix
samples = [s for s in trait_map.keys() if trait_map[s] in ['tolerant', 'sensitive']]
gene_list = genes_df['gene_id'].tolist()
X = np.array([[gene_expr[g].get(s, np.nan) for g in gene_list] for s in samples])
y = np.array([1 if trait_map[s]=='tolerant' else 0 for s in samples])

# Remove samples with missing data
good_idx = ~np.isnan(X).any(axis=1)
X = X[good_idx]
y = y[good_idx]

# Find minimal subset for >90% accuracy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n in range(1, min(100, X.shape[1])+1):
    X_sub = X[:, :n]
    scores = []
    for train, test in skf.split(X_sub, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_sub[train], y[train])
        pred = clf.predict(X_sub[test])
        scores.append(accuracy_score(y[test], pred))
    acc = np.mean(scores)
    if acc > 0.75:
        print(f"Minimal subset size for >75% accuracy: {n} genes")
        print("Gene IDs:", gene_list[:n])
        break
else:
    print("No subset of up to 100 genes achieves >75% accuracy.")
