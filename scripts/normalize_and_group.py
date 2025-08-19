"""
Normalize gene count matrices and group samples.
"""
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Paths
external_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/external"))
processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed"))

# Find all count matrix files
count_files = [f for f in os.listdir(external_dir) if f.endswith(".tsv")]

# Normalize and save
normalized_files = []
for fname in count_files:
    df = pd.read_csv(os.path.join(external_dir, fname), sep="\t")
    meta_cols = ["gene_id", "gene_name"]
    sample_cols = [col for col in df.columns if col not in meta_cols]
    scaler = StandardScaler()
    norm_samples = scaler.fit_transform(df[sample_cols])
    norm_df = pd.concat([df[meta_cols], pd.DataFrame(norm_samples, columns=sample_cols)], axis=1)
    out_path = os.path.join(processed_dir, fname.replace(".tsv", ".normalized.tsv"))
    norm_df.to_csv(out_path, sep="\t", index=False)
    normalized_files.append(out_path)

# Concatenate normalized matrices for clustering
all_samples = []
for f in normalized_files:
    df = pd.read_csv(f, sep="\t")
    sample_cols = [col for col in df.columns if col not in ["gene_id", "gene_name"]]
    # Each sample is a column, so transpose to have samples as rows
    sample_df = df[sample_cols].T
    sample_df.index.name = "Sample"
    all_samples.append(sample_df)
all_samples_df = pd.concat(all_samples)

# Combine all processed files, samples as rows
all_samples = []
sample_file_prefix = {}
for f in normalized_files:
    df = pd.read_csv(f, sep="\t")
    sample_cols = [col for col in df.columns if col not in ["gene_id", "gene_name"]]
    # Each sample is a column, so transpose to have samples as rows
    sample_df = df[sample_cols].T
    sample_df.index.name = "Sample"
    prefix = os.path.basename(f).split('-')[0]
    for s in sample_df.index:
        sample_file_prefix[s] = prefix
    all_samples.append(sample_df)
all_samples_df = pd.concat(all_samples)

# Group samples into 2-5 clusters
results = {}
for n_clusters in range(2, 3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(all_samples_df)
    results[n_clusters] = pd.DataFrame({"Sample": all_samples_df.index, "Cluster": labels})
    results[n_clusters].to_csv(os.path.join(processed_dir, f"sample_groups_{n_clusters}.csv"), index=False)


# Cluster into 2 groups only
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(all_samples_df)
results = pd.DataFrame({"Sample": all_samples_df.index, "Cluster": labels})
results.to_csv(os.path.join(processed_dir, "sample_groups_2.csv"), index=False)

print("Normalization and clustering into 2 groups complete.")
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Visualize clusters for n_clusters=5
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_samples_df)

# Color by file prefix (01-05)
prefixes = sorted(set(sample_file_prefix.values()))
prefix_colors = {p: plt.cm.tab10(i) for i, p in enumerate(prefixes)}
sample_colors = [prefix_colors[sample_file_prefix[s]] for s in all_samples_df.index]

# Cluster shapes: 0=square, 1=triangle
markers = {0: 's', 1: '^'}
plt.figure(figsize=(8,6))
for cluster in range(n_clusters):
    idxs = (labels == cluster)
    plt.scatter(pca_result[idxs,0], pca_result[idxs,1],
                c=[sample_colors[i] for i in range(len(sample_colors)) if idxs[i]],
                marker=markers[cluster],
                label=f"Cluster {cluster+1}", edgecolor='k', s=80)
for i, s in enumerate(all_samples_df.index):
    plt.text(pca_result[i,0], pca_result[i,1], s, fontsize=8, alpha=0.6)
plt.title("Sample Clusters (2 groups)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, "sample_clusters.png"))
plt.show()
labels = results[n_clusters]["Cluster"].values
samples = results[n_clusters]["Sample"].values

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_samples_df)

# Assign each sample to its source file for coloring

# Map each sample to its source file by tracking sample names per file
sample_to_file = {}
for idx, f in enumerate(normalized_files):
    df = pd.read_csv(f, sep="\t")
    sample_cols = [col for col in df.columns if col not in ["gene_id", "gene_name"]]
    for s in sample_cols:
        sample_to_file[s] = os.path.basename(f)
file_colors = {os.path.basename(f): plt.cm.tab10(i) for i, f in enumerate(normalized_files)}
sample_colors = [file_colors[sample_to_file[s]] if s in sample_to_file else 'gray' for s in samples]

plt.figure(figsize=(8,6))
for cluster in range(n_clusters):
    idxs = labels == cluster
    plt.scatter(pca_result[idxs,0], pca_result[idxs,1],
                c=[sample_colors[i] for i in range(len(samples)) if idxs[i]],
                label=f"Cluster {cluster+1}", edgecolor='k', s=80)
for i, s in enumerate(samples):
    plt.text(pca_result[i,0], pca_result[i,1], s, fontsize=8, alpha=0.6)
plt.title(f"Sample Clusters (n={n_clusters})")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(processed_dir, "sample_clusters.png"))
plt.show()
