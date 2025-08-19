
import os
import pandas as pd
from scipy.stats import ttest_ind

def load_metadata(metadata_path):
    meta = pd.read_csv(metadata_path)
    trait_map = dict(zip(meta['Experiment'], meta['Trait']))
    return trait_map

def load_all_expression(processed_dir):
    gene_expr = {}
    gene_names = {}
    for fname in os.listdir(processed_dir):
        if fname.endswith('.tsv'):
            fpath = os.path.join(processed_dir, fname)
            df = pd.read_csv(fpath, sep='\t')
            for idx, row in df.iterrows():
                gene_id = row['gene_id']
                gene_name = row['gene_name']
                if gene_id not in gene_expr:
                    gene_expr[gene_id] = {}
                    gene_names[gene_id] = gene_name
                for sample in df.columns[2:]:
                    val = row[sample]
                    gene_expr[gene_id][sample] = val
    return gene_expr, gene_names

def main():
    metadata_path = 'data/external/merged_metadata.csv'
    processed_dir = 'data/processed'
    trait_map = load_metadata(metadata_path)
    out_path = 'predictive_genes.tsv'
    print('Loading all expression data...')
    gene_expr, gene_names = load_all_expression(processed_dir)
    results = []
    for gene_id, sample_expr in gene_expr.items():
        tolerant = []
        sensitive = []
        for sample, val in sample_expr.items():
            trait = trait_map.get(sample)
            if trait == 'tolerant':
                tolerant.append(val)
            elif trait == 'sensitive':
                sensitive.append(val)
        if tolerant and sensitive:
            stat, pval = ttest_ind(tolerant, sensitive, equal_var=False)
            results.append({'gene_id': gene_id, 'gene_name': gene_names[gene_id], 'pval': pval})
    results.sort(key=lambda x: x['pval'])
    pd.DataFrame(results).to_csv(out_path, sep='\t', index=False)
    print(f'Results saved to {out_path}')

if __name__ == '__main__':
    main()
