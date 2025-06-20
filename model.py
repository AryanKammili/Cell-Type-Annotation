import scanpy as sc
from collections import defaultdict
import pickle
import numpy as np
from preprocessing import load_markers
from preprocessing import load_ensembl_to_symbol_map
from preprocessing import load_predicted_to_labels_map

key = "PBMC"
marker_dict = load_markers(key)

positive_markers_dict = {ctype: markers["positive"] for ctype, markers in marker_dict.items()}
negative_markers_dict = {ctype: markers["negative"] for ctype, markers in marker_dict.items()}


def get_marker_gene_specificity(genes):
    gene_counts = defaultdict(int)
    for genes in genes.values():
        unique_genes = set(genes)
        for gene in unique_genes:
            gene_counts[gene] = gene_counts[gene] + 1
    return {gene: 1.0 / count for gene, count in gene_counts.items()}


positive_specificity_weights = get_marker_gene_specificity(positive_markers_dict)
negative_specificity_weights = get_marker_gene_specificity(negative_markers_dict)

ensembl_to_symbol_map = load_ensembl_to_symbol_map("custom.txt")

with open('pickle/pbmcData.pkl', 'rb') as f:
    adata = pickle.load(f)

    mapped_genes = adata.var_names.map(lambda x: ensembl_to_symbol_map.get(x, x))
    adata.var_names = mapped_genes

# Use neighbors so that leiden alogrithim can be used #
sc.pp.neighbors(adata, random_state=42)
sc.tl.umap(adata, random_state=42)

# Specify the resolution which determines how many communities is created by leiden alogrihthim #
# Save to a key to be plotted later #
sc.tl.leiden(adata, key_added=f"clustering", resolution=0.1, flavor="igraph", random_state=42)

cluster_scores = defaultdict(dict)

for cluster in adata.obs["clustering"].unique():

    community = adata[adata.obs["clustering"] == cluster]

    for cell_type, markers in marker_dict.items():
        pos_genes = list(set(markers['positive']).intersection(community.var_names))
        neg_genes = list(set(markers['negative']).intersection(community.var_names))

        if not pos_genes:
            continue

        pos_weights = np.array([positive_specificity_weights.get(g, 0) for g in pos_genes])
        neg_weights = np.array([negative_specificity_weights.get(g, 0) for g in neg_genes])

        if pos_weights.sum() == 0:
            continue

        pos_score, neg_score = 0.0, 0.0

        pos_idx = [community.var_names.get_loc(g) for g in pos_genes]
        pos_expr = community.X[:, pos_idx].mean(axis=0)

        pos_score = np.dot(pos_expr, pos_weights) / np.sqrt(len(pos_genes))

        if neg_genes:
            neg_idx = [community.var_names.get_loc(g) for g in neg_genes]
            neg_expr = community.X[:, neg_idx].mean(axis=0)

            neg_score = np.dot(neg_expr, neg_weights) / np.sqrt(len(neg_genes))

        else:
            neg_score = 0.0

        final_score = pos_score - neg_score
        cluster_scores[cluster][cell_type] = float(final_score)

cluster_annotations = {}
for cluster, scores in cluster_scores.items():

    if not scores:
        cluster_annotations[cluster] = "unknown"
        continue

    best_type = max(scores, key=scores.get)
    cluster_annotations[cluster] = best_type

adata.obs['manual_annotations'] = adata.obs["clustering"].map(cluster_annotations)

sc.pl.umap(adata, color="manual_annotations", legend_loc="on data")
sc.pl.umap(adata, color="clustering", legend_loc="on data")

# Drop cells with no annotation
predicted = adata.obs["manual_annotations"]
actual = adata.obs["true_annotations"]

actual = load_predicted_to_labels_map(actual)

score = 0.0
for i in range(len(predicted)):
    if predicted[i] != actual[i]:
        print("predicted: " + predicted[i] + " | actual: " + actual[i])

score /= len(predicted)

from sklearn.metrics import classification_report

print(classification_report(actual, predicted))
