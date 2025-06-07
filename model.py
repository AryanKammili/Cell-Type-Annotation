import math
from pprint import pprint

import scanpy as sc
from collections import defaultdict
import pickle
import numpy as np

marker_dict = {
    "T cell": {
        "positive": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7", "LEF1", "CD8A", "CD8B", "GZMH", "PRF1"],
        "negative": ["CD8A", "CD14", "MS4A1", "FCER1A", "NKG7", "CD4", "MS4A1", "FCER1A"]
    },
    "B cell": {
        "positive": ["MS4A1", "CD79A", "CD19", "CD22", "BANK1"],
        "negative": ["CD3D", "CD8A", "CD14", "NKG7"]
    },
    "NK cell": {
        "positive": ["NKG7", "GNLY", "KLRD1", "GZMB", "PRF1"],
        "negative": ["CD3D", "CD14", "MS4A1", "FCER1A"]
    },
    "monocyte": {
        "positive": ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'FCGR3A'],
        "negative": ["CD3D", "MS4A1", "FCER1A", "NKG7", "CD79A"]
    },
    "Dendritic cell": {
        "positive": ["FCER1A", "CST3", "ITGAX", "IRF8", "HLA-DRA"],
        "negative": ["CD3D", "CD14", "MS4A1", "NKG7"]
    },
    "macrophage":{
        'positive': ['CD68', 'CD163', 'MRC1', 'CSF1R', 'APOE', 'SPP1'],
        'negative': ['CD3D', 'CD79A', 'NKG7', 'CLEC9A', 'PPBP']
    },
    "Plasma cell": {
        "positive": ["SDC1", "XBP1", "MZB1", "TNFRSF17", "PRDM1"],
        "negative": ["CD3D", "CD14", "NKG7", "MS4A1"]
    },
    "Megakaryocyte": {
        "positive": ["PPBP", "PF4", "ITGA2B", "GP9", "MKI67"],
        "negative": ["CD3D", "MS4A1", "CD14", "FCER1A"]
    }
}

positive_markers_dict = {ctype: markers["positive"] for ctype, markers in marker_dict.items()}
negative_markers_dict = {ctype: markers["negative"] for ctype, markers in marker_dict.items()}

def get_positive_marker_gene_specificity(positive_genes):
    gene_counts = defaultdict(int)
    for genes in positive_genes.values():
        unique_genes = set(genes)
        for gene in unique_genes:
            gene_counts[gene] = gene_counts[gene] + 1
    return {gene: 1.0 / count for gene, count in gene_counts.items()}
def get_negative_marker_gene_specificity(negative_genes):
    gene_counts = defaultdict(int)
    for genes in negative_genes.values():
        unique_genes = set(genes)
        for gene in unique_genes:
            gene_counts[gene] = gene_counts[gene] + 1
    return {gene: 1.0 / count for gene, count in gene_counts.items()}


positive_specificity_weights = get_positive_marker_gene_specificity(positive_markers_dict)
negative_specificity_weights = get_negative_marker_gene_specificity(negative_markers_dict)

with open('testdata.pkl', 'rb') as f:
    adata = pickle.load(f)


# Use neighbors so that leiden alogrithim can be used #
sc.pp.neighbors(adata, random_state=42)
sc.tl.umap(adata, random_state=42)

# Specify the resolution which determines how many communities is created by leiden alogrihthim #
# Save to a key to be plotted later #
sc.tl.leiden(adata, key_added=f"leiden_res_{0.5:4.2f}", resolution=0.1, flavor="igraph", random_state=42)

cluster_scores = defaultdict(dict)

for cluster in adata.obs["leiden_res_0.50"].unique():

    community = adata[adata.obs["leiden_res_0.50"] == cluster]

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

        pprint(pos_expr)
        pos_score = np.dot(pos_expr, pos_weights) / np.sqrt(len(pos_genes))

        if neg_genes:
            neg_idx = [community.var_names.get_loc(g) for g in neg_genes]
            neg_expr = community.X[:, neg_idx].mean(axis=0)

            neg_score = np.dot(neg_expr, neg_weights) / np.sqrt(len(neg_genes))

        else:
            neg_score = 0.0

        pprint(pos_weights)
        pprint(neg_weights)
        final_score = pos_score - neg_score
        cluster_scores[cluster][cell_type] = float(final_score)

pprint(cluster_scores)

cluster_annotations = {}
for cluster, scores in cluster_scores.items():

    if not scores:
        cluster_annotations[cluster] = "unknown"
        continue

    best_type = max(scores, key=scores.get)
    cluster_annotations[cluster] = best_type
    print(f"Cluster {cluster} → {best_type} (score: {scores[best_type]:.2f})")

adata.obs['manual_annotations'] = adata.obs["leiden_res_0.50"].map(cluster_annotations)
adata.obs['manual_annotations'] = adata.obs['manual_annotations'].fillna('Unknown')

sc.pl.umap(adata, color="manual_annotations", legend_loc="on data")
sc.pl.umap(adata, color="leiden_res_0.50", legend_loc="on data")

print(adata.obs['manual_annotations'].value_counts())

# Drop cells with no annotation
predicted = adata.obs["manual_annotations"]
actual = adata.obs["true_annotations"]

score = 0.0
for i in range(len(predicted)):
    if predicted[i] == actual[i]:
        score += 1.0

    else:
        print("predicted: " + predicted[i] + " | actual: " + actual[i])

score /= len(predicted)

print(score)

