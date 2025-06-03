from pprint import pprint

import scanpy as sc
from collections import defaultdict
import pickle

marker_dict = {
    "CD4+ T cells": {
        "positive": ["CD3D", "CD3E", "CD4", "IL7R", "CCR7", "LEF1"],
        "negative": ["CD8A", "CD14", "MS4A1", "FCER1A", "NKG7"]
    },
    "CD8+ T cells": {
        "positive": ["CD3D", "CD3E", "CD8A", "CD8B", "GZMH", "PRF1"],
        "negative": ["CD4", "MS4A1", "CD14", "FCER1A"]
    },
    "B cells": {
        "positive": ["MS4A1", "CD79A", "CD19", "CD22", "BANK1"],
        "negative": ["CD3D", "CD8A", "CD14", "NKG7"]
    },
    "NK cells": {
        "positive": ["NKG7", "GNLY", "KLRD1", "GZMB", "PRF1"],
        "negative": ["CD3D", "CD14", "MS4A1", "FCER1A"]
    },
    "Monocytes": {
        "positive": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN"],
        "negative": ["CD3D", "MS4A1", "FCER1A", "NKG7"]
    },
    "Dendritic cells": {
        "positive": ["FCER1A", "CST3", "ITGAX", "IRF8", "HLA-DRA"],
        "negative": ["CD3D", "CD14", "MS4A1", "NKG7"]
    },
    "Plasma cells": {
        "positive": ["SDC1", "XBP1", "MZB1", "TNFRSF17", "PRDM1"],
        "negative": ["CD3D", "CD14", "NKG7", "MS4A1"]
    },
    "Megakaryocytes": {
        "positive": ["PPBP", "PF4", "ITGA2B", "GP9", "MKI67"],
        "negative": ["CD3D", "MS4A1", "CD14", "FCER1A"]
    }
}

with open('data.pkl', 'rb') as f:
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

        if not pos_genes: continue

        pos_score, neg_score = 0.0, 0.0

        # Get positive marker expression
        pos_expr = community[:, pos_genes].X
        pos_score = pos_expr.mean() / len(pos_genes)

        # Get negative marker expression
        if neg_genes:
            neg_expr = community[:, neg_genes].X
            neg_score = neg_expr.mean() / len(neg_genes)
        else:
            neg_score = 0

        # Skip cell type if expression is too low (optional)
        if pos_score < 0.05:
            continue

        final_score = pos_score - 0.5 * neg_score
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



