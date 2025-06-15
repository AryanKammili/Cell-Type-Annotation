import scanpy as sc
from collections import defaultdict
import pickle
import numpy as np

key = "Prostate"

marker_dict = {

    "PBMC": {
        "B cells": {
            "positive": ["MS4A1", "CD79A", "CD19", "CD22", "BANK1"],
            "negative": ["CD3D", "CD8A", "CD14", "NKG7"]
        },
        "T cells": {
            "positive": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B", "TRAC"],
            "negative": ["KLRD1", "NKG7", "GNLY", "PECAM1", "EPCAM"]
        },
        "NK cells": {
            "positive": ["GNLY", "NKG7", "KLRD1", "KLRF1", "PRF1"],
            "negative": ["CD3D", "CD3E", "CD4", "CD8B"]
        },
        "monocyte": {
            "positive": ['CD14', 'LYZ', 'S100A8', 'S100A9', 'VCAN', 'FCGR3A'],
            "negative": ["CD3D", "MS4A1", "FCER1A", "NKG7", "CD79A"]
        },
        "Dendritic cell": {
            "positive": ["FCER1A", "CST3", "ITGAX", "IRF8", "HLA-DRA"],
            "negative": ["CD3D", "CD14", "MS4A1", "NKG7"]
        },
        "macrophage": {
            'positive': ['CD68', 'CD163', 'MRC1', 'CSF1R', 'APOE', 'SPP1'],
            'negative': ['CD3D', 'CD79A', 'NKG7', 'CLEC9A', 'PPBP']
        },
        "Megakaryocyte": {
            "positive": ["PPBP", "PF4", "ITGA2B", "GP9", "MKI67"],
            "negative": ["CD3D", "MS4A1", "CD14", "FCER1A"]
        },
        "hematopoietic cell": {
            "positive": ["PTPRC", "SPI1", "ITGB2", "CD34", "CD43", "CD99", "LYZ"],
            "negative": ["EPCAM", "CDH1", "PECAM1", "ACTA2", "PDGFRA", "VIM", "GFAP"]
        }
    },

    "Kidney": {
        "Distal tubule": {
            "positive": ["SLC12A3", "CALB1", "FXYD4", "CLDN8"],
            "negative": ["UMOD", "SLC5A2", "NPHS1", "PDGFRB"]
        },
        "Collecting duct": {
            "positive": ["AQP2", "AVPR2", "ATP6V1B1", "SLC4A1"],
            "negative": ["SLC12A3", "SLC5A2", "NPHS1"]
        },
        "Endothelial cell": {
            "positive": ["PECAM1", "VWF", "CD34", "MCAM"],
            "negative": ["SLC34A1", "NPHS1", "AQP2", "EPCAM", "KRT8", "KRT18", "CDH1"]
        },
        "Fibroblast (Kidney)": {
            "positive": ["PDGFRA", "PDGFRB", "COL1A1", "VIM", "DCN"],
            "negative": ["SLC34A1", "NPHS1", "AQP2", "PECAM1"]
        },
        "Epithelial cell": {
            "positive": ["EPCAM", "KRT8", "KRT18", "CDH1"],
            "negative": ["PTPRC", "PECAM1", "COL1A1", "VWF", "CD34", "MCAM"]
        }
    },

    "Prostate": {
        "Basal cell": {
            "positive": ["TP63", "KRT5", "KRT14", "EGFR"],
            "negative": ["KLK3", "ACPP", "AR", "KRT8", "KRT18"]
        },
        "Ductal epithelial cell": {
            "positive": ["EPCAM", "KRT8", "KRT18", "PSCA", "MUC1"],
            "negative": ["TP63", "KRT5", "ACTA2"]
        },
        "Cholangiocyte": {
            "positive": ["KRT7", "KRT19", "EPCAM", "CFTR", "SOX9"],
            "negative": ["TP63", "KRT14", "ACTA2"]
        },
        "Fibroblast": {
            "positive": ["COL1A1", "DCN", "PDGFRA", "FAP", "VIM"],
            "negative": ["KRT5", "KRT8", "EPCAM"]
        },
        "Keratinocyte": {
            "positive": ["KRT1", "KRT10", "DSG1", "IVL", "TP63"],
            "negative": ["EPCAM", "KRT8", "VIM"]
        }
    }

}

marker_dict = marker_dict.get(key)

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

with open('prostate.pkl', 'rb') as f:
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

adata.obs['manual_annotations'] = adata.obs["leiden_res_0.50"].map(cluster_annotations)

sc.pl.umap(adata, color="manual_annotations", legend_loc="on data")
sc.pl.umap(adata, color="leiden_res_0.50", legend_loc="on data")

# Drop cells with no annotation
predicted = adata.obs["manual_annotations"]
actual = adata.obs["true_annotations"]

label_mapping = {
    "columnar epithelial cell": "Epithelial cell",
    "cuboidal epithelial cell": "Epithelial cell",
    "columnar/cuboidal epithelial cell": "Epithelial cell",
    "endothelial cell": "Endothelial cell",
    "epithelial cell": "Epithelial cell",
    "T cell": "T cells",
    "B cell": "B cells",
    "NK cell": "NK cells",
}

actual = [label_mapping.get(label, label) for label in actual]

score = 0.0
for i in range(len(predicted)):
    if predicted[i] == actual[i]:
        score += 1.0

    else:
        print("predicted: " + predicted[i] + " | actual: " + actual[i])

score /= len(predicted)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(actual, predicted))
