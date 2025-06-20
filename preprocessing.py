import csv
from collections import defaultdict


def load_markers(key):
    markers = []
    with open("marker_dictionary.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Tissue"] == key:
                markers.append(row)

    markers_dictionary = defaultdict(lambda: {"positive": [], "negative": []})
    for row in markers:
        cell_type = row["Cell Type"]
        marker_type = row["Marker Type"]
        marker = row["Marker"]
        markers_dictionary[cell_type][marker_type].append(marker)

    return dict(markers_dictionary)


def load_ensembl_to_symbol_map(tsv_file):
    ensembl_to_symbol = {}

    with open(tsv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            symbol = row["Approved symbol"]

            # Current Ensembl ID
            if row["Ensembl gene ID"]:
                ensembl_to_symbol[row["Ensembl gene ID"]] = symbol
    return ensembl_to_symbol


def load_predicted_to_labels_map(data):
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

    return [label_mapping.get(label, label) for label in data]
