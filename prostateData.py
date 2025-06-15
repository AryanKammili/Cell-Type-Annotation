import scanpy as sc
import pandas as pd
import anndata
import pickle
import json
from sklearn.preprocessing import StandardScaler
import scanpy as sc
import pandas as pd
import numpy as np
from io import StringIO

# Stops pandas from wrapping over the columns and rows of the database #
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

annotations = pd.read_csv("data/Prostate/SRA742961_SRS3565197.clusters.txt", delimiter=" ", header=None, names=["Barcode", "coarse_cell_type"])
annotations = annotations["coarse_cell_type"]
annotations = annotations.tolist()

label_mapping = {
    0: "Basal Cells",
    1: "Basal Cells",
    2: "Basal Cells",
    3: "Basal Cells",
    4: "Cholangiocytes",
    5: "Basal Cells",
    6: "Cholangiocytes",
    7: "Ductal Cells",
    8: "Keratinocytes",
    9: "Ducatal Cells",
    10: "Fibroblasts",
    11: "Unknown",
    12: "Unknown",
    13: "Cholangiocytes",
    14: "Unknown",
    15: "Unknown"
}

annotations = [label_mapping.get(label, label) for label in annotations]

df = pd.read_csv("data/Prostate/SRA742961_SRS3565197.mat", delimiter="\t").transpose()

adata = anndata.AnnData(df)
adata.var_names = df.columns
adata.obs_names = df.index

if "True" in np.isinf(df.values):
    print("has infinity")

adata.obs["true_annotations"] = annotations

# Normalize the data using count depth sampling technique #
# Normalize the data + apply a log1p transform to remove major outliers #
# log1p essentially is y = ln(1+x)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

adata.X = StandardScaler(with_mean=False).fit_transform(adata.X)

# Expects  log1p'd data #
# sc.pp.highly_variable_genes(adata)
# sc.pl.highly_variable_genes(adata)

# Principal Component Analysis #
# Simplifies data and helps delete dimensions #
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

# Upload the final manipulated data to pickle #
# Allows for the model to just open the .pkl file instead of preprocessing the data each time #
with open('prostate.pkl', 'wb') as f:
    pickle.dump(adata, f)
