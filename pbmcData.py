import scanpy as sc
import pandas as pd
import anndata
import pickle
import json
from sklearn.preprocessing import StandardScaler

# Stops pandas from wrapping over the columns and rows of the database #
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

annotations = pd.read_csv("data/PBMC/test/cell_types.csv")
annotations = annotations["coarse_cell_type"]
annotations = annotations.tolist()
print(annotations)

adata = sc.read_10x_mtx(
    "data/PBMC/test-data/",  # the folder containing your 3 files
    cache=True  # speeds up loading
)

print(adata.to_df().head())

adata.obs["true_annotations"] = annotations

sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=True)


# Normalize the data using count depth sampling technique #
# Normalize the data + apply a log1p transform to remove major outliers #
# log1p essentially is y = ln(1+x)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

adata.X = StandardScaler(with_mean=False).fit_transform(adata.X)

# Expects  log1p'd data #
sc.pp.highly_variable_genes(adata)
sc.pl.highly_variable_genes(adata)

# Principal Component Analysis #
# Simplifies data and helps delete dimensions #
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)

# Upload the final manipulated data to pickle #
# Allows for the model to just open the .pkl file instead of preprocessing the data each time #
with open('pickle/pbmcData.pkl', 'wb') as f:
    pickle.dump(adata, f)
