import scanpy as sc
import pandas as pd
import anndata
import pickle

# Stops pandas from wrapping over the columns and rows of the database #
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

healthy_data = sc.read_10x_mtx(
    "data/healthy/filtered_feature_bc_matrix/",  # the folder containing your 3 files
    cache=True  # speeds up loading
)

diseased_data = sc.read_10x_mtx(
    "data/diseased/filtered_feature_bc_matrix/",  # the folder containing your 3 files
    cache=True  # speeds up loading
)

healthy_data.obs['source'] = 'healthy'
diseased_data.obs['source'] = 'diseased'

adata = anndata.concat(
    [healthy_data, diseased_data],
    join='outer',         # keeps all genes; fills missing values with zeros
    label='source',       # creates `obs['source']` column automatically if not set
    keys=['healthy', 'diseased'],
    index_unique=None     # keeps original cell barcodes
)

database = adata.to_df()

print(database.head())

sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=True)

# Normalize the data using count depth sampling technique #
# Normalize the data + apply a log1p transform to remove major outliers #
# log1p essentially is y = ln(1+x)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Expects  log1p'd data #
sc.pp.highly_variable_genes(adata)
sc.pl.highly_variable_genes(adata)

# Principal Component Analysis #
# Simplifies data and helps delete dimensions #
sc.tl.pca(adata)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True)


# Upload the final manipulated data to pickle #
# Allows for the model to just open the .pkl file instead of preprocessing the data each time #
with open('data.pkl', 'wb') as f:
    pickle.dump(adata, f)

