import scanpy as sc
import pandas as pd
import anndata

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

adata_combined = anndata.concat(
    [healthy_data, diseased_data],
    join='outer',         # keeps all genes; fills missing values with zeros
    label='source',       # creates `obs['source']` column automatically if not set
    keys=['healthy', 'diseased'],
    index_unique=None     # keeps original cell barcodes
)

data = adata_combined.to_df()

print(data.head())

print(diseased_data.to_df().head())
print(healthy_data.to_df().head())
