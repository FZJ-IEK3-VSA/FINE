# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os 
import sys


import FINE as fn
import FINE.IOManagement.xarrayIO as xrIO 
import pandas as pd

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # How to save an energy system model instance and set it back up? 
# 
# **Xarray and NetCDF files to the rescue!** The data contained within an Energy System Model (ESM) instance is vast and complex. Saving it directly is not possible. It can, however, be saved as a NetCDF file which supports complex data structures. 
# 
# #### What exactly is NetCDF? 
# NetCDF (Network Common Data Format) is a set of software libraries and machine-independent data formats that support the creation, access, and sharing of array-oriented scientific data. It is also a community standard for sharing scientific data. 
# 
# #### Python modules that support working with NetCDF files:
# 1. netcdf4-python: Official Python interface to netCDF files
# 2. PyNIO: To access different file formats such as netCDF, HDF, and GRIB
# 3. xarray: Based on NumPy and pandas
# 
# Note: xarray module is used here. 
# 
# For our use case, the following functionalities are provided: 
# * Conversion of ESM instance to xarray dataset. Additionally, possible to save this dataset as NetCDF file in a desired folder, with a desired file name. 
# * Conversion of xarray dataset/saved NetCDF file back to ESM instance.
# 
# #### Structure of the xarray dataset: 
# 
# <img src="xarray_fine.png" style="width: 1000px;"/>
# 
# %% [markdown]
# ## Conversion of ESM instance to xarray dataset and saving it as a NetCDF file
# %% [markdown]
# #### STEP 1. Set up your  ESM instance 

# %%
from getModel import getModel
esM = getModel()
esM.optimize()

# %% [markdown]
# #### STEP 2. Conversion to xarray datasets and saving as NetCDF file
# You can convert the esM to xarray datasets with `esm_to_datasets` and access Input, Parameters or Result.
# 

# %%
import warnings
warnings.filterwarnings('ignore')
esm_datasets = xrIO.esm_to_datasets(esM)


# %%
esM.componentModelingDict['SourceSinkModel'].optSummary


# %%
esM2 = xrIO.datasets_to_esm(esm_datasets)
for model in esM2.componentModelingDict.keys():
    print(getattr(esM2.componentModelingDict[model],'optSummary'))


# %%
xr_dss = esm_datasets['Results']

#pd.DataFrame.from_dict(xr_dss['SourceSinkModel'])
xr_dss['SourceSinkModel']


# %%
# xr_dataset['ts_operationRateMax'].loc['Source, Wind (onshore)', :, :]
esm_datasets["Input"]["Sink"]["Industry site"]["ts_operationRateFix"].to_dataframe().unstack()


# %%
esm_datasets["Results"]["SourceSinkModel"]["Electricity market"]


# %%
esm_datasets["Parameters"]

# %% [markdown]
# Or save it directly to NetCDF with `esm_to_netcdf`:

# %%
_ = xrIO.esm_to_netcdf(esM, outputFileName="my_esm.nc")

# %% [markdown]
# #### STEP 3. Load esM from NetCDF file or xarray datasets
# 
# You can load an esM from file with `netcdf_to_esm`.

# %%
# esm_from_file = xrIO.netcdf_to_esm("my_esm.nc")  # Not implemented


# %%
# esm_from_file.getComponentAttribute('Wind (onshore)', 'operationRateMax')

# %% [markdown]
# Or from datasets with `datasets_to_esm`.

# %%
# Alternative to giving an xr dataset, you could pass the full path to your NETCDF file 
esm_from_datasets = xrIO.datasets_to_esm(esm_datasets)


# %%
esm_from_datasets.getComponentAttribute('Industry site', 'operationRateFix')


# %%
#esm_datasets["Results"]["SourceSinkModel"]["Wind (onshore)"]


# %%



# %%



