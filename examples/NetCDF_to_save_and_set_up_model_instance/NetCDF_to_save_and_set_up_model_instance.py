# -*- coding: utf-8 -*-
# %%
import os
import sys

import FINE as fn
import FINE.IOManagement.xarrayIO as xrIO

# # %load_ext autoreload
# # %autoreload 2

# %% [markdown]
# # How to save an energy system model instance and set it back up?
#
# **Xarray and NetCDF files to the rescue!** The data contained within an Energy System Model (ESM) instance and the optimization results is vast and complex. Saving it directly is not possible. It can, however, be saved as a NetCDF file which supports complex data structures.
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
# #### High-level structure of the data:
#
# <img src="overall_structure.png" style="width: 1000px;"/>
#
#
# #### Structure of xarray dataset - For a non-transmission component:
#
# <img src="non_transmission.png" style="width: 1000px;"/>
#
# #### Structure of xarray dataset - For a transmission component:
#
# <img src="transmission.png" style="width: 1000px;"/>
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
esm_datasets = xrIO.writeEnergySystemModelToDatasets(esM)

# %%
esm_datasets["Input"]["Sink"]["Industry site"][
    "ts_operationRateFix"
].to_dataframe().unstack()

# %%
esm_datasets["Results"]["SourceSinkModel"]["Electricity market"]

# %%
esm_datasets["Parameters"]

# %% [markdown]
# Or save it directly to NetCDF with `esm_to_netcdf`:

# %%
_ = xrIO.writeEnergySystemModelToNetCDF(
    esM, outputFilePath="my_esm.nc", overwriteExisting=True
)

# %% [markdown]
# #### STEP 3. Load esM from NetCDF file or xarray datasets
#
# You can load an esM from file with `netcdf_to_esm`.

# %%
esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel("my_esm.nc")

# %%
esm_from_netcdf.getComponentAttribute("Industry site", "operationRateFix")

# %% [markdown]
# Or from datasets with `datasets_to_esm`.

# %%
esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

# %%
esm_from_datasets.getComponentAttribute("Industry site", "operationRateFix")

# %%
esm_datasets["Results"]["SourceSinkModel"]["Electricity market"][
    "operationVariablesOptimum"
]
