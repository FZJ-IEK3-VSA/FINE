#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import pandas as pd
import xarray as xr

import FINE as fn
import FINE.IOManagement.xarrayIO as xrIO

cwd = os.getcwd()

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# # Workflow for spatial aggregation of an energy system model
#
# This example notebook shows how model regions can be merged to obtain fewer regions
#
# <img src="spatial_aggregation_for_example_notebook.png" style="width: 1000px;"/>
#
# The figure above dipicts the basic idea behind spatial aggregation. The term spatial aggregation refers to grouping
# and subsequently merging the regions that are similar in some sense (NOTE: Please look into the documentation for
# different methods to group regions).

# ## STEP 1. Set up your ESM instance

# In[2]:


from getModel import getModel

esM = getModel()


# In[3]:


esM.locations


# ## STEP 2. Spatial grouping of regions

# In[4]:


# The input data to spatial aggregation are esM instance and the shapefile containing model regions' geometries
SHAPEFILE_PATH = os.path.join(
    cwd,
    "..",
    "Multi-regional_Energy_System_Workflow/InputData/SpatialData/ShapeFiles/clusteredRegions.shp",
)


# In[5]:


# Once the regions are grouped, the data witin each region group needs to be aggregated. Through the aggregation_function_dict
# parameter, it is possible to define/change how each variable show be aggregated. Please refer to the documentation for more
# information.

aggregation_function_dict = {
    "operationRateMax": ("mean", "capacityMax"),
    "operationRateFix": ("sum", None),
}


# In[6]:


# You can provide a path to save the results with desired file names. Two files are saved - a shapefile containing
# the merged region geometries and a netcdf file containing the aggregated esM instance data.
shp_name = "aggregated_regions"
aggregated_xr_filename = "aggregated_xr_ds.nc"


# In[7]:


# Spatial aggregation
aggregated_esM = esM.aggregateSpatially(
    shapefile=SHAPEFILE_PATH,
    grouping_mode="parameter_based",
    n_groups=4,
    aggregatedResultsPath=os.path.join(cwd, "output_data"),
    aggregation_function_dict=aggregation_function_dict,
    shp_name=shp_name,
    aggregated_xr_filename=aggregated_xr_filename,
    solver="glpk",
)


# In[8]:


# Original spatial resolution
fig, ax = fn.plotLocations(SHAPEFILE_PATH, plotLocNames=True, indexColumn="index")


# In[9]:


# Spatial resolution after aggregation
AGGREGATED_SHP_PATH = os.path.join(cwd, "output_data", f"{shp_name}.shp")

fig, ax = fn.plotLocations(AGGREGATED_SHP_PATH, plotLocNames=True, indexColumn="space")


# In[10]:


# The locations in the resulting esM instance are now 4.
aggregated_esM.locations


# In[11]:


#  And corresponding data has also been aggregated
aggregated_esM.getComponentAttribute("Wind (onshore)", "operationRateMax")


# # Step 3. Temporal Aggregation
#
# Although spatial aggregation aids in reducing the computational complexity of optimization, temporal aggregation is still necessary.
#
# Spatial aggregation is not here is replace temporal aggregation. They both go hand-in-hand.
#
# Imagine performing temporal aggregation on a model with too many regions. You have to reduce the temporal resolution to a large extent. Or you can take too few regions and reduce the temporal resolution to a smaller extent.
#
# With spatial and temporal aggregation, you need not compromise on either the temporal or spatial resolution of your model.

# In[12]:


aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=7)


# # Step 5. Optimization

# In[13]:


aggregated_esM.optimize(timeSeriesAggregation=True)
# The following `optimizationSpecs` are recommended if you use the Gurobi solver.
# aggregated_esM.optimize(timeSeriesAggregation=True,
#                         optimizationSpecs='OptimalityTol=1e-3 method=2 cuts=0')
