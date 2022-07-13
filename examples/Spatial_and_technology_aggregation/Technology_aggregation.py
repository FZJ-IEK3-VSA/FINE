#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import pandas as pd
import xarray as xr

import FINE as fn
from FINE.aggregations.technologyAggregation.techAggregation import (
    aggregate_RE_technology,
)
import FINE.IOManagement.xarrayIO as xrIO

cwd = os.getcwd()

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# # Workflow for technology aggregation
#
# This example notebook shows how Variable Renewable Energy Sources (VRES) can be aggregated to obtain fewer types.
#
# <img src="tech_aggregation_for_example_notebook.png" style="width: 1000px;"/>
#
# The figure above dipicts the basic idea behind technology aggregation. Here, the number of VRES within each region is reduced to a desired number. To give you an example, if the results of your PV simulation are spatially detailed or spatially highly resolved, then you could reduce these to a few types within each region. The time series profiles are matched during grouping of these technologies.
#

# # STEP 1. Technology Aggregation

# In[2]:


# The input data for technology aggregation could either be provided in gridded or non-gridded form.
## The example here shows the data in the gridded form. In this case, a corresponding shapefile should also be provided to
## match the data in the grids to appropriate regions. Please refer to the documentation to know more.

# shapefile containing model regions' geometries
SHP_PATH = os.path.join(cwd, "output_data", "aggregated_regions.shp")

# netcdf files containing highly resolved VRES data. In this example, both PV and wind turbines are aggregated

ONSHORE_WIND_DATA_PATH = os.path.join(
    cwd, "input_tech_aggregation_data", "DEU_wind.nc4"
)
PV_DATA_PATH = os.path.join(cwd, "input_tech_aggregation_data", "DEU_pv.nc4")


# In[3]:


# Let us first take a look at one of these datasets

xr.open_dataset(ONSHORE_WIND_DATA_PATH)


# In[4]:


## Aggregation
aggregated_wind_ds = aggregate_RE_technology(
    ONSHORE_WIND_DATA_PATH,
    "xy_reference_system",
    SHP_PATH,
    n_timeSeries_perRegion=5,
    capacity_var_name="capacity",
    capfac_var_name="capfac",
    shp_index_col="space",
)

aggregated_pv_ds = aggregate_RE_technology(
    PV_DATA_PATH,
    "xy_reference_system",
    SHP_PATH,
    n_timeSeries_perRegion=5,
    capacity_var_name="capacity",
    capfac_var_name="capfac",
    shp_index_col="space",
)


# In[5]:


aggregated_wind_ds


# In[6]:


aggregated_pv_ds


# # STEP 2. Adding the results to ESM instance

# In[7]:


# If you have an ESM instance, then you could add the results of technology aggregation to this instance


# In[8]:


# First, set up the ESM instance
esm = xrIO.readNetCDFtoEnergySystemModel(
    os.path.join(cwd, "output_data", "aggregated_xr_ds.nc")
)


# In[9]:


# If wind turbine and PV are already present in the ESM instance, we need to replace them like shown in the cells below.
esm.componentModelingDict["SourceSinkModel"].componentsDict


# In[10]:


## First, get certain info corresponding to these techs as they remain the same:
var_list = ["investPerCapacity", "opexPerCapacity", "interestRate", "economicLifetime"]

wind_vars = {}
pv_vars = {}

for var in var_list:
    wind_vars.update({var: esm.getComponentAttribute("Wind (onshore)", var).mean()})
    pv_vars.update({var: esm.getComponentAttribute("PV", var).mean()})

## And now we delete them
esm.removeComponent("Wind (onshore)")
esm.removeComponent("PV")


# In[11]:


esm.componentModelingDict["SourceSinkModel"].componentsDict


# In[12]:


## Prepare the aggregation results and add them to the esm
data = {}

time_steps = esm.totalTimeSteps
regions = aggregated_wind_ds["region_ids"].values
clusters = aggregated_wind_ds["TS_ids"].values  # technology types per region


for i, cluster in enumerate(clusters):
    # Add a wind turbine
    data.update(
        {
            f"Wind (onshore), capacityMax {i}": pd.Series(
                aggregated_wind_ds.capacity.loc[:, cluster], index=regions
            )
        }
    )

    data.update(
        {
            f"Wind (onshore), operationRateMax {i}": pd.DataFrame(
                aggregated_wind_ds.capfac.loc[:, :, cluster].values,
                index=time_steps,
                columns=regions,
            )
        }
    )

    # Add a pv
    data.update(
        {
            f"PV, capacityMax {i}": pd.Series(
                aggregated_pv_ds.capacity.loc[:, cluster], index=regions
            )
        }
    )

    data.update(
        {
            f"PV, operationRateMax {i}": pd.DataFrame(
                aggregated_pv_ds.capfac.loc[:, :, cluster].values,
                index=time_steps,
                columns=regions,
            )
        }
    )


# In[13]:


## add the data
for i, cluster in enumerate(clusters):
    esm.add(
        fn.Source(
            esM=esm,
            name=f"Wind (onshore) {i}",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data[f"Wind (onshore), operationRateMax {i}"],
            capacityMax=data[f"Wind (onshore), capacityMax {i}"],
            investPerCapacity=wind_vars.get("investPerCapacity"),
            opexPerCapacity=wind_vars.get("opexPerCapacity"),
            interestRate=wind_vars.get("interestRate"),
            economicLifetime=wind_vars.get("economicLifetime"),
        )
    )

    esm.add(
        fn.Source(
            esM=esm,
            name=f"PV {i}",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data[f"PV, operationRateMax {i}"],
            capacityMax=data[f"PV, capacityMax {i}"],
            investPerCapacity=pv_vars.get("investPerCapacity"),
            opexPerCapacity=pv_vars.get("opexPerCapacity"),
            interestRate=pv_vars.get("interestRate"),
            economicLifetime=pv_vars.get("economicLifetime"),
        )
    )


# In[14]:


esm.componentModelingDict["SourceSinkModel"].componentsDict


# # Step 4. Temporal Aggregation

# In[15]:


esm.aggregateTemporally(numberOfTypicalPeriods=7)


# # Step 5. Optimization

# In[16]:


esm.optimize(timeSeriesAggregation=True)
# The following `optimizationSpecs` are recommended if you use the Gurobi solver.
# aggregated_esM.optimize(timeSeriesAggregation=True,
#                         optimizationSpecs='OptimalityTol=1e-3 method=2 cuts=0')
