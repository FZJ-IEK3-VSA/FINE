# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os 
import sys

import pandas as pd
import xarray as xr

import FINE as fn
from FINE.spagat.RE_representation import represent_RE_technology

cwd = os.getcwd()

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Workflow for spatial aggregation of an energy system model
#
# This example notebook shows how model regions can be merged to obtain fewer regions and also how the number of VRES technologies within each region can be reduced to fewer technology types. 
#
# <img src="spagat_basic_depiction.png" style="width: 1000px;"/>
#
# The figure above dipicts the basic idea behind spatial aggregation. The term spatial grouping refers to grouping (and subsequently merging) of regions that are similar in some sense (NOTE: Please look into the documentation for different methods to group regions). 
#
# Additionally, it is also possible to reduced VRES technologies within each region to a desired number. This process is called spatial representation of VRES technologies. To give you an example, if the results of your PV simulation are spatially detailed or spatially highly resolved, then you could reduce these to a few types within each region. The time series profiles are matched during grouping of these technologies. 
#

# %% [markdown]
# ## STEP 1. Set up your ESM instance 

# %%
sys.path.append(os.path.join(cwd, '..', 'Multi-regional_Energy_System_Workflow'))
from getData import getData

data = getData()

# 1. Create an energy system model instance
locations = {'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'}
commodityUnitDict = {'electricity': r'GW$_{el}$', 'methane': r'GW$_{CH_{4},LHV}$', 'biogas': r'GW$_{biogas,LHV}$',
                     'CO2': r'Mio. t$_{CO_2}$/h', 'hydrogen': r'GW$_{H_{2},LHV}$'}
commodities = {'electricity', 'hydrogen', 'methane', 'biogas', 'CO2'}

esM = fn.EnergySystemModel(locations=locations, commodities=commodities, numberOfTimeSteps=8760,
                           commodityUnitsDict=commodityUnitDict,
                           hoursPerTimeStep=1, costUnit='1e9 Euro', lengthUnit='km', verboseLogLevel=0)

CO2_reductionTarget = 1


# 2. Add commodity sources to the energy system model

### Wind onshore
esM.add(fn.Source(esM=esM, name='Wind (onshore)', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['Wind (onshore), operationRateMax'],
                  capacityMax=data['Wind (onshore), capacityMax'],
                  investPerCapacity=1.1, opexPerCapacity=1.1*0.02, interestRate=0.08,
                  economicLifetime=20))

### PV
esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['PV, operationRateMax'], capacityMax=data['PV, capacityMax'],
                  investPerCapacity=0.65, opexPerCapacity=0.65*0.02, interestRate=0.08,
                  economicLifetime=25))


# 3. Add conversion components to the energy system model

### New combined cycly gas turbines for hydrogen
esM.add(fn.Conversion(esM=esM, name='New CCGT plants (hydrogen)', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.6},
                      hasCapacityVariable=True,
                      investPerCapacity=0.7, opexPerCapacity=0.021, interestRate=0.08,
                      economicLifetime=33))

### Electrolyzers
esM.add(fn.Conversion(esM=esM, name='Electroylzers', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':-1, 'hydrogen':0.7},
                      hasCapacityVariable=True,
                      investPerCapacity=0.5, opexPerCapacity=0.5*0.025, interestRate=0.08,
                      economicLifetime=10))


# 4. Add commodity storages to the energy system model

### Lithium ion batteries
esM.add(fn.Storage(esM=esM, name='Li-ion batteries', commodity='electricity',
                   hasCapacityVariable=True, chargeEfficiency=0.95,
                   cyclicLifetime=10000, dischargeEfficiency=0.95, selfDischarge=1-(1-0.03)**(1/(30*24)),
                   chargeRate=1, dischargeRate=1, doPreciseTsaModeling=False,
                   investPerCapacity=0.151, opexPerCapacity=0.002, interestRate=0.08,
                   economicLifetime=22))

### Hydrogen filled salt caverns
esM.add(fn.Storage(esM=esM, name='Salt caverns (hydrogen)', commodity='hydrogen',
                   hasCapacityVariable=True, capacityVariableDomain='continuous',
                   capacityPerPlantUnit=133,
                   chargeRate=1/470.37, dischargeRate=1/470.37, sharedPotentialID='Existing salt caverns',
                   stateOfChargeMin=0.33, stateOfChargeMax=1, capacityMax=data['Salt caverns (hydrogen), capacityMax'],
                   investPerCapacity=0.00011, opexPerCapacity=0.00057, interestRate=0.08,
                   economicLifetime=30))


# 5. Add commodity transmission components to the energy system model

### AC cables
esM.add(fn.LinearOptimalPowerFlow(esM=esM, name='AC cables', commodity='electricity',
                                  hasCapacityVariable=True, capacityFix=data['AC cables, capacityFix'],
                                  reactances=data['AC cables, reactances']))

### DC cables
esM.add(fn.Transmission(esM=esM, name='DC cables', commodity='electricity', losses=data['DC cables, losses'],
                        distances=data['DC cables, distances'],
                        hasCapacityVariable=True, capacityFix=data['DC cables, capacityFix']))


### Hydrogen pipelines
esM.add(fn.Transmission(esM=esM, name='Pipelines (hydrogen)', commodity='hydrogen',
                        distances=data['Pipelines, distances'],
                        hasCapacityVariable=True, hasIsBuiltBinaryVariable=False, bigM=300,
                        locationalEligibility=data['Pipelines, eligibility'],
                        capacityMax=data['Pipelines, eligibility']*15, sharedPotentialID='pipelines',
                        investPerCapacity=0.000177, investIfBuilt=0.00033,
                        interestRate=0.08, economicLifetime=40))

# 6. Add commodity sinks to the energy system model

### Electricity demand
esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                hasCapacityVariable=False, operationRateFix=data['Electricity demand, operationRateFix']))

## 7.2. Hydrogen sinks
FCEV_penetration=0.5
esM.add(fn.Sink(esM=esM, name='Hydrogen demand', commodity='hydrogen', hasCapacityVariable=False,
                operationRateFix=data['Hydrogen demand, operationRateFix']*FCEV_penetration))


# %% [markdown]
# ## STEP 2. Spatial grouping of regions

# %%
# The input data to spatial grouping are esM instance and the shapefile containing model regions' geometries
SHAPEFILE_PATH = os.path.join(cwd, '..', 'Multi-regional_Energy_System_Workflow/InputData/SpatialData/ShapeFiles/clusteredRegions.shp')

# %%
# Once the regions are grouped, the data witin each region group needs to be aggregated. Through the aggregation_function_dict
# parameter, it is possible to define how each variable show be aggregated. Please refer to the documentation for more 
# information. 

aggregation_function_dict = {'operationRateMax': ('mean', None),
                             'operationRateFix': ('sum', None),
                             'locationalEligibility': ('bool', None),
                             'capacityMax': ('sum', None),
                             'investPerCapacity': ('sum', None),
                             'investIfBuilt': ('sum', None),
                             'opexPerOperation': ('sum', None),
                             'opexPerCapacity': ('sum', None),
                             'opexIfBuilt': ('sum', None),
                             'interestRate': ('mean', None),
                             'economicLifetime': ('mean', None),
                             'capacityFix': ('sum', None),
                             'losses': ('mean', None),
                             'distances': ('mean', None),
                             'commodityCost': ('mean', None),
                             'commodityRevenue': ('mean', None),
                             'opexPerChargeOperation': ('mean', None),
                             'opexPerDischargeOperation': ('mean', None),
                             'QPcostScale': ('sum', None), 
                              'technicalLifetime': ('sum', None)}

# %%
# You can provide a path to save the grouping results with desired file names. Two files are saved - a shapefile containing
# the merged region geometries and a netcdf file containing the aggregated esM instance data. 
sds_region_filename='aggregated_regions'
sds_xr_dataset_filename='aggregated_xr_ds.nc4'

# %%
# NBVAL_IGNORE_OUTPUT

# Spatial grouping 
aggregated_esM = esM.aggregateSpatially(shapefile=SHAPEFILE_PATH, 
                                       grouping_mode='parameter_based', 
                                       nRegionsForRepresentation=6,
                                       aggregatedResultsPath=os.path.join(cwd, 'output_data'), 
                                       aggregation_function_dict=aggregation_function_dict,
                                       sds_region_filename=sds_region_filename,
                                       sds_xr_dataset_filename=sds_xr_dataset_filename)

# NOTE: The UserWarnings basically say that constant variables (variables that remain the same across all regions) and 
# geometry related varialbes are not considered for spatial grouping. 

# %%
# NBVAL_SKIP

# Original spatial resolution
fig, ax = fn.plotLocations(SHAPEFILE_PATH, plotLocNames=True, indexColumn='index')

# %%
# Spatial resolution after aggregation
AGGREGATED_SHP_PATH = os.path.join(cwd, 'output_data', f'{sds_region_filename}.shp')

# %%
# NBVAL_SKIP

fig, ax = fn.plotLocations(AGGREGATED_SHP_PATH, plotLocNames=True, indexColumn='space')

# %%
# NBVAL_IGNORE_OUTPUT

# The locations in the resulting esM instance are now 6.
new_locations = list(aggregated_esM.locations)
new_locations

# %%
#  And corresponding data has also been aggregated
aggregated_esM.getComponentAttribute('Wind (onshore)', 'operationRateMax')

# %% [markdown]
# # STEP 3. Spatial Representation of VRES (Optional)

# %% [markdown]
# ### STEP 3a. Spatial representation

# %%
# The input data to spatial representation are a netcdf file containing highly resolved VRES data 
# and the shapefile containing model regions' geometries

# Here, both PV and wind turbines are represented 

ONSHORE_WIND_DATA_PATH = os.path.join(cwd, 'input_RE_representation_data', 'DEU_wind.nc4')
PV_DATA_PATH = os.path.join(cwd, 'input_RE_representation_data', 'DEU_PV.nc4')

# %%
# NBVAL_IGNORE_OUTPUT

# Let us first take a look at one of these datasets 

xr.open_dataset(ONSHORE_WIND_DATA_PATH)

# %%
# NBVAL_IGNORE_OUTPUT

## Representation 
represented_wind_ds = represent_RE_technology(ONSHORE_WIND_DATA_PATH, 
                                            'xy_reference_system',
                                            AGGREGATED_SHP_PATH,
                                            n_timeSeries_perRegion=5,
                                            capacity_var_name='capacity',
                                            capfac_var_name='capfac',
                                            index_col = 'space')

represented_pv_ds = represent_RE_technology(PV_DATA_PATH, 
                                            'xy_reference_system',
                                            AGGREGATED_SHP_PATH,
                                            n_timeSeries_perRegion=5,
                                            capacity_var_name='capacity',
                                            capfac_var_name='capfac',
                                            index_col = 'space')

# %%
# NBVAL_IGNORE_OUTPUT

represented_wind_ds

# %%
# NBVAL_IGNORE_OUTPUT

represented_pv_ds

# %% [markdown]
# ### STEP 3a. Adding the results to esM instance 

# %%
# Now we need to delete 'Wind (onshore)' and 'PV' compoents from aggregated_esM 
# and add the represented results 

# %%
## But first we need certain info corresponding to these techs as they remain the same:
wind_investPerCapacity = aggregated_esM.getComponentAttribute('Wind (onshore)', 'investPerCapacity').mean()
wind_opexPerCapacity = aggregated_esM.getComponentAttribute('Wind (onshore)', 'opexPerCapacity').mean()
wind_interestRate = aggregated_esM.getComponentAttribute('Wind (onshore)', 'interestRate').mean()
wind_economicLifetime = aggregated_esM.getComponentAttribute('Wind (onshore)', 'economicLifetime').mean()

pv_investPerCapacity = aggregated_esM.getComponentAttribute('PV', 'investPerCapacity').mean()
pv_opexPerCapacity = aggregated_esM.getComponentAttribute('PV', 'opexPerCapacity').mean()
pv_interestRate = aggregated_esM.getComponentAttribute('PV', 'interestRate').mean()
pv_economicLifetime = aggregated_esM.getComponentAttribute('PV', 'economicLifetime').mean()

# %%
## And now we delete them
aggregated_esM.removeComponent('Wind (onshore)')
aggregated_esM.removeComponent('PV')

# %%
# NBVAL_SKIP

aggregated_esM.componentModelingDict['SourceSinkModel'].componentsDict

# %%
## Prepare the representation results and add them to aggregated_esM
data = {}   

time_steps = aggregated_esM.totalTimeSteps
regions = represented_wind_ds['region_ids'].values
clusters = represented_wind_ds['TS_ids'].values # technology types per region


for i, cluster in enumerate(clusters):
    # Add a wind turbine
    data.update({f'Wind (onshore), capacityMax {i}': pd.Series(represented_wind_ds.capacity.loc[:,cluster], index=regions)})

    data.update({f'Wind (onshore), operationRateMax {i}': pd.DataFrame(represented_wind_ds.capfac.loc[:,:,cluster].values,
                                                                       index=time_steps, columns=regions)})
    

    # Add a pv
    data.update({f'PV, capacityMax {i}': pd.Series(represented_pv_ds.capacity.loc[:,cluster], index=regions)})

    data.update({f'PV, operationRateMax {i}': pd.DataFrame(represented_pv_ds.capfac.loc[:,:,cluster].values,
                                                                       index=time_steps, columns=regions)})

# %%
## add the data 
for i, cluster in enumerate(clusters):
    aggregated_esM.add(fn.Source(esM=aggregated_esM, 
                      name=f'Wind (onshore) {i}',
                      commodity='electricity', 
                      hasCapacityVariable=True,
                      operationRateMax=data[f'Wind (onshore), operationRateMax {i}'],
                      capacityMax=data[f'Wind (onshore), capacityMax {i}'],
                      investPerCapacity=wind_investPerCapacity, 
                      opexPerCapacity=wind_opexPerCapacity,
                      interestRate=pv_interestRate,
                      economicLifetime=wind_economicLifetime
                      ))
    
    aggregated_esM.add(fn.Source(esM=aggregated_esM, 
                      name=f'PV {i}', 
                      commodity='electricity',
                      hasCapacityVariable=True,
                      operationRateMax=data[f'PV, operationRateMax {i}'], 
                      capacityMax=data[f'PV, capacityMax {i}'],
                      investPerCapacity=pv_investPerCapacity, 
                      opexPerCapacity=pv_opexPerCapacity, 
                      interestRate=pv_interestRate,
                      economicLifetime=pv_economicLifetime))

# %%
# NBVAL_SKIP

aggregated_esM.componentModelingDict['SourceSinkModel'].componentsDict

# %% [markdown]
# # Step 4. Temporal Aggregation
#
# Although spatial aggregation aids in reducing the computational complexity of optimization, temporal aggregation is still necessary. 
#
# Spatial aggregation is not here is replace temporal aggregation. They both go hand-in-hand. 
#
# Imagine performing temporal aggregation on a model with too many regions and too many VRES technologies per region. You have to reduce the temporal resolution to a large extent. Or you can take too few regions and 1 time series per VRES technology, per region and reduce the temporal resolution to a smaller extent. 
#
# With spatial and temporal aggregation, you need not compromise on either the temporal or spatial resolution of your model. 

# %%
# NBVAL_IGNORE_OUTPUT

aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=7)

# %% [markdown]
# # Step 5. Optimization

# %%
# NBVAL_IGNORE_OUTPUT

aggregated_esM.optimize(timeSeriesAggregation=True, 
                        optimizationSpecs='OptimalityTol=1e-3 method=2 cuts=0')
