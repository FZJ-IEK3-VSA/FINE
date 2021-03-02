# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Workflow for a multi-regional energy system
#
# In this application of the FINE framework, a multi-regional energy system is modeled and optimized.
#
# All classes which are available to the user are utilized and examples of the selection of different parameters within these classes are given.
#
# The workflow is structures as follows:
# 1. Required packages are imported and the input data path is set
# 2. An energy system model instance is created
# 3. Commodity sources are added to the energy system model
# 4. Commodity conversion components are added to the energy system model
# 5. Commodity storages are added to the energy system model
# 6. Commodity transmission components are added to the energy system model
# 7. Commodity sinks are added to the energy system model
# 8. The energy system model is optimized
# 9. Selected optimization results are presented
#

# %% [markdown]
# # 1. Import required packages and set input data path
#
# The FINE framework is imported which provides the required classes and functions for modeling the energy system.

# %%
import FINE as fn
import matplotlib.pyplot as plt
from getData import getData
import pandas as pd
import os

cwd = os.getcwd()
data = getData()

# %matplotlib inline  
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # 2. Create an energy system model instance 
#
# The structure of the energy system model is given by the considered locations, commodities, the number of time steps as well as the hours per time step.
#
# The commodities are specified by a unit (i.e. 'GW_electric', 'GW_H2lowerHeatingValue', 'Mio. t CO2/h') which can be given as an energy or mass unit per hour. Furthermore, the cost unit and length unit are specified.

# %%
locations = {'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'}
commodityUnitDict = {'electricity': r'GW$_{el}$', 'methane': r'GW$_{CH_{4},LHV}$', 'biogas': r'GW$_{biogas,LHV}$',
                     'CO2': r'Mio. t$_{CO_2}$/h', 'hydrogen': r'GW$_{H_{2},LHV}$'}
commodities = {'electricity', 'hydrogen', 'methane', 'biogas', 'CO2'}
numberOfTimeSteps=8760
hoursPerTimeStep=1

# %%
esM = fn.EnergySystemModel(locations=locations, commodities=commodities, numberOfTimeSteps=8760,
                           commodityUnitsDict=commodityUnitDict,
                           hoursPerTimeStep=1, costUnit='1e9 Euro', lengthUnit='km', verboseLogLevel=0)

# %%
CO2_reductionTarget = 1

# %% [markdown]
# # 3. Add commodity sources to the energy system model

# %% [markdown]
# ## 3.1. Electricity sources

# %% [markdown]
# ### Wind onshore

# %%
esM.add(fn.Source(esM=esM, name='Wind (onshore)', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['Wind (onshore), operationRateMax'],
                  capacityMax=data['Wind (onshore), capacityMax'],
                  investPerCapacity=1.1, opexPerCapacity=1.1*0.02, interestRate=0.08,
                  economicLifetime=20))

# %% [markdown]
# Full load hours:

# %%
data['Wind (onshore), operationRateMax'].sum()

# %% [markdown]
# ### Wind offshore

# %%
esM.add(fn.Source(esM=esM, name='Wind (offshore)', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['Wind (offshore), operationRateMax'],
                  capacityMax=data['Wind (offshore), capacityMax'],
                  investPerCapacity=2.3, opexPerCapacity=2.3*0.02, interestRate=0.08,
                  economicLifetime=20))

# %% [markdown]
# Full load hours:

# %%
data['Wind (offshore), operationRateMax'].sum()

# %% [markdown]
# ### PV

# %%
esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=data['PV, operationRateMax'], capacityMax=data['PV, capacityMax'],
                  investPerCapacity=0.65, opexPerCapacity=0.65*0.02, interestRate=0.08,
                  economicLifetime=25))

# %% [markdown]
# Full load hours:

# %%
data['PV, operationRateMax'].sum()

# %% [markdown]
# ### Exisisting run-of-river hydroelectricity plants

# %%
esM.add(fn.Source(esM=esM, name='Existing run-of-river plants', commodity='electricity',
                  hasCapacityVariable=True,
                  operationRateFix=data['Existing run-of-river plants, operationRateFix'], tsaWeight=0.01,
                  capacityFix=data['Existing run-of-river plants, capacityFix'],
                  investPerCapacity=0, opexPerCapacity=0.208))

# %% [markdown]
# ## 3.2. Methane (natural gas and biogas)

# %% [markdown]
# ### Natural gas

# %%
esM.add(fn.Source(esM=esM, name='Natural gas purchase', commodity='methane',
                  hasCapacityVariable=False, commodityCost=0.0331*1e-3))

# %% [markdown]
# ### Biogas

# %%
esM.add(fn.Source(esM=esM, name='Biogas purchase', commodity='biogas',
                  operationRateMax=data['Biogas, operationRateMax'], hasCapacityVariable=False,
                  commodityCost=0.05409*1e-3))

# %% [markdown]
# # 4. Add conversion components to the energy system model

# %% [markdown]
# ### Combined cycle gas turbine plants

# %%
esM.add(fn.Conversion(esM=esM, name='CCGT plants (methane)', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'methane':-1/0.6, 'CO2':201*1e-6/0.6},
                      hasCapacityVariable=True,
                      investPerCapacity=0.65, opexPerCapacity=0.021, interestRate=0.08,
                      economicLifetime=33))

# %% [markdown]
# ### New combined cycle gas turbine plants for biogas

# %%
esM.add(fn.Conversion(esM=esM, name='New CCGT plants (biogas)', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'biogas':-1/0.63},
                      hasCapacityVariable=True, 
                      investPerCapacity=0.7, opexPerCapacity=0.021, interestRate=0.08,
                      economicLifetime=33))

# %% [markdown]
# ### New combined cycly gas turbines for hydrogen

# %%
esM.add(fn.Conversion(esM=esM, name='New CCGT plants (hydrogen)', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.63},
                      hasCapacityVariable=True, 
                      investPerCapacity=0.7, opexPerCapacity=0.021, interestRate=0.08,
                      economicLifetime=33))

# %% [markdown]
# ### Electrolyzers

# %%
esM.add(fn.Conversion(esM=esM, name='Electrolyzer', physicalUnit=r'GW$_{el}$',
                      commodityConversionFactors={'electricity':-1, 'hydrogen':0.7},
                      hasCapacityVariable=True, 
                      investPerCapacity=0.5, opexPerCapacity=0.5*0.025, interestRate=0.08,
                      economicLifetime=10))

# %% [markdown]
# ### rSOC

# %%
capexRSOC=1.5

esM.add(fn.Conversion(esM=esM, name='rSOEC', physicalUnit=r'GW$_{el}$', linkedConversionCapacityID='rSOC',
                      commodityConversionFactors={'electricity':-1, 'hydrogen':0.6},
                      hasCapacityVariable=True, 
                      investPerCapacity=capexRSOC/2, opexPerCapacity=capexRSOC*0.02/2, interestRate=0.08,
                      economicLifetime=10))

esM.add(fn.Conversion(esM=esM, name='rSOFC', physicalUnit=r'GW$_{el}$', linkedConversionCapacityID='rSOC',
                      commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.6},
                      hasCapacityVariable=True, 
                      investPerCapacity=capexRSOC/2, opexPerCapacity=capexRSOC*0.02/2, interestRate=0.08,
                      economicLifetime=10))

# %% [markdown]
# # 5. Add commodity storages to the energy system model

# %% [markdown]
# ## 5.1. Electricity storage

# %% [markdown]
# ### Lithium ion batteries
#
# The self discharge of a lithium ion battery is here described as 3% per month. The self discharge per hours is obtained using the equation (1-$\text{selfDischarge}_\text{hour})^{30*24\text{h}} = 1-\text{selfDischarge}_\text{month}$.

# %%
esM.add(fn.Storage(esM=esM, name='Li-ion batteries', commodity='electricity',
                   hasCapacityVariable=True, chargeEfficiency=0.95,
                   cyclicLifetime=10000, dischargeEfficiency=0.95, selfDischarge=1-(1-0.03)**(1/(30*24)),
                   chargeRate=1, dischargeRate=1, doPreciseTsaModeling=False,
                   investPerCapacity=0.151, opexPerCapacity=0.002, interestRate=0.08,
                   economicLifetime=22))

# %% [markdown]
# ## 5.2. Hydrogen storage

# %% [markdown]
# ### Hydrogen filled salt caverns
# The maximum capacity is here obtained by: dividing the given capacity (which is given for methane) by the lower heating value of methane and then multiplying it with the lower heating value of hydrogen.

# %%
esM.add(fn.Storage(esM=esM, name='Salt caverns (hydrogen)', commodity='hydrogen',
                   hasCapacityVariable=True, capacityVariableDomain='continuous',
                   capacityPerPlantUnit=133,
                   chargeRate=1/470.37, dischargeRate=1/470.37, sharedPotentialID='Existing salt caverns',
                   stateOfChargeMin=0.33, stateOfChargeMax=1, capacityMax=data['Salt caverns (hydrogen), capacityMax'],
                   investPerCapacity=0.00011, opexPerCapacity=0.00057, interestRate=0.08,
                   economicLifetime=30))

# %% [markdown]
# ## 5.3. Methane storage

# %% [markdown]
# ### Methane filled salt caverns

# %%
esM.add(fn.Storage(esM=esM, name='Salt caverns (biogas)', commodity='biogas',
                   hasCapacityVariable=True, capacityVariableDomain='continuous',
                   capacityPerPlantUnit=443,
                   chargeRate=1/470.37, dischargeRate=1/470.37, sharedPotentialID='Existing salt caverns',
                   stateOfChargeMin=0.33, stateOfChargeMax=1, capacityMax=data['Salt caverns (methane), capacityMax'],
                   investPerCapacity=0.00004, opexPerCapacity=0.00001, interestRate=0.08,
                   economicLifetime=30))

# %% [markdown]
# ## 5.4 Pumped hydro storage

# %% [markdown]
# ### Pumped hydro storage

# %%
esM.add(fn.Storage(esM=esM, name='Pumped hydro storage', commodity='electricity',
                   chargeEfficiency=0.88, dischargeEfficiency=0.88,
                   hasCapacityVariable=True, selfDischarge=1-(1-0.00375)**(1/(30*24)),
                   chargeRate=0.16, dischargeRate=0.12, capacityFix=data['Pumped hydro storage, capacityFix'],
                   investPerCapacity=0, opexPerCapacity=0.000153))

# %% [markdown]
# # 6. Add commodity transmission components to the energy system model

# %% [markdown]
# ## 6.1. Electricity transmission

# %% [markdown]
# ### AC cables

# %% [markdown]
# esM.add(fn.LinearOptimalPowerFlow(esM=esM, name='AC cables', commodity='electricity',
#                                   hasCapacityVariable=True, capacityFix=data['AC cables, capacityFix'],
#                                   reactances=data['AC cables, reactances']))

# %%
esM.add(fn.Transmission(esM=esM, name='AC cables', commodity='electricity',
                                  hasCapacityVariable=True, capacityFix=data['AC cables, capacityFix']))

# %% [markdown]
# ### DC cables

# %%
esM.add(fn.Transmission(esM=esM, name='DC cables', commodity='electricity', losses=data['DC cables, losses'],
                        distances=data['DC cables, distances'],
                        hasCapacityVariable=True, capacityFix=data['DC cables, capacityFix']))

# %% [markdown]
# ## 6.2 Methane transmission

# %% [markdown]
# ### Methane pipeline

# %%
esM.add(fn.Transmission(esM=esM, name='Pipelines (biogas)', commodity='biogas', 
                        distances=data['Pipelines, distances'],
                        hasCapacityVariable=True, hasIsBuiltBinaryVariable=True, bigM=300,
                        locationalEligibility=data['Pipelines, eligibility'],
                        capacityMax=data['Pipelines, eligibility']*15, sharedPotentialID='pipelines',
                        investPerCapacity=0.000037, investIfBuilt=0.000314,
                        interestRate=0.08, economicLifetime=40))

# %% [markdown]
# ## 6.3 Hydrogen transmission

# %% [markdown]
# ### Hydrogen pipelines

# %%
esM.add(fn.Transmission(esM=esM, name='Pipelines (hydrogen)', commodity='hydrogen',
                        distances=data['Pipelines, distances'],
                        hasCapacityVariable=True, hasIsBuiltBinaryVariable=True, bigM=300,
                        locationalEligibility=data['Pipelines, eligibility'],
                        capacityMax=data['Pipelines, eligibility']*15, sharedPotentialID='pipelines',
                        investPerCapacity=0.000177, investIfBuilt=0.00033,
                        interestRate=0.08, economicLifetime=40))

# %% [markdown]
# # 7. Add commodity sinks to the energy system model

# %% [markdown]
# ## 7.1. Electricity sinks

# %% [markdown]
# ### Electricity demand

# %%
esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                hasCapacityVariable=False, operationRateFix=data['Electricity demand, operationRateFix']))

# %% [markdown]
# ## 7.2. Hydrogen sinks

# %% [markdown]
# ### Fuel cell electric vehicle (FCEV) demand

# %%
FCEV_penetration=0.5
esM.add(fn.Sink(esM=esM, name='Hydrogen demand', commodity='hydrogen', hasCapacityVariable=False,
                operationRateFix=data['Hydrogen demand, operationRateFix']*FCEV_penetration))

# %% [markdown]
# ## 7.3. CO2 sinks

# %% [markdown]
# ### CO2 exiting the system's boundary

# %%
esM.add(fn.Sink(esM=esM, name='CO2 to enviroment', commodity='CO2',
                hasCapacityVariable=False, commodityLimitID='CO2 limit', yearlyLimit=366*(1-CO2_reductionTarget)))

# %% [markdown]
# # 8. Optimize energy system model

# %% [markdown]
# All components are now added to the model and the model can be optimized. If the computational complexity of the optimization should be reduced, the time series data of the specified components can be clustered before the optimization and the parameter timeSeriesAggregation is set to True in the optimize call.

# %% tags=["nbval-skip"]
esM.cluster(numberOfTypicalPeriods=7)

# %% tags=["nbval-skip"]
esM.optimize(timeSeriesAggregation=True, optimizationSpecs='OptimalityTol=1e-3 method=2 cuts=0')

# %% [markdown]
# # 9. Selected results output
#
# Plot locations (GeoPandas required)

# %% tags=["nbval-skip"]
# Import the geopandas package for plotting the locations
import geopandas as gpd

# %% tags=["nbval-skip"]
locFilePath = os.path.join(cwd, 'InputData', 'SpatialData','ShapeFiles', 'clusteredRegions.shp')
fig, ax = fn.plotLocations(locFilePath, plotLocNames=True, indexColumn='index')

# %% [markdown]
# ### Sources and Sink
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

# %% [markdown]
# Plot installed capacities

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(esM, 'Wind (offshore)', locFilePath, 'index', perArea=False)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(esM, 'Wind (onshore)', locFilePath, 'index', perArea=False)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(esM, 'PV', locFilePath, 'index', perArea=False)

# %% [markdown]
# Plot operation time series (either one or two dimensional)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperation(esM, 'Electricity demand', 'cluster_0')

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, 'Electricity demand', 'cluster_0')

# %% [markdown]
# ### Conversion
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("ConversionModel", outputLevel=2)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(esM, 'Electrolyzer', locFilePath, 'index', perArea=False)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, 'New CCGT plants (biogas)', 'cluster_2')

# %% [markdown]
# ### Storage
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("StorageModel", outputLevel=2)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, 'Li-ion batteries', 'cluster_2', 
                                   variableName='stateOfChargeOperationVariablesOptimum')

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, 'Pumped hydro storage', 'cluster_2',
                                  variableName='stateOfChargeOperationVariablesOptimum')

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, 'Salt caverns (biogas)', 'cluster_2',
                                  variableName='stateOfChargeOperationVariablesOptimum')

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, 'Salt caverns (hydrogen)', 'cluster_2',
                                  variableName='stateOfChargeOperationVariablesOptimum')

# %% [markdown]
# ## Transmission
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("TransmissionModel", outputLevel=2).loc['Pipelines (hydrogen)']

# %% [markdown]
# Check that the shared capacity of the pipelines are not exceeded

# %% tags=["nbval-skip"]
df=esM.componentModelingDict["TransmissionModel"].capacityVariablesOptimum
df.loc['Pipelines (biogas)']+df.loc['Pipelines (hydrogen)']

# %% [markdown]
# Plot installed transmission capacities

# %% tags=["nbval-skip"]
transFilePath = os.path.join(cwd, 'InputData', 'SpatialData','ShapeFiles', 'AClines.shp')

fig, ax = fn.plotLocations(locFilePath, indexColumn='index')                                 
fig, ax = fn.plotTransmission(esM, 'AC cables', transFilePath, loc0='bus0', loc1='bus1', fig=fig, ax=ax)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(cwd, 'InputData', 'SpatialData','ShapeFiles', 'DClines.shp')

fig, ax = fn.plotLocations(locFilePath, indexColumn='index')                                 
fig, ax = fn.plotTransmission(esM, 'DC cables', transFilePath, loc0='cluster0', loc1='cluster1', fig=fig, ax=ax)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(cwd, 'InputData', 'SpatialData','ShapeFiles', 'transmissionPipeline.shp')

fig, ax = fn.plotLocations(locFilePath, indexColumn='index')                                 
fig, ax = fn.plotTransmission(esM, 'Pipelines (hydrogen)', transFilePath, loc0='loc1', loc1='loc2',
                              fig=fig, ax=ax)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(cwd, 'InputData', 'SpatialData','ShapeFiles', 'transmissionPipeline.shp')

fig, ax = fn.plotLocations(locFilePath, indexColumn='index')                                 
fig, ax = fn.plotTransmission(esM, 'Pipelines (biogas)', transFilePath, loc0='loc1', loc1='loc2',
                              fig=fig, ax=ax)

# %% [markdown]
# # Postprocessing: Determine robust pipeline design

# %% tags=["nbval-skip"]
# Import module expansion "robustPipelineSizing"
from FINE.expansionModules import robustPipelineSizing

# %% tags=["nbval-skip"]
# 1. Option to get the injection and withdrawal rates for the pipeline sizing (in kg/s)
rates = robustPipelineSizing.getInjectionWithdrawalRates(componentName='Pipelines (hydrogen)',esM=esM) # in GWh
# Convert GWh to kg/s: GWh * (kWh/GWh) * (kg/kWh) * (1/ timestepLengthInSeconds) with timestepLengthInSeconds
# being 3600 seconds for the present example
rates = rates * (10 ** 6) * (1/33.32) * (1/3600) 
rates.head()

# %% tags=["nbval-skip"]
# 2. Option to get the injection and withdrawal rates for the pipeline sizing (in kg/s)
op = esM.componentModelingDict[esM.componentNames['Pipelines (hydrogen)']].\
    getOptimalValues('operationVariablesOptimum')['values'].loc['Pipelines (hydrogen)']
rates = robustPipelineSizing.getInjectionWithdrawalRates(operationVariablesOptimumData=op) # in GWh
# Convert GWh to kg/s: GWh * (kWh/GWh) * (kg/kWh) * (1/ timestepLengthInSeconds) with timestepLengthInSeconds
# being 3600 seconds for the present example
rates = rates * (10 ** 6) * (1/33.32) * (1/3600) 
rates.head()

# %% tags=["nbval-skip"]
# Determine unique withdrawal and injection scenarios to save computation time
rates = rates.drop_duplicates()
rates.head()

# %% tags=["nbval-skip"]
# Get the lengths of the pipeline (in m)
lengths = robustPipelineSizing.getNetworkLengthsFromESM('Pipelines (hydrogen)', esM)
lengths = lengths * 1e3
lengths.head()

# %% tags=["nbval-skip"]
# Specify minimum and maximum pressure levels for all injection and withdrawal nodes (in bar)
p_min_nodes = {'cluster_5': 60, 'cluster_3': 50, 'cluster_7': 50, 'cluster_1': 60, 'cluster_6': 50,
               'cluster_4': 50, 'cluster_0': 50, 'cluster_2': 50}

p_max_nodes = {'cluster_5': 100, 'cluster_3': 100, 'cluster_7': 100, 'cluster_1': 90, 'cluster_6': 100,
               'cluster_4': 100, 'cluster_0': 100, 'cluster_2': 100}

# %% tags=["nbval-skip"]
# Specify the investment cost of the available diameter classes in €/m

# For single pipes
dic_diameter_costs = {0.1063: 37.51, 0.1307: 38.45, 0.1593: 39.64, 0.2065: 42.12,
                      0.2588: 45.26, 0.3063: 48.69, 0.3356: 51.07, 0.3844: 55.24,
                      0.432: 59.86, 0.4796: 64.98, 0.527: 70.56, 0.578: 76.61,
                      0.625: 82.99, 0.671: 89.95, 0.722: 97.38, 0.7686: 105.28,
                      0.814: 113.63, 0.864: 122.28, 0.915: 131.56, 0.96: 141.3,
                      1.011: 151.5, 1.058: 162.17, 1.104: 173.08, 1.155: 184.67,
                      1.249: 209.24, 1.342: 235.4, 1.444: 263.66, 1.536: 293.78}

# For parallel pipes
dic_candidateMergedDiam_costs={1.342: 235.4, 1.444: 263.66, 1.536: 293.78}

# %% tags=["nbval-skip"]
# Choose if a robust pipeline desin should be determined or the design should be optimized based on the
# given injection and withdrawal rates only
robust = True

# %% [markdown]
# ### Determine design for simple network structure

# %% tags=["nbval-skip"]
# Compute with 7 threads and simple network structure

dic_arc_optimalDiameters, dic_scen_PressLevels, dic_scen_MaxViolPress, dic_timeStep_PressLevels, \
    dic_timeStep_MaxViolPress, _ = robustPipelineSizing.determineDiscretePipelineDesign(
        robust=robust, injectionWithdrawalRates=rates,
        distances=lengths, dic_node_minPress=p_min_nodes, dic_node_maxPress=p_max_nodes, 
        dic_diameter_costs=dic_diameter_costs, dic_candidateMergedDiam_costs=dic_candidateMergedDiam_costs,
        threads=7)

# %% tags=["nbval-skip"]
# Compute with 2 threads and simple network structure

dic_arc_optimalDiameters, dic_scen_PressLevels, dic_scen_MaxViolPress, dic_timeStep_PressLevels, \
    dic_timeStep_MaxViolPress, _ = robustPipelineSizing.determineDiscretePipelineDesign(
        robust=robust, injectionWithdrawalRates=rates,
        distances=lengths, dic_node_minPress=p_min_nodes, dic_node_maxPress=p_max_nodes, 
        dic_diameter_costs=dic_diameter_costs, dic_candidateMergedDiam_costs=dic_candidateMergedDiam_costs,
        threads=2)

print("Finished Discrete Pipeline Optimization")

# %% [markdown]
# ### Determine design for more complex network structure

# %% tags=["nbval-skip"]
# Redfine minimum and maximum pressure levels to reduce robust scenario computation time
p_min_nodes = {'cluster_5': 50, 'cluster_3': 50, 'cluster_7': 50, 'cluster_1': 50, 'cluster_6': 50,
               'cluster_4': 50, 'cluster_0': 50, 'cluster_2': 50}

p_max_nodes = {'cluster_5': 100, 'cluster_3': 100, 'cluster_7': 100, 'cluster_1': 100, 'cluster_6': 100,
               'cluster_4': 100, 'cluster_0': 100, 'cluster_2': 100}

# %% tags=["nbval-skip"]
# Get more complex network structure
regColumn1 = 'loc1'
regColumn2 = 'loc2'

dic_node_minPress=p_min_nodes
dic_node_maxPress=p_max_nodes

maxPipeLength= 35 * 1e3
minPipeLength= 1 * 1e3

distances_new, dic_node_minPress_new, dic_node_maxPress_new, gdfNodes, gdfEdges = \
    robustPipelineSizing.getRefinedShapeFile(transFilePath, regColumn1, regColumn2, dic_node_minPress,
                                             dic_node_maxPress, minPipeLength, maxPipeLength)

fig, ax = fn.plotLocations(locFilePath, indexColumn='index') 

gdfEdges.plot(ax=ax, color='grey')
gdfNodes.plot(ax=ax, color='red', markersize=2)
ax.axis('off')

plt.show()

# %% tags=["nbval-skip"]
# Determine optimal pipeline design with seven threads
dic_arc_optimalDiameters, dic_scen_PressLevels, dic_scen_MaxViolPress, dic_timeStep_PressLevels, \
           dic_timeStep_MaxViolPress, gdfEdges = robustPipelineSizing.determineDiscretePipelineDesign(
        robust=robust, injectionWithdrawalRates=rates,
        distances=distances_new, dic_node_minPress=dic_node_minPress_new, dic_node_maxPress=dic_node_maxPress_new,
        dic_diameter_costs=dic_diameter_costs, dic_candidateMergedDiam_costs=dic_candidateMergedDiam_costs,
        gdfEdges=gdfEdges, regColumn1='nodeIn', regColumn2='nodeOut', threads=7, solver='gurobi')

# %% [markdown]
# ### Plot scenario output

# %% tags=["nbval-skip"]
# Get regions shapefile as geopandas GeoDataFrame
gdf_regions = gpd.read_file(locFilePath)

# %% tags=["nbval-skip"]
# Visualize pipeline diameters
fig, ax =robustPipelineSizing.plotOptimizedNetwork(gdfEdges, gdf_regions=gdf_regions, figsize=(5,5),
    line_scaling=0.8)

# %% tags=["nbval-skip"]
# Get a minimum and maximum pressure scenario
dic_timeStep_PressLevels = pd.DataFrame.from_dict(dic_scen_PressLevels)

scen_min = dic_timeStep_PressLevels.mean().min()
scen_min = dic_timeStep_PressLevels.loc[:,dic_timeStep_PressLevels.mean() == scen_min].iloc[:,0]
scen_max = dic_timeStep_PressLevels.mean().max()
scen_max = dic_timeStep_PressLevels.loc[:,dic_timeStep_PressLevels.mean() == scen_max].iloc[:,0]

# %% tags=["nbval-skip"]
fig, ax =robustPipelineSizing.plotOptimizedNetwork(gdfEdges, gdf_regions=gdf_regions, figsize=(5,5),
    line_scaling=0.9, pressureLevels=scen_min)

# %% tags=["nbval-skip"]
fig, ax =robustPipelineSizing.plotOptimizedNetwork(gdfEdges, gdf_regions=gdf_regions, figsize=(5,5),
    line_scaling=0.9, pressureLevels=scen_max)

# %%
