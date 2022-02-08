# -*- coding: utf-8 -*-
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
locations = {
    "cluster_0",
    "cluster_1",
    "cluster_2",
    "cluster_3",
    "cluster_4",
    "cluster_5",
    "cluster_6",
    "cluster_7",
}
commodityUnitDict = {
    "electricity": r"GW$_{el}$",
    "methane": r"GW$_{CH_{4},LHV}$",
    "biogas": r"GW$_{biogas,LHV}$",
    "CO2": r"Mio. t$_{CO_2}$/h",
    "hydrogen": r"GW$_{H_{2},LHV}$",
}
commodities = {"electricity", "hydrogen", "methane", "biogas", "CO2"}
numberOfTimeSteps = 8760
hoursPerTimeStep = 1

# %%
esM = fn.EnergySystemModel(
    locations=locations,
    commodities=commodities,
    numberOfTimeSteps=8760,
    commodityUnitsDict=commodityUnitDict,
    hoursPerTimeStep=1,
    costUnit="1e9 Euro",
    lengthUnit="km",
    verboseLogLevel=0,
)

# %%
CO2_reductionTarget = 1

# %% [markdown]
# # 3. Add commodity sources to the energy system model

# %% [markdown]
# ## 3.1. Electricity sources

# %% [markdown]
# ### Wind onshore

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="Wind (onshore)",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateMax=data["Wind (onshore), operationRateMax"],
        capacityMax=data["Wind (onshore), capacityMax"],
        investPerCapacity=1.1,
        opexPerCapacity=1.1 * 0.02,
        interestRate=0.08,
        economicLifetime=20,
    )
)

# %% [markdown]
# Full load hours:

# %%
data["Wind (onshore), operationRateMax"].sum()

# %% [markdown]
# ### Wind offshore

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="Wind (offshore)",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateMax=data["Wind (offshore), operationRateMax"],
        capacityMax=data["Wind (offshore), capacityMax"],
        investPerCapacity=2.3,
        opexPerCapacity=2.3 * 0.02,
        interestRate=0.08,
        economicLifetime=20,
    )
)

# %% [markdown]
# Full load hours:

# %%
data["Wind (offshore), operationRateMax"].sum()

# %% [markdown]
# ### PV

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="PV",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateMax=data["PV, operationRateMax"],
        capacityMax=data["PV, capacityMax"],
        investPerCapacity=0.65,
        opexPerCapacity=0.65 * 0.02,
        interestRate=0.08,
        economicLifetime=25,
    )
)

# %% [markdown]
# Full load hours:

# %%
data["PV, operationRateMax"].sum()

# %% [markdown]
# ### Exisisting run-of-river hydroelectricity plants

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="Existing run-of-river plants",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateFix=data["Existing run-of-river plants, operationRateFix"],
        tsaWeight=0.01,
        capacityFix=data["Existing run-of-river plants, capacityFix"],
        investPerCapacity=0,
        opexPerCapacity=0.208,
    )
)

# %% [markdown]
# ## 3.2. Methane (natural gas and biogas)

# %% [markdown]
# ### Natural gas

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="Natural gas purchase",
        commodity="methane",
        hasCapacityVariable=False,
        commodityCost=0.0331 * 1e-3,
    )
)

# %% [markdown]
# ### Biogas

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="Biogas purchase",
        commodity="biogas",
        operationRateMax=data["Biogas, operationRateMax"],
        hasCapacityVariable=False,
        commodityCost=0.05409 * 1e-3,
    )
)

# %% [markdown]
# # 4. Add conversion components to the energy system model

# %% [markdown]
# ### Combined cycle gas turbine plants

# %%
esM.add(
    fn.Conversion(
        esM=esM,
        name="CCGT plants (methane)",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={
            "electricity": 1,
            "methane": -1 / 0.6,
            "CO2": 201 * 1e-6 / 0.6,
        },
        hasCapacityVariable=True,
        investPerCapacity=0.65,
        opexPerCapacity=0.021,
        interestRate=0.08,
        economicLifetime=33,
    )
)

# %% [markdown]
# ### New combined cycle gas turbine plants for biogas

# %%
esM.add(
    fn.Conversion(
        esM=esM,
        name="New CCGT plants (biogas)",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.63},
        hasCapacityVariable=True,
        investPerCapacity=0.7,
        opexPerCapacity=0.021,
        interestRate=0.08,
        economicLifetime=33,
    )
)

# %% [markdown]
# ### New combined cycly gas turbines for hydrogen

# %%
esM.add(
    fn.Conversion(
        esM=esM,
        name="New CCGT plants (hydrogen)",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.63},
        hasCapacityVariable=True,
        investPerCapacity=0.7,
        opexPerCapacity=0.021,
        interestRate=0.08,
        economicLifetime=33,
    )
)

# %% [markdown]
# ### Electrolyzers

# %%
esM.add(
    fn.Conversion(
        esM=esM,
        name="Electrolyzer",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
        hasCapacityVariable=True,
        investPerCapacity=0.5,
        opexPerCapacity=0.5 * 0.025,
        interestRate=0.08,
        economicLifetime=10,
    )
)

# %% [markdown]
# ### rSOC

# %%
capexRSOC = 1.5

esM.add(
    fn.Conversion(
        esM=esM,
        name="rSOEC",
        physicalUnit=r"GW$_{el}$",
        linkedConversionCapacityID="rSOC",
        commodityConversionFactors={"electricity": -1, "hydrogen": 0.6},
        hasCapacityVariable=True,
        investPerCapacity=capexRSOC / 2,
        opexPerCapacity=capexRSOC * 0.02 / 2,
        interestRate=0.08,
        economicLifetime=10,
    )
)

esM.add(
    fn.Conversion(
        esM=esM,
        name="rSOFC",
        physicalUnit=r"GW$_{el}$",
        linkedConversionCapacityID="rSOC",
        commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
        hasCapacityVariable=True,
        investPerCapacity=capexRSOC / 2,
        opexPerCapacity=capexRSOC * 0.02 / 2,
        interestRate=0.08,
        economicLifetime=10,
    )
)

# %% [markdown]
# # 5. Add commodity storages to the energy system model

# %% [markdown]
# ## 5.1. Electricity storage

# %% [markdown]
# ### Lithium ion batteries
#
# The self discharge of a lithium ion battery is here described as 3% per month. The self discharge per hours is obtained using the equation (1-$\text{selfDischarge}_\text{hour})^{30*24\text{h}} = 1-\text{selfDischarge}_\text{month}$.

# %%
esM.add(
    fn.Storage(
        esM=esM,
        name="Li-ion batteries",
        commodity="electricity",
        hasCapacityVariable=True,
        chargeEfficiency=0.95,
        cyclicLifetime=10000,
        dischargeEfficiency=0.95,
        selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
        chargeRate=1,
        dischargeRate=1,
        doPreciseTsaModeling=False,
        investPerCapacity=0.151,
        opexPerCapacity=0.002,
        interestRate=0.08,
        economicLifetime=22,
    )
)

# %% [markdown]
# ## 5.2. Hydrogen storage

# %% [markdown]
# ### Hydrogen filled salt caverns
# The maximum capacity is here obtained by: dividing the given capacity (which is given for methane) by the lower heating value of methane and then multiplying it with the lower heating value of hydrogen.

# %%
esM.add(
    fn.Storage(
        esM=esM,
        name="Salt caverns (hydrogen)",
        commodity="hydrogen",
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=133,
        chargeRate=1 / 470.37,
        dischargeRate=1 / 470.37,
        sharedPotentialID="Existing salt caverns",
        stateOfChargeMin=0.33,
        stateOfChargeMax=1,
        capacityMax=data["Salt caverns (hydrogen), capacityMax"],
        investPerCapacity=0.00011,
        opexPerCapacity=0.00057,
        interestRate=0.08,
        economicLifetime=30,
    )
)

# %% [markdown]
# ## 5.3. Methane storage

# %% [markdown]
# ### Methane filled salt caverns

# %%
esM.add(
    fn.Storage(
        esM=esM,
        name="Salt caverns (biogas)",
        commodity="biogas",
        hasCapacityVariable=True,
        capacityVariableDomain="continuous",
        capacityPerPlantUnit=443,
        chargeRate=1 / 470.37,
        dischargeRate=1 / 470.37,
        sharedPotentialID="Existing salt caverns",
        stateOfChargeMin=0.33,
        stateOfChargeMax=1,
        capacityMax=data["Salt caverns (methane), capacityMax"],
        investPerCapacity=0.00004,
        opexPerCapacity=0.00001,
        interestRate=0.08,
        economicLifetime=30,
    )
)

# %% [markdown]
# ## 5.4 Pumped hydro storage

# %% [markdown]
# ### Pumped hydro storage

# %%
esM.add(
    fn.Storage(
        esM=esM,
        name="Pumped hydro storage",
        commodity="electricity",
        chargeEfficiency=0.88,
        dischargeEfficiency=0.88,
        hasCapacityVariable=True,
        selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
        chargeRate=0.16,
        dischargeRate=0.12,
        capacityFix=data["Pumped hydro storage, capacityFix"],
        investPerCapacity=0,
        opexPerCapacity=0.000153,
    )
)

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
esM.add(
    fn.Transmission(
        esM=esM,
        name="AC cables",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityFix=data["AC cables, capacityFix"],
    )
)

# %% [markdown]
# ### DC cables

# %%
esM.add(
    fn.Transmission(
        esM=esM,
        name="DC cables",
        commodity="electricity",
        losses=data["DC cables, losses"],
        distances=data["DC cables, distances"],
        hasCapacityVariable=True,
        capacityFix=data["DC cables, capacityFix"],
    )
)

# %% [markdown]
# ## 6.2 Methane transmission

# %% [markdown]
# ### Methane pipeline

# %%
esM.add(
    fn.Transmission(
        esM=esM,
        name="Pipelines (biogas)",
        commodity="biogas",
        distances=data["Pipelines, distances"],
        hasCapacityVariable=True,
        hasIsBuiltBinaryVariable=True,
        bigM=300,
        locationalEligibility=data["Pipelines, eligibility"],
        capacityMax=data["Pipelines, eligibility"] * 15,
        sharedPotentialID="pipelines",
        investPerCapacity=0.000037,
        investIfBuilt=0.000314,
        interestRate=0.08,
        economicLifetime=40,
    )
)

# %% [markdown]
# ## 6.3 Hydrogen transmission

# %% [markdown]
# ### Hydrogen pipelines

# %%
esM.add(
    fn.Transmission(
        esM=esM,
        name="Pipelines (hydrogen)",
        commodity="hydrogen",
        distances=data["Pipelines, distances"],
        hasCapacityVariable=True,
        hasIsBuiltBinaryVariable=True,
        bigM=300,
        locationalEligibility=data["Pipelines, eligibility"],
        capacityMax=data["Pipelines, eligibility"] * 15,
        sharedPotentialID="pipelines",
        investPerCapacity=0.000177,
        investIfBuilt=0.00033,
        interestRate=0.08,
        economicLifetime=40,
    )
)

# %% [markdown]
# # 7. Add commodity sinks to the energy system model

# %% [markdown]
# ## 7.1. Electricity sinks

# %% [markdown]
# ### Electricity demand

# %%
esM.add(
    fn.Sink(
        esM=esM,
        name="Electricity demand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix=data["Electricity demand, operationRateFix"],
    )
)

# %% [markdown]
# ## 7.2. Hydrogen sinks

# %% [markdown]
# ### Fuel cell electric vehicle (FCEV) demand

# %%
FCEV_penetration = 0.5
esM.add(
    fn.Sink(
        esM=esM,
        name="Hydrogen demand",
        commodity="hydrogen",
        hasCapacityVariable=False,
        operationRateFix=data["Hydrogen demand, operationRateFix"] * FCEV_penetration,
    )
)

# %% [markdown]
# ## 7.3. CO2 sinks

# %% [markdown]
# ### CO2 exiting the system's boundary

# %%
esM.add(
    fn.Sink(
        esM=esM,
        name="CO2 to enviroment",
        commodity="CO2",
        hasCapacityVariable=False,
        commodityLimitID="CO2 limit",
        yearlyLimit=366 * (1 - CO2_reductionTarget),
    )
)

# %% [markdown]
# All components are now added to the model and the model can be optimized. If the computational complexity of the optimization should be reduced, the time series data of the specified components can be clustered before the optimization and the parameter timeSeriesAggregation is set to True in the optimize call.

# %% [markdown]
# # 8 Temporal Aggregation

# %% tags=["nbval-ignore-output"]
esM.aggregateTemporally(numberOfTypicalPeriods=7)

# %% [markdown]
# ### Optimization

# %% tags=["nbval-skip"]
# The `optimizationSpecs` only work with the Gurobi solver. If you are using another solver you need to choose
# specs spcecific to this solver or no specs.
esM.optimize(
    timeSeriesAggregation=True,
    optimizationSpecs="OptimalityTol=1e-3 method=2 cuts=0 MIPGap=5e-3",
)

# %% [markdown]
# # 9. Selected results output
#
# Plot locations (GeoPandas required)

# %% tags=["nbval-skip"]
# Import the geopandas package for plotting the locations
import geopandas as gpd

# %% tags=["nbval-skip"]
locFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "clusteredRegions.shp"
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocations(locFilePath, plotLocNames=True, indexColumn="index")

# %% [markdown]
# ### Sources and Sink
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

# %% [markdown]
# Plot installed capacities

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(
    esM, "Wind (offshore)", locFilePath, "index", perArea=False
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(
    esM, "Wind (onshore)", locFilePath, "index", perArea=False
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(esM, "PV", locFilePath, "index", perArea=False)

# %% [markdown]
# Plot operation time series (either one or two dimensional)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperation(esM, "Electricity demand", "cluster_0")

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, "Electricity demand", "cluster_0")

# %% [markdown]
# ### Conversion
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("ConversionModel", outputLevel=2)

# %% tags=["nbval-skip"]
fig, ax = fn.plotLocationalColorMap(
    esM, "Electrolyzer", locFilePath, "index", perArea=False
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(esM, "New CCGT plants (biogas)", "cluster_2")

# %% [markdown]
# ### Storage
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("StorageModel", outputLevel=2)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(
    esM,
    "Li-ion batteries",
    "cluster_2",
    variableName="stateOfChargeOperationVariablesOptimum",
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(
    esM,
    "Pumped hydro storage",
    "cluster_2",
    variableName="stateOfChargeOperationVariablesOptimum",
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(
    esM,
    "Salt caverns (biogas)",
    "cluster_2",
    variableName="stateOfChargeOperationVariablesOptimum",
)

# %% tags=["nbval-skip"]
fig, ax = fn.plotOperationColorMap(
    esM,
    "Salt caverns (hydrogen)",
    "cluster_2",
    variableName="stateOfChargeOperationVariablesOptimum",
)

# %% [markdown]
# ## Transmission
#
# Show optimization summary

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("TransmissionModel", outputLevel=2)

# %% tags=["nbval-skip"]
esM.getOptimizationSummary("TransmissionModel", outputLevel=2).loc[
    "Pipelines (hydrogen)"
]

# %% [markdown]
# Check that the shared capacity of the pipelines are not exceeded

# %% tags=["nbval-skip"]
df = esM.componentModelingDict["TransmissionModel"].capacityVariablesOptimum
df.loc["Pipelines (biogas)"] + df.loc["Pipelines (hydrogen)"]

# %% [markdown]
# Plot installed transmission capacities

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "AClines.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "AC cables", transFilePath, loc0="bus0", loc1="bus1", fig=fig, ax=ax
)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "DClines.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "DC cables", transFilePath, loc0="cluster0", loc1="cluster1", fig=fig, ax=ax
)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "transmissionPipeline.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "Pipelines (hydrogen)", transFilePath, loc0="loc1", loc1="loc2", fig=fig, ax=ax
)

# %% tags=["nbval-skip"]
df = esM.componentModelingDict["TransmissionModel"].capacityVariablesOptimum
df.loc["Pipelines (biogas)"] + df.loc["Pipelines (hydrogen)"]

# %% [markdown]
# Plot installed transmission capacities

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "AClines.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "AC cables", transFilePath, loc0="bus0", loc1="bus1", fig=fig, ax=ax
)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "DClines.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "DC cables", transFilePath, loc0="cluster0", loc1="cluster1", fig=fig, ax=ax
)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "transmissionPipeline.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "Pipelines (hydrogen)", transFilePath, loc0="loc1", loc1="loc2", fig=fig, ax=ax
)

# %% tags=["nbval-skip"]
transFilePath = os.path.join(
    cwd, "InputData", "SpatialData", "ShapeFiles", "transmissionPipeline.shp"
)

fig, ax = fn.plotLocations(locFilePath, indexColumn="index")
fig, ax = fn.plotTransmission(
    esM, "Pipelines (biogas)", transFilePath, loc0="loc1", loc1="loc2", fig=fig, ax=ax
)
