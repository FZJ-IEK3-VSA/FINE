# %% [markdown]
# # Workflow for a 1node energy system
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
from getData import getData

# import pandas as pd
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
locations = {"GermanyRegion"}
commodityUnitDict = {"electricity": r"GW$_{el}$", "hydrogen": r"GW$_{H_{2},LHV}$"}
commodities = {"electricity", "hydrogen"}
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

# %% tags=["nbval-check-output"]
data["Wind (onshore), operationRateMax"].sum()

# %% [markdown]
# # 4. Add conversion components to the energy system model

# %% [markdown]
# ### New combined cycly gas turbines for hydrogen

# %%
esM.add(
    fn.Conversion(
        esM=esM,
        name="New CCGT plants (hydrogen)",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
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
        name="Electroylzers",
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
# # 8. Optimize energy system model

# %% [markdown]
# All components are now added to the model and the model can be optimized. If the computational complexity of the optimization should be reduced, the time series data of the specified components can be clustered before the optimization and the parameter timeSeriesAggregation is set to True in the optimize call.

# %%
esM.aggregateTemporally(numberOfTypicalPeriods=30)

# %%
esM.optimize(timeSeriesAggregation=True, solver="glpk")

# %% [markdown]
# # 9. Selected results output

# %% [markdown]
# ### Sources and Sink
#
# Show optimization summary

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

# %% [markdown]
# Plot operation time series (either one or two dimensional)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperation(esM, "Electricity demand", "GermanyRegion")

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "Electricity demand", "GermanyRegion")

# %% [markdown]
# ### Conversion
#
# Show optimization summary

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("ConversionModel", outputLevel=2)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "New CCGT plants (hydrogen)", "GermanyRegion")

# %% [markdown]
# ### Storage
#
# Show optimization summary

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("StorageModel", outputLevel=2)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(
    esM,
    "Li-ion batteries",
    "GermanyRegion",
    variableName="stateOfChargeOperationVariablesOptimum",
)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(
    esM,
    "Salt caverns (hydrogen)",
    "GermanyRegion",
    variableName="stateOfChargeOperationVariablesOptimum",
)
