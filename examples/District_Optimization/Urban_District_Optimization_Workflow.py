# -*- coding: utf-8 -*-
# %% [markdown]
# # Workflow for a district optimization
#
# In this application of the FINE framework, a small district is modeled and optimized.
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
from getData import getData
import pandas as pd

data = getData()

# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# %% [markdown]
# # 2. Create an energy system model instance
#
# The structure of the energy system model is given by the considered locations, commodities, the number of time steps as well as the hours per time step.
#
# The commodities are specified by a unit (i.e. 'GW_electric', 'GW_H2lowerHeatingValue', 'Mio. t CO2/h') which can be given as an energy or mass unit per hour. Furthermore, the cost unit and length unit are specified.

# %%
locations = data["locations"]
commodityUnitDict = {"electricity": "kW_el", "methane": "kW_CH4_LHV", "heat": "kW_th"}
commodities = {"electricity", "methane", "heat"}
numberOfTimeSteps = 8760
hoursPerTimeStep = 1

# %%
esM = fn.EnergySystemModel(
    locations=locations,
    commodities=commodities,
    numberOfTimeSteps=8760,
    commodityUnitsDict=commodityUnitDict,
    hoursPerTimeStep=1,
    costUnit="€",
    lengthUnit="m",
    verboseLogLevel=2,
)

# %% [markdown]
# # 3. Add commodity sources to the energy system model

# %% [markdown]
# ### Electricity Purchase

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="Electricity purchase",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateMax=data["El Purchase, operationRateMax"],
        commodityCost=0.298,
    )
)

# %% [markdown]
# ### Natural Gas Purchase

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="NaturalGas purchase",
        commodity="methane",
        hasCapacityVariable=False,
        operationRateMax=data["NG Purchase, operationRateMax"],
        commodityCost=0.065,
    )
)

# %% [markdown]
# ### PV

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="PV",
        commodity="electricity",
        hasCapacityVariable=True,
        hasIsBuiltBinaryVariable=True,
        operationRateMax=data["PV, operationRateMax"],
        capacityMax=data["PV, capacityMax"],
        interestRate=0.04,
        economicLifetime=20,
        investIfBuilt=1000,
        investPerCapacity=1400,
        opexIfBuilt=10,
        bigM=40,
    )
)

# %% [markdown]
# # 4. Add conversion components to the energy system model

# %% [markdown]
# ### Boiler

# %%
esM.add(
    fn.Conversion(
        esM=esM,
        name="Boiler",
        physicalUnit="kW_th",
        commodityConversionFactors={"methane": -1.1, "heat": 1},
        hasIsBuiltBinaryVariable=True,
        hasCapacityVariable=True,
        interestRate=0.04,
        economicLifetime=20,
        investIfBuilt=2800,
        investPerCapacity=100,
        opexIfBuilt=24,
        bigM=200,
    )
)

# %% [markdown]
# # 5. Add commodity storages to the energy system model

# %% [markdown]
# ### Thermal Storage

# %%
esM.add(
    fn.Storage(
        esM=esM,
        name="Thermal Storage",
        commodity="heat",
        selfDischarge=0.001,
        hasIsBuiltBinaryVariable=True,
        capacityMax=data["TS, capacityMax"],
        interestRate=0.04,
        economicLifetime=25,
        investIfBuilt=23,
        investPerCapacity=24,
        bigM=250,
    )
)

# %% [markdown]
# ### Battery Storage

# %%
esM.add(
    fn.Storage(
        esM=esM,
        name="Battery Storage",
        commodity="electricity",
        cyclicLifetime=10000,
        chargeEfficiency=0.95,
        dischargeEfficiency=0.95,
        chargeRate=0.5,
        dischargeRate=0.5,
        hasIsBuiltBinaryVariable=True,
        capacityMax=data["BS, capacityMax"],
        interestRate=0.04,
        economicLifetime=12,
        investIfBuilt=2000,
        investPerCapacity=700,
        bigM=110,
    )
)

# %% [markdown]
# # 6. Add commodity transmission components to the energy system model

# %% [markdown]
# ### Cable Electricty

# %%
esM.add(
    fn.Transmission(
        esM=esM,
        name="E_Distribution_Grid",
        commodity="electricity",
        losses=0.00001,
        distances=data["cables, distances"],
        capacityFix=data["cables, capacityFix"],
    )
)

# %% [markdown]
# ### Natural Gas Pipeline

# %%
esM.add(
    fn.Transmission(
        esM=esM,
        name="NG_Distribution_Grid",
        commodity="methane",
        distances=data["NG, distances"],
        capacityFix=data["NG, capacityFix"],
    )
)

# %% [markdown]
# # 7. Add commodity sinks to the energy system model
#
# ### Electricity Demand

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
# ### Heat Demand

# %%
esM.add(
    fn.Sink(
        esM=esM,
        name="BuildingsHeat",
        commodity="heat",
        hasCapacityVariable=False,
        operationRateFix=data["Heat demand, operationRateFix"],
    )
)

# %% [markdown]
# # 8. Optimize energy system model

# %% [markdown]
# All components are now added to the model and the model can be optimized. If the computational complexity of the optimization should be reduced, the time series data of the specified components can be clustered before the optimization and the parameter timeSeriesAggregation is set to True in the optimize call.

# %%
esM.aggregateTemporally(numberOfTypicalPeriods=7)

# %%
# esM.optimize(timeSeriesAggregation=True, optimizationSpecs='cuts=0 method=2')
esM.optimize(timeSeriesAggregation=True, solver="glpk")

# %% [markdown]
# # 9. Selected results output

# %% [markdown]
# ### Sources and Sink

# %%
esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "PV", "bd1")

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "Electricity demand", "bd1")

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "Electricity purchase", "transformer")

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "NaturalGas purchase", "transformer")

# %% [markdown]
# ### Conversion

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("ConversionModel", outputLevel=2)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(esM, "Boiler", "bd1")

# %% [markdown]
# ### Storage

# %%
esM.getOptimizationSummary("StorageModel", outputLevel=2)

# %% tags=["nbval-check-output"]
fig, ax = fn.plotOperationColorMap(
    esM, "Thermal Storage", "bd1", variableName="stateOfChargeOperationVariablesOptimum"
)

# %% [markdown]
# ### Transmission

# %%
esM.getOptimizationSummary("TransmissionModel", outputLevel=2)
