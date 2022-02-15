# -*- coding: utf-8 -*-
# %%
import FINE as fn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # Water supply of a small mountain village
#
# Two new houses (house 5 and 6) were built in a small mountain village in Utopia which requires an update of the existing clean water supply system of the village which now consists of 6 houses:
#
# <img src="MountainVillage.png" style="width: 500px;"/>
#
#
# ### Water demand
# The demand for clean water occurs in spring between 5 am and 11 pm, in summer between 4 am and 12 pm, in autumn between 5 am and 11 pm and in winter between 6 am and 11 pm. The demand for one house assumes random values between 0 to 1 Uh (Unit*hour) during the demand hours. These values are uniformly distributed and are 0 outside the demand hours.
#
# ### Water supply
# The water supply comes from a small tributary of a glacier river, which provides more water in summer and less in winter: the profile is given for each hour of the year as
#
# f(t) = 8 \* sin(π*t/8760) + g(t)
#
# where g(t) is a uniformly distributed random value between 0 and 4.
#
# ### Water storage
# Clean water can be stored in a water tank (newly purchased). The invest per capacity is 100€/Uh, the economic lifetime is 20 years.
#
# ### Water treatment
# The river water is converted to clean water in a water treatment plant (newly purchased). The invest per capacity is 7000€/U, the economic lifetime is 20 years. Further, it needs some electricity wherefore it has operational cost of 0.05 €/U.
#
# ### Water transmission
# The clean water can be transported via water pipes, where some already exist between the houses 1-4, the water treatment plant and the
# water tank, however new ones might need to
# be built to connect the newly built houses or reinforce the transmission along the old pipes. The invest for new pipes per capacity is 100 €/(m\*U), the invest if a new pipe route is built is 500 €/(m\*U), the economic lifetime is 20 years.
#

# %%
locations = [
    "House 1",
    "House 2",
    "House 3",
    "House 4",
    "House 5",
    "House 6",
    "Node 1",
    "Node 2",
    "Node 3",
    "Node 4",
    "Water treatment",
    "Water tank",
]
commodityUnitDict = {"clean water": "U", "river water": "U"}
commodities = {"clean water", "river water"}
numberOfTimeSteps = 8760
hoursPerTimeStep = 1

# %%
esM = fn.EnergySystemModel(
    locations=set(locations),
    commodities=commodities,
    numberOfTimeSteps=8760,
    commodityUnitsDict=commodityUnitDict,
    hoursPerTimeStep=1,
    costUnit="1e3 Euro",
    lengthUnit="m",
)

# %% [markdown]
# # Source

# %%
riverFlow = pd.DataFrame(np.zeros((8760, 12)), columns=locations)
np.random.seed(42)
riverFlow.loc[:, "Water treatment"] = np.random.uniform(0, 4, (8760)) + 8 * np.sin(
    np.pi * np.arange(8760) / 8760
)

# %%
esM.add(
    fn.Source(
        esM=esM,
        name="River",
        commodity="river water",
        hasCapacityVariable=False,
        operationRateMax=riverFlow,
        opexPerOperation=0.05,
    )
)

# %% [markdown]
# # Conversion

# %%
eligibility = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], index=locations)
esM.add(
    fn.Conversion(
        esM=esM,
        name="Water treatment plant",
        physicalUnit="U",
        commodityConversionFactors={"river water": -1, "clean water": 1},
        hasCapacityVariable=True,
        locationalEligibility=eligibility,
        investPerCapacity=7,
        opexPerCapacity=0.02 * 7,
        interestRate=0.08,
        economicLifetime=20,
    )
)

# %% [markdown]
# # Storage

# %%
eligibility = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], index=locations)
esM.add(
    fn.Storage(
        esM=esM,
        name="Water tank",
        commodity="clean water",
        hasCapacityVariable=True,
        chargeRate=1 / 24,
        dischargeRate=1 / 24,
        locationalEligibility=eligibility,
        investPerCapacity=0.10,
        opexPerCapacity=0.02 * 0.1,
        interestRate=0.08,
        economicLifetime=20,
    )
)

# %% [markdown]
# # Transmission

# %% [markdown]
# ### Distances between eligible regions

# %% tags=["nbval-check-output"]
distances = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 40, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 38, 40, 0, 105, 0, 0, 0, 0],
        [0, 0, 38, 40, 0, 0, 105, 0, 100, 0, 0, 0],
        [38, 40, 0, 0, 0, 0, 0, 100, 0, 30, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 20, 50],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 0, 0],
    ]
)

distances = pd.DataFrame(distances, index=locations, columns=locations)
distances

# %% [markdown]
# ## Old water pipes

# %% tags=["nbval-check-output"]
capacityFix = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 2, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    ]
)

capacityFix = pd.DataFrame(capacityFix, index=locations, columns=locations)
capacityFix

# %% [markdown]
# The old pipes have many leckages wherefore they lose 0.1%/m of the water they transport.

# %%
isBuiltFix = capacityFix.copy()
isBuiltFix[isBuiltFix > 0] = 1

esM.add(
    fn.Transmission(
        esM=esM,
        name="Old water pipes",
        commodity="clean water",
        losses=0.1e-2,
        distances=distances,
        hasCapacityVariable=True,
        hasIsBuiltBinaryVariable=True,
        bigM=100,
        capacityFix=capacityFix,
        isBuiltFix=isBuiltFix,
    )
)

# %% [markdown]
# ## New water pipes

# %% tags=["nbval-check-output"]
incidence = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]
)

eligibility = pd.DataFrame(incidence, index=locations, columns=locations)
eligibility

# %% [markdown]
# The new are pipes are better but still lose 0.05%/m of the water they transport.

# %%
esM.add(
    fn.Transmission(
        esM=esM,
        name="New water pipes",
        commodity="clean water",
        losses=0.05e-2,
        distances=distances,
        hasCapacityVariable=True,
        hasIsBuiltBinaryVariable=True,
        bigM=100,
        locationalEligibility=eligibility,
        investPerCapacity=0.1,
        investIfBuilt=0.5,
        interestRate=0.08,
        economicLifetime=50,
    )
)

# %% [markdown]
# # Sink

# %%
winterHours = np.append(range(8520, 8760), range(1920))
springHours, summerHours, autumnHours = (
    np.arange(1920, 4128),
    np.arange(4128, 6384),
    np.arange(6384, 8520),
)

demand = pd.DataFrame(np.zeros((8760, 12)), columns=list(locations))
np.random.seed(42)
demand[locations[0:6]] = np.random.uniform(0, 1, (8760, 6))

demand.loc[winterHours[(winterHours % 24 < 5) | (winterHours % 24 >= 23)]] = 0
demand.loc[springHours[(springHours % 24 < 4)]] = 0
demand.loc[summerHours[(summerHours % 24 < 5) | (summerHours % 24 >= 23)]] = 0
demand.loc[autumnHours[(autumnHours % 24 < 6) | (autumnHours % 24 >= 23)]] = 0

# %%
demand.sum().sum()

# %%
esM.add(
    fn.Sink(
        esM=esM,
        name="Water demand",
        commodity="clean water",
        hasCapacityVariable=False,
        operationRateFix=demand,
    )
)

# %% [markdown]
# # Optimize the system

# %%
esM.aggregateTemporally(numberOfTypicalPeriods=7)

# %%
# esM.optimize(timeSeriesAggregation=True, optimizationSpecs='LogToConsole=1 OptimalityTol=1e-6 crossover=1')
esM.optimize(timeSeriesAggregation=True, solver="glpk")

# %% [markdown]
# # Selected results output

# %% [markdown]
# ### Sources and Sinks

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)

# %% [markdown]
# ### Storage

# %%
esM.getOptimizationSummary("StorageModel", outputLevel=2)

# %% [markdown]
# ### Conversion

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("ConversionModel", outputLevel=2)

# %% [markdown]
# ### Transmission

# %% tags=["nbval-check-output"]
esM.getOptimizationSummary("TransmissionModel", outputLevel=2)

# %% tags=["nbval-check-output"]
esM.componentModelingDict["TransmissionModel"].operationVariablesOptimum.sum(axis=1)
