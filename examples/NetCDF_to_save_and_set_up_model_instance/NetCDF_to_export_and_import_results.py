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
# # Save and read optimization results with netCDF

# %% tags=["nbval-ignore-output"]
import os
import sys

import xarray

import FINE as fn
import FINE.IOManagement.xarrayIO as xrIO

import sys

sys.path.append(
    os.path.join(os.getcwd(), "..", "Multi-regional_Energy_System_Workflow")
)
from getData import getData

data = getData()

# 1. Create an energy system model instance
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

CO2_reductionTarget = 1


# 2. Add commodity sources to the energy system model
### Wind onshore

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

### PV

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


# 3. Add conversion components to the energy system model

### New combined cycly gas turbines for hydrogen
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

### Electrolyzers
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


# 4. Add commodity storages to the energy system model

### Lithium ion batteries
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

### Hydrogen filled salt caverns
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


# 5. Add commodity transmission components to the energy system model

### AC cables
esM.add(
    fn.LinearOptimalPowerFlow(
        esM=esM,
        name="AC cables",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityFix=data["AC cables, capacityFix"],
        reactances=data["AC cables, reactances"],
    )
)

### DC cables
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


### Hydrogen pipelines
esM.add(
    fn.Transmission(
        esM=esM,
        name="Pipelines (hydrogen)",
        commodity="hydrogen",
        distances=data["Pipelines, distances"],
        hasCapacityVariable=True,
        hasIsBuiltBinaryVariable=False,
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

# 6. Add commodity sinks to the energy system model

### Electricity demand
esM.add(
    fn.Sink(
        esM=esM,
        name="Electricity demand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix=data["Electricity demand, operationRateFix"],
    )
)

## 7.2. Hydrogen sinks
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


esM.aggregateTemporally(numberOfTypicalPeriods=7)
esM.optimize(
    timeSeriesAggregation=True,
    optimizationSpecs="OptimalityTol=1e-3 method=2 cuts=0 MIPGap=5e-3",
)


# %% tags=["nbval-ignore-output"]
xr_dss = xrIO.writeOptimizationOutputToNetCDF(
    esM,
    outputFileName="my_esm_results.nc4",
    overwrite_existing=True,
    optSumOutputLevel=2,
    optValOutputLevel=1,
)

# %% tags=["nbval-ignore-output"]
xr_dss["SourceSinkModel"]["Wind (onshore)"]

# %% tags=["nbval-ignore-output"]
xr_dss["SourceSinkModel"]["Wind (onshore)"].drop_dims("time").to_dataframe().T

# %% tags=["nbval-ignore-output"]
xr_dss["SourceSinkModel"]["Wind (onshore)"]["operationVariablesOptimum"].plot(
    col="space"
)

# %% tags=["nbval-ignore-output"]
xr_dss["StorageModel"]["Salt caverns (hydrogen)"][
    "stateOfChargeOperationVariablesOptimum"
].plot(hue="space");

# %% tags=["nbval-ignore-output"]
# Read from netCDF file
xr_dss_imported = xrIO.readOptimizationOutputFromNetCDF(
    inputFileName="my_esm_results.nc4"
)
