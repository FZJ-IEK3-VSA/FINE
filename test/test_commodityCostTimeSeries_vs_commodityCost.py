# https://unix.stackexchange.com/questions/74571/vim-shortcut-to-open-a-file-under-cursor-in-an-already-opened-window!/usr/bin/env python

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

# 1. Import required packages and set input data path

import fine as fn
import pandas as pd


def test_miniSystem():
    locations = {"loc1", "loc2"}
    numberOfTimeSteps = 36
    hoursPerTimeStep = 6
    commodities = {"electricity"}
    commodityUnitDict = {"electricity": r"GW$_{el}$"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )
    # print(vars(esM))

    costTS = pd.DataFrame(
        [
            [(j % 5) * (i + 1) for i in range(len(locations))]
            for j in range(numberOfTimeSteps)
        ],
        columns=["loc1", "loc2"],
    )

    costSeries = pd.Series([1, 2], index=["loc1", "loc2"], dtype=float)

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity purchase",
            commodity="electricity",
            hasCapacityVariable=False,
            # commodityCost = {0: costTS},
            commodityCost={0: costSeries},
            # commodityCost = {0:None}
            # commodityCost = 1,
            # commodityCost = None,
            # commodityCost = costSeries,
            # commodityCost = costTS,
            # commodityCostTimeSeries=costTS,
        )
    )

    demandTS = pd.DataFrame(
        [[i + 1.5 for i in range(len(locations))] for j in range(numberOfTimeSteps)],
        columns=["loc1", "loc2"],
    )
    # print(costTS, demandTS)

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            operationRateFix=demandTS,
            hasCapacityVariable=False,
            # commodityRevenue = {0: costTS},
            # commodityRevenue = {0:costSeries},
            # commodityRevenue = {0:None}
            # commodityRevenue = 2,
            # commodityRevenue = None,
            # commodityRevenue = costSeries,
            commodityRevenue=costTS,
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
