#!/usr/bin/env python

# # Workflow for a multi-regional energy system
#
import fine as fn
import pandas as pd
import numpy as np
import pytest


HEAT_GRID_PRICE = 0.5
GAS_PRICE = 0.1

DEMAND = 5
PARTLOADMIN = 6
CAPACITY = 10

OPERATION_HOURS = 10


@pytest.mark.parametrize("hoursPerTimeStep", [0.25, 1])
def test_conversionPartLoad_simple(hoursPerTimeStep):
    """Create energy system with several components.

    Methan boiler is forced to produce with higher rate than heat demand due to partLoadMin.
    The rest of the produced heat is dumped.
    Heat purchase (grid) -------------------------------------------->
                                                                        Heat Demand + DUMMY Source
    Methane purchase -----> Methane boiler (Conversion Dynamic) ----->
    """
    esM = fn.EnergySystemModel(
        locations={
            "region1",
        },
        numberOfTimeSteps=int(OPERATION_HOURS / hoursPerTimeStep),
        hoursPerTimeStep=hoursPerTimeStep,
        commodities={"electricity", "methane", "heat"},
        commodityUnitsDict={"electricity": "kW", "methane": "kW", "heat": "kW"},
        verboseLogLevel=2,
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Methane heater",
            physicalUnit="kW",
            commodityConversionFactors={
                "methane": -1,
                "heat": 1,
            },
            hasCapacityVariable=True,
            capacityFix=CAPACITY,
            partLoadMin=PARTLOADMIN / CAPACITY,
            bigM=1000,
        )
    )

    # add heat source from grid
    esM.add(
        fn.Source(
            esM=esM,
            name="Heating Grid",
            commodity="heat",
            hasCapacityVariable=False,
            commodityCost=HEAT_GRID_PRICE,  # some value higher than methan
        )
    )

    # add methane purchase
    esM.add(
        fn.Source(
            esM=esM,
            name="Methane purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCost=GAS_PRICE,
        )
    )
    # add methane purchase
    esM.add(
        fn.Sink(
            esM=esM,
            name="Heat demand",
            commodity="heat",
            operationRateFix=pd.Series(
                data=[DEMAND * hoursPerTimeStep]
                * int(OPERATION_HOURS / hoursPerTimeStep)
            ),
            hasCapacityVariable=False,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Dummy Sink",
            commodity="heat",
            hasCapacityVariable=False,
        )
    )
    esM.optimize()

    # Check results: operation of the methane heater must be 3h*10kW less
    expectedOperation = PARTLOADMIN * OPERATION_HOURS

    heater_operation = (
        esM.getOptimizationSummary("ConversionModel")
        .loc["Methane heater", "operation", "[kW*h]"]
        .loc["region1"]
    )

    assert expectedOperation == heater_operation


def test_minimumPartLoad():
    """Two conversion components can serve the demand. One 10 GW conversion has
    high operation costs and no investment costs, one varible sized conversion
    has low operation costs but investment costs. The 10 GW conversion is
    restricted to a minimum part load of 4 GW.

    The cost optimal solution builds 1 GW of the component with low operation
    costs and runs it whenever possible. A higher capacity of this component
    would not be economically beneficial since the 10 GW has no cost. The
    restricted component should not run under 4 GW.
    """
    # read in original results
    results = [4.0, 4.0, 0.0, 0.0, 4.0]

    # 2. Create an energy system model instance
    locations = {"example_region"}
    commodityUnitDict = {"electricity": r"GW$_{el}$", "methane": r"GW$_{CH_{4},LHV}$"}
    commodities = {"electricity", "methane"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=5,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    data_cost = {"example_region": [10, 10, 10, 10, 10]}
    data_cost_df = pd.DataFrame(data=data_cost)

    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCostTimeSeries=data_cost_df,
        )
    )

    # 4. Add conversion components to the energy system model

    ### Combined cycle gas turbine plants

    esM.add(
        fn.Conversion(
            esM=esM,
            name="unrestricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
            hasCapacityVariable=True,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            opexPerOperation=0.01 / 8760,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    data_cap = pd.Series(index=["example_region"], data=10)

    esM.add(
        fn.Conversion(
            esM=esM,
            name="restricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
            capacityFix=data_cap,
            partLoadMin=0.4,
            bigM=10000,
            opexPerOperation=0.02 / 8760,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    data_demand = {"example_region": [5, 5, 1, 1, 4]}
    data_demand_df = pd.DataFrame(data=data_demand)
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=data_demand_df,
        )
    )
    esM.optimize(timeSeriesAggregation=False, solver="appsi_highs")

    print("restricted dispatch:\n")
    print(
        esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs(
            "restricted"
        )
    )
    print("unrestricted dispatch:\n")
    print(
        esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs(
            "unrestricted"
        )
    )
    # test if here solved fits with original results

    # test if here solved fits with original results
    testresults = esM.componentModelingDict[
        "ConversionModel"
    ].operationVariablesOptimum.xs("restricted")
    np.testing.assert_array_almost_equal(testresults.values[0], results, decimal=2)
