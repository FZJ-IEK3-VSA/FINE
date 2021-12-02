#!/usr/bin/env python
# coding: utf-8

# # Workflow for a multi-regional energy system
#
import FINE as fn
import os
import pandas as pd
import numpy as np

import sys

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "Multi-regional_Energy_System_Workflow",
    )
)


def test_rampUpMax():
    # read in original results
    results = [
        8.0,
        10.0,
        0.0,
        0.0,
        4.0,
        5.0,
        5.0,
        0.0,
        0.0,
        4.0,
        8.0,
        5.0,
        4.0,
        0.0,
        4.0,
        5.0,
        5.0,
        0.0,
        0.0,
        4.0,
    ]

    # 2. Create an energy system model instance
    locations = {"example_region1", "example_region2"}
    commodityUnitDict = {"electricity": r"GW$_{el}$", "methane": r"GW$_{CH_{4},LHV}$"}
    commodities = {"electricity", "methane"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=20,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    data_cost = {
        "example_region1": [
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
        ],
        "example_region2": [
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
            10,
        ],
    }
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

    data_cap = pd.Series(index=["example_region1", "example_region2"], data=[10, 10])

    esM.add(
        fn.ConversionDynamic(
            esM=esM,
            name="unrestricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
            capacityFix=data_cap,
            partLoadMin=0.1,
            bigM=100,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            opexPerOperation=10,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    esM.add(
        fn.ConversionDynamic(
            esM=esM,
            name="restricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
            capacityFix=data_cap,
            partLoadMin=0.3,
            bigM=100,
            rampUpMax=0.4,
            investPerCapacity=0.5,
            opexPerCapacity=0.021,
            opexPerOperation=1,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    data_demand = {
        "example_region1": [
            10,
            10,
            2,
            2,
            4,
            5,
            5,
            2,
            2,
            10,
            10,
            5,
            4,
            2,
            4,
            5,
            5,
            2,
            2,
            4,
        ],
        "example_region2": [
            10,
            10,
            2,
            2,
            4,
            5,
            5,
            2,
            2,
            10,
            10,
            5,
            4,
            2,
            4,
            5,
            5,
            2,
            2,
            4,
        ],
    }
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
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    print("restricted dispatch:\n")
    print(
        esM.componentModelingDict[
            "ConversionDynamicModel"
        ].operationVariablesOptimum.xs("restricted")
    )
    print("unrestricted dispatch:\n")
    print(
        esM.componentModelingDict[
            "ConversionDynamicModel"
        ].operationVariablesOptimum.xs("unrestricted")
    )
    #    print(esM.componentModelingDict['ConversionDynamicModel'].operationVariablesOptimum.xs('unrestricted'))
    # test if here solved fits with original results

    # test if here solved fits with original results
    testresults = esM.componentModelingDict[
        "ConversionDynamicModel"
    ].operationVariablesOptimum.xs("restricted")
    np.testing.assert_array_almost_equal(testresults.values[0], results, decimal=2)
    np.testing.assert_array_almost_equal(testresults.values[1], results, decimal=2)


##

if __name__ == "__main__":
    test_rampUpMax()
