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


def test_minimumPartLoad():
    """
    Two conversion components can serve the demand. One 10 GW conversion has
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
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

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


#

if __name__ == "__main__":
    test_minimumPartLoad()
