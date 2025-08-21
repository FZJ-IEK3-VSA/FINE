# %%
import fine as fn
import numpy as np
import pandas as pd


def test_operationRateMin(minimal_test_esM):
    """Test that the minimum operation rate of a conversion component is enforced.

    A Conversion component representing electrolyzers is added to a minimal
    energy system model with a specified minimum load factor. After
    optimization, the test verifies that the actual operation of the
    component never falls below the specified minimum operation rate.
    """
    esM = minimal_test_esM
    numberOfTimeSteps = esM.numberOfTimeSteps

    min_load_factor = 0.8
    operationRateMin = pd.DataFrame(
        np.ones((numberOfTimeSteps, 2)) * min_load_factor,
        columns=["ElectrolyzerLocation", "IndustryLocation"],
    )
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            operationRateMin=operationRateMin,
        )
    )
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    ts = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.loc[
        "Electrolyzers"
    ]
    cap = esM.componentModelingDict["ConversionModel"].capacityVariablesOptimum.loc[
        "Electrolyzers"
    ]

    cf_ts = ts.div(cap, axis=0) / esM.hoursPerTimeStep

    assert (cf_ts.min() >= min_load_factor).all()
