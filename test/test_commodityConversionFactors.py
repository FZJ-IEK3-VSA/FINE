import FINE as fn
import pandas as pd
import numpy as np
import pytest

"""
Here we are testing differnt inputs for time-invariant conversion factors that are
not covered in the minimal test system or other tests.
"""


def create_core_esm():
    """
    We create a core esm that only consists of a source and a sink in one location.
    """
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    # Source
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )
    # Sink
    demand = pd.Series(np.array([1.0, 1.0, 1.0, 1.0])) * hoursPerTimeStep
    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    return esM


def test_conversion_factors_as_series():
    """
    Input as pandas.Series for one location.
    """

    esM = create_core_esm()

    with pytest.raises(
        ValueError, match=r".*commodityConversionFactor must be a dict.*"
    ):
        esM.add(
            fn.Conversion(
                esM=esM,
                name="Electrolyzers_VarConvFac",
                physicalUnit=r"kW$_{el}$",
                commodityConversionFactors=pd.Series(
                    [0.7, -1], index=["hydrogen", "electricity"]
                ),  # Here we add a Series of time invariant conversion factors.
                hasCapacityVariable=True,
                investPerCapacity=1000,  # euro/kW
                opexPerCapacity=500 * 0.025,
                interestRate=0.08,
                capacityMax=1000,
                economicLifetime=10,
                locationalEligibility=pd.Series([1], ["ElectrolyzerLocation"]),
            )
        )

    # optimize
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
