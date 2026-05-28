import pandas as pd
import pytest
import fine as fn

def test_shared_potential_capacity_fix_exceeds_capacity_max():
    """
    Test whether a ValueError is raised if the sum of fixed capacities
    of components with the same sharedPotentialID exceeds the available
    shared potential.

    The test passes if the expected error is raised successfully.
    """
    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        numberOfTimeSteps=24,
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
    )

    dailyProfilePV = [
        0, 0, 0, 0, 0, 0, 0,
        0.05, 0.15, 0.2, 0.4, 0.8,
        0.7, 0.4, 0.2, 0.15, 0.05,
        0, 0, 0, 0, 0, 0, 0,
    ]

    dailyProfileWind = [
        0.6, 0.55, 0.5, 0.45, 0.4, 0.5,
        0.6, 0.7, 0.75, 0.8, 0.7, 0.65,
        0.6, 0.55, 0.5, 0.45, 0.5, 0.6,
        0.7, 0.75, 0.8, 0.75, 0.7, 0.65,
    ]

    operationRateMaxPV = pd.DataFrame(
        {"loc1": dailyProfilePV},
        index=range(24),
    )

    operationRateMaxWind = pd.DataFrame(
        {"loc1": dailyProfileWind},
        index=range(24),
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="source_1",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=operationRateMaxPV,
            capacityFix=0.7,
            capacityMax=1,
            sharedPotentialID="sharedPot",
        )
    )

    with pytest.raises(ValueError, match="sharedPotentialID"):
        esM.add(
            fn.Source(
                esM=esM,
                name="source_2",
                commodity="electricity",
                hasCapacityVariable=True,
                operationRateMax=operationRateMaxWind,
                capacityFix=0.7,
                capacityMax=2,
                sharedPotentialID="sharedPot",
            )
        )