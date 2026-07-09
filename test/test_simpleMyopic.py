import fine as fn
import numpy as np
import pandas as pd
import pytest

from fine.utils import ImplementedSolvers

np.random.seed(
    42
)  # Sets a "seed" to produce the same random input data in each model run

# fn.optimizeSimpleMyopic is deprecated; fine.expansionModules.rollingHorizon.
# rollingHorizonOptimization with numberOfInvestmentPeriodsForRollingHorizon=1
# covers the same myopic foresight use case (see issue #640) and is the
# recommended replacement. It also supports CO2 reduction pathways (via a
# per-investment-period balanceLimit) and technical-lifetime expiry, which are
# covered by test_co2_target_* and test_exceeded_lifetime_* in
# test/test_rolling_horizon.py.


def test_exceededLifetime():
    # load a minimal test system
    """Returns minimal instance of esM."""
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"OneLocation"},
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

    ### Buy electricity at the electricity market
    costs = pd.DataFrame(
        [
            np.array(
                [
                    0.05,
                    0.0,
                    0.1,
                    0.051,
                ]
            )
        ],
        index=["OneLocation"],
    ).T
    revenues = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                ]
            )
        ],
        index=["OneLocation"],
    ).T
    maxpurchase = (
        pd.DataFrame(
            [
                np.array(
                    [
                        1e6,
                        1e6,
                        1e6,
                        1e6,
                    ]
                )
            ],
            index=["OneLocation"],
        ).T
        * hoursPerTimeStep
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=maxpurchase,
            commodityCostTimeSeries=costs,
            commodityRevenueTimeSeries=revenues,
        )
    )  # eur/kWh

    ### Electrolyzers
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
        )
    )

    ### Hydrogen filled somewhere
    esM.add(
        fn.Storage(
            esM=esM,
            name="Pressure tank",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ### Industry site
    demand = (
        pd.DataFrame(
            [
                np.array(
                    [
                        6e3,
                        6e3,
                        6e3,
                        6e3,
                    ]
                )
            ],
            index=["OneLocation"],
        ).T
        * hoursPerTimeStep
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # Set the technical lifetime of the electrolyzers to 7 years.
    setattr(
        esM.componentModelingDict["ConversionModel"].componentsDict["Electrolyzers"],
        "technicalLifetime",
        pd.Series([7], index=["OneLocation"]),
    )

    with pytest.deprecated_call():
        results = fn.optimizeSimpleMyopic(
            esM,
            startYear=2020,
            endYear=2030,
            nbOfRepresentedYears=5,
            timeSeriesAggregation=False,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
            saveResults=False,
            trackESMs=True,
        )

    # Check if electrolyzers which are installed in 2020 are not included in the system of 2030 due to the exceeded lifetime
    assert "Electrolyzers_stock_2020" not in results["ESM_2030"].componentNames.keys()
