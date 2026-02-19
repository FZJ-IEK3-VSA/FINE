import fine as fn
import pandas as pd
import numpy as np
import pytest

"""
Here we are testing differnt inputs for time-invariant conversion factors that are
not covered in the minimal test system or other tests.
"""


def create_core_esm():
    """We create a core esm that only consists of a source and a sink in one location."""
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


def create_two_loc_esm():
    """We create a core esm with two locations, one source and one sink."""
    numberOfTimeSteps = 1
    hoursPerTimeStep = 1
    esM = fn.EnergySystemModel(
        locations={"Loc1", "Loc2"},
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
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )
    demand = pd.DataFrame({"Loc1": [10.0], "Loc2": [10.0]}, index=esM.totalTimeSteps)
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
    """Input as pandas.Series for one location."""
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


def test_location_specific_conversion_factors_series():
    """Location-specific conversion factors are accepted as pandas.Series."""
    esM = create_two_loc_esm()

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_LocCcf",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "hydrogen": pd.Series({"Loc1": 0.5, "Loc2": 1.0}),
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    comp = esM.getComponent("Electrolyzers_LocCcf")
    processed = comp.processedCommodityConversionFactors[0]["hydrogen"]
    assert isinstance(processed, pd.Series)
    np.testing.assert_almost_equal(processed.loc["Loc1"], 0.5)
    np.testing.assert_almost_equal(processed.loc["Loc2"], 1.0)

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    op = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs(
        "Electrolyzers_LocCcf"
    )
    # Hydrogen deamnd in both regions is 10 kWh.
    # The conversion factor for Loc1 is 0.5, so to produce 10 kWh of hydrogen, we need to consume 20 kWh of electricity.
    # For Loc2 the conversion factor is 1.0, so to produce 10 kWh of hydrogen, we need to consume 10 kWh of electricity.
    np.testing.assert_almost_equal(op.loc["Loc1", 0], 20.0, decimal=6)
    np.testing.assert_almost_equal(op.loc["Loc2", 0], 10.0, decimal=6)


def test_location_specific_timeseries_conversion_factors_dataframe():
    """Location-specific time series conversion factors via DataFrame (4 timesteps)."""

    # --- Build ESM ---
    numberOfTimeSteps = 4
    hoursPerTimeStep = 1

    esM = fn.EnergySystemModel(
        locations={"Loc1", "Loc2"},
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

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
        )
    )

    # --- Hydrogen demand (10 each timestep, each location) ---
    demand = pd.DataFrame(
        {
            "Loc1": [10.0, 10.0, 10.0, 10.0],
            "Loc2": [10.0, 10.0, 10.0, 10.0],
        },
        index=esM.totalTimeSteps,
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

    # --- Time- & location-specific conversion factors ---
    # Efficiency varies over time AND location
    ccf_h2 = pd.DataFrame(
        {
            "Loc1": [0.5, 0.6, 0.7, 0.8],
            "Loc2": [1.0, 0.9, 0.8, 0.7],
        },
        index=esM.totalTimeSteps,
        dtype="float64",
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers_LocTsCcf",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "hydrogen": ccf_h2,
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # --- Check processed structure ---
    comp = esM.getComponent("Electrolyzers_LocTsCcf")
    full = comp.fullCommodityConversionFactors[0]["hydrogen"]

    assert isinstance(full, pd.DataFrame)

    # Check a few entries
    np.testing.assert_almost_equal(full.at[(0, 0), "Loc1"], 0.5)
    np.testing.assert_almost_equal(full.at[(0, 3), "Loc1"], 0.8)
    np.testing.assert_almost_equal(full.at[(0, 1), "Loc2"], 0.9)

    # --- Optimize ---
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    op = esM.componentModelingDict["ConversionModel"].operationVariablesOptimum.xs(
        "Electrolyzers_LocTsCcf"
    )

    # --- Expected electricity consumption ---
    # electricity = hydrogen_demand / efficiency

    expected_loc1 = [
        10 / 0.5,  # 20
        10 / 0.6,  # 16.6667
        10 / 0.7,  # 14.2857
        10 / 0.8,  # 12.5
    ]

    expected_loc2 = [
        10 / 1.0,  # 10
        10 / 0.9,  # 11.1111
        10 / 0.8,  # 12.5
        10 / 0.7,  # 14.2857
    ]

    for t in range(4):
        np.testing.assert_almost_equal(op.loc["Loc1", t], expected_loc1[t], decimal=5)
        np.testing.assert_almost_equal(op.loc["Loc2", t], expected_loc2[t], decimal=5)

