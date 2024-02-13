import fine as fn
import numpy as np
import pytest
import pandas as pd
import math


def test_pathwayBudget():
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity", "methane", "CO2"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "methane": r"kW$_{CH_{4},LHV}$",
            "CO2": r"t$_{CO_2}$/h",
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=5,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=None,
        pathwayBalanceLimit=pd.DataFrame(
            index=["CO2 limit"],
            columns=["PerfectLand", "lowerBound"],
            data=[[-1000, True]],
        ),
    )

    # 1.1. pv source
    PvOperationRateMax = pd.DataFrame(columns=["PerfectLand"], data=[1, 1])

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PvOperationRateMax,
            investPerCapacity=1000000,  # dummy values to make gas plant cheaper
            interestRate=0.02,
            opexPerOperation=1000000,  # dummy values to make gas plant cheaper
            economicLifetime=25,
        )
    )

    # 1.2. sink
    demand = {}
    demand[2020] = pd.DataFrame(
        columns=["PerfectLand"],
        data=[
            50000,
            100000,
        ],
    )
    demand[2025] = pd.DataFrame(
        columns=["PerfectLand"],
        data=[
            100000,
            50000,
        ],
    )
    demand[2030] = demand[2025] * 1
    demand[2035] = demand[2030] * 1.5
    demand[2040] = demand[2030] * 2

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # 1.3 conversion technologies with stock and emissions
    esM.add(
        fn.Conversion(
            esM=esM,
            name="CCGT plants (methane)",
            physicalUnit=r"kW$_{CH_{4},LHV}$",
            commodityConversionFactors={
                "electricity": 1,
                "methane": -1 / 0.625,
                "CO2": 201 * 1e-6 / 0.625,
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0.08,
            economicLifetime=35,
        )
    )
    # methane source
    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=True,
            commodityCost=0.1,
        )
    )
    # CO2 sink
    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 for balance",
            commodity="CO2",
            hasCapacityVariable=False,
            pathwayBalanceLimitID="CO2 limit",
        )
    )
    # 2. optimize
    esM.optimize(solver="glpk")

    # 3. test
    # Without a budget limit for CO2, the cost optimal system would only build
    # gas plants, as it is for free and PV is expensive.
    # gas plant capacity in 2020: 22.83
    # CO2 operation in 2020: 48.24
    # CO2 operation in 2025: 48.239999999999995
    # CO2 operation in 2030: 48.239999999999995
    # CO2 operation in 2035: 72.36
    # CO2 operation in 2040: 96.47999999999999
    # CO2 over the pathway: (15*48.24)+(5*72.36)+(5*96.48) = 1567.8

    # with limitation, it must be below 1000
    co2_emissions = (
        esM.getOptimizationSummary("SourceSinkModel", ip=2020)
        + esM.getOptimizationSummary("SourceSinkModel", ip=2025)
        + esM.getOptimizationSummary("SourceSinkModel", ip=2030)
        + esM.getOptimizationSummary("SourceSinkModel", ip=2035)
        + esM.getOptimizationSummary("SourceSinkModel", ip=2040)
    ).loc["CO2 for balance", "operation", "[t$_{CO_2}$/h*h]"]["PerfectLand"]
    assert co2_emissions * 5 < 1000

    # With limitation the expensive PV must be installed
    installed_cap_PV_2045 = esM.getOptimizationSummary("SourceSinkModel", ip=2040).loc[
        "PV", "capacity", "[kW$_{el}$]"
    ]["PerfectLand"]
    np.testing.assert_almost_equal(installed_cap_PV_2045, 45.662100456621)
