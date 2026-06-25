import fine as fn
from fine.expansionModules.rollingHorizon import rollingHorizonOptimization
import numpy as np
import pandas as pd
from pathlib import Path


def test_rolling_horizon():
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=7860,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=4,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Source_cheap_then_expensive",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e3,
            interestRate=0.02,
            opexPerOperation={2020: 1, 2025: 1, 2030: 1, 2035: 100},
            economicLifetime=15,
            technicalLifetime=15
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Source_expensive_then_cheap",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e3,
            interestRate=0.02,
            opexPerOperation={2020: 100, 2025: 100, 2030: 100, 2035: 1},
            economicLifetime=15,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix={
                2020: pd.DataFrame(np.array([2190, 2190]), columns=["PerfectLand"], index=[0, 1]),
                2025: pd.DataFrame(np.array([4380, 4380]), columns=["PerfectLand"], index=[0, 1]),
                2030: pd.DataFrame(np.array([6570, 6570]), columns=["PerfectLand"], index=[0, 1]),
                2035: pd.DataFrame(np.array([8760, 8760]), columns=["PerfectLand"], index=[0, 1])
                },
        )
    )


    results = rollingHorizonOptimization(
        esM=esM,
        resultExportPath=Path(__file__).resolve().parent,
        scenario_name="test",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        numberOfTimeStepsPerPeriod=1,
        numberOfSegments=1,
        numberOfTypicalPeriods=1,
    )

    # check that commissioning of first year is in stock of second year
    assert (
        results[2020]
        .getOptimizationSummary("SourceSinkModel", ip=2020)
        .loc["Source_cheap_then_expensive", "commissioning"].iloc[0, 0]
        == results[2025].getComponent("Source_cheap_then_expensive").stockCommissioning[2020]["PerfectLand"]
    )

    commis_2020 = results[2020].getOptimizationSummary("SourceSinkModel", ip=2020).loc["Source_cheap_then_expensive", "commissioning"].iloc[0, 0]
    commis_2025 = results[2025].getOptimizationSummary("SourceSinkModel", ip=2025).loc["Source_cheap_then_expensive", "commissioning"].iloc[0, 0]
    # stock_2030_from_2020 = results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning[2020]["PerfectLand"]
    # stock_2030_from_2025 = results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning[2025]["PerfectLand"]

    stockCommis_2030 = results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning

    print(f"Commissioning in 2020: {commis_2020}")
    print(f"Commissioning in 2025: {commis_2025}")
    print(f"Stock Commissioning in 2030: {stockCommis_2030}")
    # assert (
    #     commis_2020 + commis_2025 == stock_2030_from_2020 + stock_2030_from_2025
    # )

    # delete created excel lists
    for year in [2020, 2025, 2030, 2035]:
        path = Path(__file__).resolve().parent
        # (path / f"test_rollingHorizon_{year}.xlsx").unlink()

test_rolling_horizon()
