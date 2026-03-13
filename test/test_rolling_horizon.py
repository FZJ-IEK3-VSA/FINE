import fine as fn
import numpy as np
import pandas as pd


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
        verboseLogLevel=2,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Source_cheap_then_expensive",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e3,
            interestRate=0.02,
            opexPerOperation={2020: 1, 2025: 1, 2030: 100, 2035: 100},
            economicLifetime=15,
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
            opexPerOperation={2020: 100, 2025: 100, 2030: 1, 2035: 1},
            economicLifetime=15,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                np.array([4380, 4380]), columns=["PerfectLand"], index=[0, 1]
            ),
        )
    )

    # fn.expansionModules.rollingHorizon.rollingHorizonOptimization(
    #     esM=esM,
    #     scenario_name="test_scenario",
    #     resultExportPath=r"C:\Users\j.behrens\work\fine\test\rolling_horizon_test_export",
    #     numberOfInvestmentPeriodsForRollingHorizon=2,
    #     timeSeriesAggregation=False,
    # )
