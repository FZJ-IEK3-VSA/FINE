import numpy as np
import pandas as pd
from pathlib import Path
import fine as fn
import fine.IOManagement.xarrayIO as xrIO
from fine.utils import ImplementedSolvers


def test_etl_NPV():
    """Test case for basic npv calculation with etl module and Input Output test."""
    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        investmentPeriodInterval=10,
        numberOfInvestmentPeriods=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=10,
            interestRate=0,
            pwlcfParameters={
                "etlParameters": {
                    "initCost": 1,
                    "learningRate": 0.15,
                    "initCapacity": 9.569184,
                    "maxCapacity": 58.52369,
                    "noSegments": 4,
                },
            },
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity_sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series([2190] * 4),
        )
    )

    xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath="test_esM_etl.nc")

    esM.declareOptimizationProblem()

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    commissioning = [
        esM.getOptimizationSummary("SourceSinkModel", ip=ip).loc[
            "PV", "commissioning", "[kW$_{el}$]"
        ]["loc1"]
        for ip in esM.investmentPeriodNames
    ]
    np.testing.assert_almost_equal(commissioning, [1] * 5)

    slope = esM.pwlcfModel.modulesDict["PV"].linEtlParameter.loc[2, "slope"]
    interception = esM.pwlcfModel.modulesDict["PV"].linEtlParameter.loc[
        2, "interception"
    ]
    initCapacity = esM.pwlcfModel.modulesDict["PV"].initCapacity
    initTotalCost = esM.pwlcfModel.modulesDict["PV"].linEtlParameter.loc[0, "totalCost"]

    np.testing.assert_almost_equal(
        esM.pyM.Obj(),
        interception + slope * (initCapacity + sum(commissioning)) - initTotalCost,
    )

    np.testing.assert_almost_equal(esM.pyM.Obj(), 4.6906658)

    # The etl rows are added to the optimization summary after the modeling classes are done
    # (see PwlcfModel.setOptimalValues) and must reach the result export as well; regression
    # guard for issue #735, where the export switched to reading the raw results dict.
    xrds = xrIO.writeEnergySystemModelToDatasets(esM)
    for ipName in esM.investmentPeriodNames:
        optSum = esM.getOptimizationSummary("SourceSinkModel", ip=ipName).loc["PV"]
        ds = xrds["Results"][ipName]["SourceSinkModel"]["PV"]
        for prop in [
            "TAC_ETL",
            "NPVcontribution_ETL",
            "invest_ETL",
            "knowledgeStock_ETL",
            # the base rows carry the etl contribution on top of the component's own costs
            "TAC",
            "NPVcontribution",
            "invest",
        ]:
            assert prop in ds.data_vars
            expected = optSum.loc[prop]
            np.testing.assert_almost_equal(
                float(ds[prop].sel(location="loc1")),
                float(expected.iloc[-1]["loc1"]),
            )
            # the unit is exported as the variable's attribute, as for every summary row
            assert ds[prop].attrs[prop] == expected.index[-1]

    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM_etl.nc")
    Path("test_esM_etl.nc").unlink()

    esm_from_netcdf.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    np.testing.assert_almost_equal(esm_from_netcdf.pyM.Obj(), esM.pyM.Obj(), 5)
    np.testing.assert_almost_equal(esm_from_netcdf.pyM.Obj(), 4.6906658, 5)
    np.testing.assert_almost_equal(
        esM.getOptimizationSummary("SourceSinkModel", ip=2030).loc[
            "PV", "invest", "[1 Euro]"
        ]["loc1"],
        0.902734,
        5,
    )


def test_etl_stock_NPV():
    """Test case for basic npv calculation with etl module when stock is considered."""
    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=8760,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=5,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV_with_etl",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=15,
            interestRate=0,
            pwlcfParameters={
                "etlParameters": {
                    "initCost": 1,
                    "learningRate": 0.18,
                    "initCapacity": 10,
                    "maxCapacity": 50,
                    "noSegments": 4,
                },
            },
            stockCommissioning={
                2010: 1,
                2015: 2,
            },
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity_sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series(4 * 8760),
        )
    )

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    commis = [
        esM.getOptimizationSummary("SourceSinkModel", ip).loc[
            "PV_with_etl", "commissioning", "[kW$_{el}$]"
        ]["loc1"]
        for ip in esM.investmentPeriodNames
    ]
    np.testing.assert_almost_equal(commis, [1, 1, 2, 1, 1])

    interception = esM.pwlcfModel.modulesDict["PV_with_etl"].linEtlParameter.loc[
        2, "interception"
    ]
    slope = esM.pwlcfModel.modulesDict["PV_with_etl"].linEtlParameter.loc[2, "slope"]
    initCapacity = esM.pwlcfModel.modulesDict["PV_with_etl"].initCapacity
    initTotalCost = esM.pwlcfModel.modulesDict["PV_with_etl"].getTotalCostEtl(
        initCapacity
    )
    stockCost2010 = (
        esM.pwlcfModel.modulesDict["PV_with_etl"].getTotalCostEtl(8)
        - esM.pwlcfModel.modulesDict["PV_with_etl"].getTotalCostEtl(7)
    ) / 3
    stockCost2015 = (
        (
            esM.pwlcfModel.modulesDict["PV_with_etl"].getTotalCostEtl(10)
            - esM.pwlcfModel.modulesDict["PV_with_etl"].getTotalCostEtl(8)
        )
        * 2
        / 3
    )

    np.testing.assert_almost_equal(
        esM.pyM.Obj(),
        stockCost2010
        + stockCost2015
        + interception
        + slope * (initCapacity + 4 + (2 / 3) + (1 / 3))
        - initTotalCost,
    )


def test_etl_multi_regional():
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=5,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=10,
            interestRate=0,
            investPerCapacity=10,
            pwlcfParameters={
                "etlParameters": {
                    "initCost": 1,
                    "learningRate": 0.18,
                    "initCapacity": 10,
                    "maxCapacity": 50,
                    "noSegments": 4,
                },
            },
            stockCommissioning={2015: pd.Series([0.1, 0.1], index=["loc1", "loc2"])},
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity_sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                [[2190, 2190 / 2]] * 4, columns=["loc1", "loc2"]
            ),
        )
    )

    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)
    print(1)
