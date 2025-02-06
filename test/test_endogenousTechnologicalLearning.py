import numpy as np
import pandas as pd
from pathlib import Path
import fine as fn
import fine.IOManagement.xarrayIO as xrIO


def test_etl_NPV():
    """
    Test case for basic npv calculation with etl modul and Input Output test.
    """

    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$"
        },
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
            etlParameter={
                "initCost": 1,
                "learningRate": 0.15,
                "initCapacity": 9.569184,
                "maxCapacity": 58.52369,
                "noSegments": 4,
            },
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name='electricity_sink',
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series([2190]*4),
        )
    )

    xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath="test_esM_etl.nc")

    esM.declareOptimizationProblem()
    esM.pyM.write("test.mps")

    esM.optimize(timeSeriesAggregation=False, solver='glpk')

    commissioning = [
        esM.getOptimizationSummary('SourceSinkModel', ip=ip).loc['PV', 'commissioning', '[kW$_{el}$]']['loc1']
        for ip in esM.investmentPeriodNames
    ]
    np.testing.assert_almost_equal(
        commissioning, [1]*5
    )

    slope = esM.etlModel.modulsDict['PV'].linEtlParameter.loc[2, 'slope']
    interception = esM.etlModel.modulsDict['PV'].linEtlParameter.loc[2, 'interception']
    initCapacity = esM.etlModel.modulsDict['PV'].initCapacity
    initTotalCost = esM.etlModel.modulsDict['PV'].linEtlParameter.loc[0, 'totalCost']

    np.testing.assert_almost_equal(
        esM.pyM.Obj(),
        interception + slope * (initCapacity + sum(commissioning)) - initTotalCost
    )

    np.testing.assert_almost_equal(
        esM.pyM.Obj(),
        4.6906658
    )

    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM_etl.nc")
    Path("test_esM_etl.nc").unlink()

    esm_from_netcdf.optimize(timeSeriesAggregation=False, solver='glpk')
    np.testing.assert_almost_equal(
        esm_from_netcdf.pyM.Obj(),
        esM.pyM.Obj(),
        4.6906658
    )

    print(esM.pyM.Obj())


def test_etl_stock_NPV():
    """
    Test case for basic npv calculation with etl modul when stock is considered.
    """

    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$"
        },
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
            etlParameter={
                "initCost": 1,
                "learningRate": 0.18,
                "initCapacity": 10,
                "maxCapacity": 50,
                "noSegments": 4,
            },
            stockCommissioning={
                2010: 1,
                2015: 2,
            }
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name='electricity_sink',
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series(4 * 8760),
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver='glpk')
    commis = [
        esM.getOptimizationSummary('SourceSinkModel', ip).loc['PV_with_etl', 'commissioning', '[kW$_{el}$]']['loc1']
        for ip in esM.investmentPeriodNames
    ]
    np.testing.assert_almost_equal(
        commis,
        [1, 1, 2, 1, 1]
    )

    interception = esM.etlModel.modulsDict['PV_with_etl'].linEtlParameter.loc[2, 'interception']
    slope = esM.etlModel.modulsDict['PV_with_etl'].linEtlParameter.loc[2, 'slope']
    initCapacity = esM.etlModel.modulsDict['PV_with_etl'].initCapacity
    initTotalCost = esM.etlModel.modulsDict['PV_with_etl'].getTotalCost(initCapacity)
    stockCost2010 = (esM.etlModel.modulsDict['PV_with_etl'].getTotalCost(8)
                     - esM.etlModel.modulsDict['PV_with_etl'].getTotalCost(7)) / 3
    stockCost2015 = (esM.etlModel.modulsDict['PV_with_etl'].getTotalCost(10)
                     - esM.etlModel.modulsDict['PV_with_etl'].getTotalCost(8)) * 2 / 3


    np.testing.assert_almost_equal(
        esM.pyM.Obj(),
        stockCost2010 + stockCost2015 + interception + slope * (initCapacity + 4 + (2/3) + (1/3)) - initTotalCost
    )

def test_etl_multi_regional():
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$"
        },
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
            etlParameter={
                "initCost": 1,
                "learningRate": 0.18,
                "initCapacity": 10,
                "maxCapacity": 50,
                "noSegments": 4,
            },
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name='electricity_sink',
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                [[2190] * 2]*4,
                columns=['loc1', 'loc2']
            ),
        )
    )

    esM.optimize(solver='glpk')

