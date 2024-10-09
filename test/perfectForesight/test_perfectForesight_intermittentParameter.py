import pandas as pd
import pytest

import fine as fn

def create_esm(balanceLimit=False):

    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity", 'hydrogen'},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_2}$"
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        investmentPeriodInterval=10,
        numberOfInvestmentPeriods=2,
        startYear=2020,
        balanceLimit=None if balanceLimit is False else {
            2020: None,
            2030: pd.DataFrame(index=['elec'], columns=['Total', 'lowerBound'], data=[[1000, True]]),
        }
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="electricity expensive",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=100,
            balanceLimitID=None if balanceLimit is False else 'elec',
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(columns=['loc1'], data=[438000, 438000]),
        )
    )
    return esM

def test_capacityCommissioningMinMaxFix():
    esM = create_esm()

    esM.add(
        fn.Source(
            esM=esM,
            name="electricity cheap",
            commodity="electricity",
            hasCapacityVariable=True,
            opexPerOperation=0.1,
            capacityMax={2020: None, 2030: 50},
            interestRate=0,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=50000,
            commissioningFix={2020: 10, 2030: None},
            interestRate=0,
            commissioningMax={2020: None, 2030: 25}
        )
    )

    assert esM.getComponent('electricity cheap').processedCapacityMax[0] is None

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    ops = {ip: esM.getOptimizationSummary('SourceSinkModel', ip=ip) for ip in esM.investmentPeriodNames}

    assert ops[2020].loc['PV', 'capacity', '[kW$_{el}$]'].values[0] == 10
    assert ops[2030].loc['electricity cheap', 'capacity', '[kW$_{el}$]'].values[0] == 50
    assert ops[2030].loc['electricity expensive', 'operation', '[kW$_{el}$*h]'].values[0] == 25*2*4380

def test_fullLoadHoursMinMax():
    esM = create_esm()

    esM.add(
        fn.Source(
            esM=esM,
            name="electricity cheap",
            commodity="electricity",
            hasCapacityVariable=True,
            opexPerOperation=0.1,
            capacityMax={2020: None, 2030: 50},
            interestRate=0,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="electrolyzer",
            commodityConversionFactors={'electricity': -1, 'hydrogen': 1},
            hasCapacityVariable=True,
            physicalUnit=r"kW$_{el}$",
            investPerCapacity=50000,
            opexPerOperation={2020: 1001, 2030: 0},
            capacityFix=1,
            yearlyFullLoadHoursMax={2020: None, 2030: 2000},
            yearlyFullLoadHoursMin={2020: 1000, 2030: None},
            interestRate=0,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="wind",
            commodity="electricity",
            hasCapacityVariable=True,
            opexPerOperation=1,
            commissioningFix=10,
            yearlyFullLoadHoursMin={2020: 1000, 2030: None},
            interestRate=0,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="hydrogen export",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateMax={2020: None, 2030: pd.DataFrame(data=[4380, 4380], columns=['loc1'])},
            commodityRevenue=1000
        )
    )

    with pytest.raises(TypeError, match=r".* can not be None for individual investment periods.*"):
        esM.add(
            fn.Sink(
                esM=esM,
                name="hydrogen export",
                commodity="hydrogen",
                hasCapacityVariable=False,
                commodityRevenueTimeSeries={2020: None, 2030: pd.Series(data=[1, 1])},
            )
        )

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    ops_srcSnk = {
        ip: esM.getOptimizationSummary('SourceSinkModel', ip=ip)
        for ip in esM.investmentPeriodNames
    }
    ops_conv = {
        ip: esM.getOptimizationSummary('ConversionModel', ip=ip)
        for ip in esM.investmentPeriodNames
    }

    assert ops_conv[2020].loc['electrolyzer', 'operation', '[kW$_{el}$*h]'].values[0] == 1000
    assert ops_conv[2030].loc['electrolyzer', 'operation', '[kW$_{el}$*h]'].values[0] == 2000
    assert ops_srcSnk[2020].loc['wind', 'operation', '[kW$_{el}$*h]'].values[0] == 10000

def test_storageAndBalanceLimit():
    esM = create_esm(balanceLimit=True)

    esM.add(
        fn.Source(
            esM=esM,
            name="electricity cheap",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerOperation=0.1,
            operationRateMax={2020: pd.DataFrame(columns=['loc1'], data=[438000, 0]), 2030: None},
            interestRate=0,
        )
    )

    esM.add(
        fn.Storage(
            esM=esM,
            name="storage",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerChargeOperation=0.01,
            opexPerDischargeOperation=0.01,
            chargeOpRateFix={2020: pd.DataFrame(columns=['loc1'], data=[438000 / 2, 10]), 2030: None},
            dischargeOpRateMax={2020: pd.DataFrame(columns=['loc1'], data=[438000, 438000 / 2]), 2030: None},
            interestRate=0,
        )
    )
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    assert esM.pyM.chargeOp_stor.get_values()[('loc1', 'storage', 0, 0, 0)] == 438000 / 2
    assert esM.pyM.chargeOp_stor.get_values()[('loc1', 'storage', 0, 0, 1)] == 10
    assert esM.pyM.dischargeOp_stor.get_values()[('loc1', 'storage', 0, 0, 0)] == 10
    assert esM.pyM.dischargeOp_stor.get_values()[('loc1', 'storage', 0, 0, 1)] == 438000 / 2
    assert esM.getOptimizationSummary(
        'SourceSinkModel',
        ip= 2030
    ).loc['electricity expensive', 'operation', '[kW$_{el}$*h]'].values[0] == 1000
