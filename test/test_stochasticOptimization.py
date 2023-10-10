import FINE as fn
import numpy as np
import pandas as pd


def stochasticESM(singleYear=False, sameParameters=False, transmissionCase=False):
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    if singleYear is True:
        numberOfInvestmentPeriods = 1
        stochasticModel = False
        investmentPeriodInterval = 1
    else:
        numberOfInvestmentPeriods = 2
        stochasticModel = True
        investmentPeriodInterval = 1

    if transmissionCase:
        locations = {"PerfectLand", "PerfectLand2"}
        locIndex = ["PerfectLand", "PerfectLand2"]
    else:
        locations = {"PerfectLand"}
        locIndex = ["PerfectLand"]

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        stochasticModel=stochasticModel,
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        investmentPeriodInterval=investmentPeriodInterval,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    # Sources
    # Electricity market
    yearlyCostsElec = pd.DataFrame(
        columns=locIndex, index=[x for x in range(0, numberOfTimeSteps)], data=1
    )
    if singleYear:
        costs = yearlyCostsElec
    else:
        costs = {}
        costs[0] = yearlyCostsElec
        costs[1] = yearlyCostsElec

    yearlyRevenueElec = pd.DataFrame(
        columns=locIndex, index=[x for x in range(0, numberOfTimeSteps)], data=0
    )
    if singleYear:
        revenue = yearlyRevenueElec
    else:
        revenue = {}
        revenue[0] = yearlyRevenueElec
        revenue[1] = yearlyRevenueElec

    yearlyMaxPurchaseElec = pd.DataFrame(
        columns=locIndex, index=[x for x in range(0, numberOfTimeSteps)]
    )
    for col in yearlyMaxPurchaseElec:  # so it works for one or several locations
        yearlyMaxPurchaseElec[col] = [
            0.5e3,
            0.5e3,
            4e3,
            4e3,
        ]
    if singleYear:
        maxpurchase = yearlyMaxPurchaseElec
    else:
        maxpurchase = {}
        maxpurchase[0] = yearlyMaxPurchaseElec
        maxpurchase[1] = yearlyMaxPurchaseElec

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=maxpurchase,
            commodityCostTimeSeries=costs,
            commodityRevenueTimeSeries=revenue,
        )
    )  # eur/kWh

    # Photovoltaic
    yearlyPVoperationRateMax = pd.DataFrame(
        columns=locIndex, index=[x for x in range(0, numberOfTimeSteps)]
    )
    for (
        col
    ) in yearlyPVoperationRateMax.columns:  # so it works for one or several locations
        yearlyPVoperationRateMax[col] = [0.4, 0.4, 0.6, 0.6]
    if singleYear:
        PVoperationRateMax = yearlyPVoperationRateMax
    else:
        PVoperationRateMax = {}
        PVoperationRateMax[0] = yearlyPVoperationRateMax
        PVoperationRateMax[1] = yearlyPVoperationRateMax

    # different opexPerOperation per investmentperiod
    if singleYear:
        PVopexPerOperation = 0.01
    elif not singleYear and sameParameters:
        PVopexPerOperation = {}
        PVopexPerOperation[0] = 0.01
        PVopexPerOperation[1] = 0.01
    else:
        PVopexPerOperation = {}
        PVopexPerOperation[0] = 0.01
        PVopexPerOperation[1] = 0.02

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            capacityMax=4e6,
            investPerCapacity=2 * 2190,
            opexPerCapacity=0,
            interestRate=0,
            opexPerOperation=PVopexPerOperation,
            economicLifetime=1,
        )
    )

    # Sinks
    ### Industry site
    yearlyRevenuesDemand = pd.DataFrame(
        columns=locIndex, index=[x for x in range(0, numberOfTimeSteps)], data=0.1
    )
    if singleYear:
        revenuesDemand = yearlyRevenuesDemand
    else:
        revenuesDemand = {}
        revenuesDemand[0] = yearlyRevenuesDemand
        revenuesDemand[1] = yearlyRevenuesDemand

    if not transmissionCase:
        yearlyDemand = pd.DataFrame(columns=locIndex, data=[2e3, 1e3, 1e3, 1e3])
        if singleYear:
            demand = yearlyDemand
        else:
            demand = {}
            demand[0] = yearlyDemand
            demand[1] = yearlyDemand

    else:
        # increase revenue for the tranmission case
        revenuesDemand[1] = 2 * revenuesDemand[1]

        # different demand
        yearlyDemand = pd.DataFrame(columns=["PerfectLand", "PerfectLand2"])
        yearlyDemand["PerfectLand"] = [2e3, 1e3, 1e3, 1e3]
        yearlyDemand["PerfectLand2"] = [0, 2e3, 2e3, 2e3]

        demand = {}
        demand[0] = yearlyDemand
        demand[1] = yearlyDemand

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
            commodityRevenueTimeSeries=revenuesDemand,
        )
    )

    # add transmission for the transmission test
    if transmissionCase:
        esM.add(
            fn.Transmission(
                esM=esM,
                name="Transmission",
                commodity="electricity",
                investPerCapacity=0.177,
                losses=0.1e-2,
                hasCapacityVariable=True,
                hasIsBuiltBinaryVariable=True,
                bigM=100,
                capacityFix=1,
            )
        )
    return esM


def test_stochasticBasic():
    singleYearesM = stochasticESM(singleYear=True)
    singleYearesM.optimize(solver="glpk")

    doubleYear = stochasticESM(singleYear=False, sameParameters=True)
    doubleYear.optimize(solver="glpk")

    # check objective values
    np.testing.assert_almost_equal(singleYearesM.pyM.Obj(), 7545)
    # stochasic with two periods with same params should have 2* objective
    # value of single year
    assert 2 * (singleYearesM.pyM.Obj()) == doubleYear.pyM.Obj()

    # check commissioning and capacity variables
    cms_vars = doubleYear.pyM.commis_srcSnk.get_values()
    cap_var = doubleYear.pyM.cap_srcSnk.get_values()
    singleYear_cms_vars = singleYearesM.pyM.commis_srcSnk.get_values()
    singleYear_cap_vars = singleYearesM.pyM.cap_srcSnk.get_values()
    assert cms_vars[("PerfectLand", "PV", 0)] == cms_vars[("PerfectLand", "PV", 1)]
    assert cap_var[("PerfectLand", "PV", 0)] == cap_var[("PerfectLand", "PV", 1)]
    assert (
        cms_vars[("PerfectLand", "PV", 0)]
        == singleYear_cms_vars[("PerfectLand", "PV", 0)]
    )
    assert (
        cap_var[("PerfectLand", "PV", 0)]
        == singleYear_cap_vars[("PerfectLand", "PV", 0)]
    )


def test_stochasticParameters():
    esM = stochasticESM(singleYear=False, sameParameters=False)
    esM.optimize(solver="glpk")

    # check objective value
    np.testing.assert_almost_equal(esM.pyM.Obj(), 15135)

    # check that commissioning and capacitiy are same between the years
    cms_vars = esM.pyM.commis_srcSnk.get_values()
    assert cms_vars[("PerfectLand", "PV", 0)] == cms_vars[("PerfectLand", "PV", 1)]
    cap_var = esM.pyM.cap_srcSnk.get_values()
    assert cap_var[("PerfectLand", "PV", 0)] == cap_var[("PerfectLand", "PV", 1)]

    # check results
    print("Electricity Market:")
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("Electricity market")
        .values[0]
    ) == [500, 0, 0, 0]

    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("Electricity market")
        .values[0]
    ) == [500, 0, 0, 0]

    print("Photovoltaic:")
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("PV")
        .values[0]
    ) == [1500, 1000, 1000, 1000]
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("PV")
        .values[0]
    ) == [1500, 1000, 1000, 1000]

    print("Demand:")
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("EDemand")
        .values[0]
    ) == [2000, 1000, 1000, 1000]
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("EDemand")
        .values[0]
    ) == [2000, 1000, 1000, 1000]


def test_stochasticTimeSeries_withTransmission():
    esM = stochasticESM(transmissionCase=True)
    # Optimize energy system model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    #
    cms_vars = esM.pyM.commis_srcSnk.get_values()
    assert cms_vars[("PerfectLand", "PV", 0)] == cms_vars[("PerfectLand", "PV", 1)]
    assert cms_vars[("PerfectLand2", "PV", 0)] == cms_vars[("PerfectLand2", "PV", 1)]
    cap_var = esM.pyM.cap_srcSnk.get_values()
    assert cap_var[("PerfectLand", "PV", 0)] == cap_var[("PerfectLand", "PV", 1)]
    assert cap_var[("PerfectLand2", "PV", 0)] == cap_var[("PerfectLand2", "PV", 1)]

    # test demand
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("EDemand")
        .loc["PerfectLand"]
        .values
    ) == [2000, 1000, 1000, 1000]
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("EDemand")
        .loc["PerfectLand2"]
        .values
    ) == [0, 2000, 2000, 2000]

    # test objective value
    np.testing.assert_almost_equal(esM.pyM.Obj(), 19003.565679567248)
