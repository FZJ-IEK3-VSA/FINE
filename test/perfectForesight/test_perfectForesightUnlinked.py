#!/usr/bin/env python
# coding: utf-8

# # Test case for perfect foresight approach

# Status: Second step towards perfect foresight
# -> two designs of energy system
# -> several operation years


import FINE as fn
import numpy as np
import pandas as pd


<<<<<<< HEAD:test/test_stochasticTimeseries.py
def stochastical_optimization_model():
=======
def test_perfectForesight_unlinked():
>>>>>>> origin/issue115_pf_capacities_stock:test/perfectForesight/test_perfectForesightUnlinked.py
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    numberOfInvestmentPeriods = 2  # new test, before =1
    yearsPerInvestmentPeriod = 1

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        yearsPerInvestmentPeriod=yearsPerInvestmentPeriod,
        mode="perfectForesight",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    # Sources

    # Electricity market
    # for one investmentperiod:

    # costs = pd.DataFrame([np.array([ 1,1,1,1,])],
    #                    index = ['PerfectLand']).T
    # revenues = pd.DataFrame([np.array([ 0., 0., 0., 0.,])],
    #                       index = ['PerfectLand']).T
    # maxpurchase = pd.DataFrame([np.array([0.5e3, 0.5e3, 4e3, 4e3,])],
    #                    index = ['PerfectLand']).T

    # for two investmentperiods
    costs = {}
    costs[0] = pd.DataFrame(
        [
            np.array(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T
    costs[1] = pd.DataFrame(
        [
            np.array(
                [
                    1,
                    1,
                    1,
                    1,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T

    revenues = {}
    revenues[0] = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T
    revenues[1] = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T

    maxpurchase = {}
    maxpurchase[0] = pd.DataFrame(
        [
            np.array(
                [
                    0.5e3,
                    0.5e3,
                    4e3,
                    4e3,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T
    maxpurchase[1] = pd.DataFrame(
        [
            np.array(
                [
                    4e3,
                    4e3,
                    4e3,
                    4e3,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T

    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=maxpurchase,
            commodityCostTimeSeries=costs,
            # commodityCost= 1,
            commodityRevenueTimeSeries=revenues,
        )
    )  # eur/kWh

    # Photovoltaic
    # single investment period
    # PVoperationRateMax = pd.DataFrame([np.array([0.4, 0.4, 0.6, 0.6,])],
    #                        index = ['PerfectLand']).T

    # 2 investmentperiods
    PVoperationRateMax = {}
    PVoperationRateMax[0] = pd.DataFrame(
        [
            np.array(
                [
                    0.4,
                    0.4,
                    0.6,
                    0.6,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T
    PVoperationRateMax[1] = pd.DataFrame(
        [
            np.array(
                [
                    0.4,
                    0.4,
                    0.6,
                    0.6,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T
    # different opexPerOperation per investmentperiod
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
            opexPerOperation=PVopexPerOperation,  # 0.01,
            economicLifetime=1,
        )
    )

    # Sinks

    ### Industry site
    # for one ip:
    # demand = pd.DataFrame([np.array([2/5, 1/5, 1/5, 1/5,])],
    #                 index = ['PerfectLand']).T
    # demand now as dict:
    # two investmentperiods

    revenuesDemand = {}
    revenuesDemand[0] = pd.DataFrame(
        [
            np.array(
                [
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T
    revenuesDemand[1] = pd.DataFrame(
        [
            np.array(
                [
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T

    demand = {}
    demand[0] = pd.DataFrame(
        [
            np.array(
                [
                    2e3,
                    1e3,
                    1e3,
                    1e3,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T  # first investmentperiod
    demand[1] = pd.DataFrame(
        [
            np.array(
                [
                    2e3,
                    1e3,
                    1e3,
                    1e3,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T  # second investmentperiod

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
    return esM


def test_stochasticalOpt_withSegmentation():
    # values not logical, just check of the workflow
    esM = stochastical_optimization_model()
    # Optimize energy system model
    esM.aggregateTemporally(
        numberOfTypicalPeriods=1,
        numberOfTimeStepsPerPeriod=4,
        storeTSAinstance=False,
        segmentation=True,
        numberOfSegmentsPerPeriod=1,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        sortValues=False,
        rescaleClusterPeriods=False,
    )
    esM.optimize(timeSeriesAggregation=True, solver="glpk")
    print("Objective value:")
    print(esM.pyM.Obj())
    np.testing.assert_almost_equal(
        esM.pyM.Obj(), 3650
    )  # capacity costs only taken for one year
    print("Electricity Market:")
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("Electricity market")
        .values[0]
    ) == [0, 0, 0, 0]

    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("Electricity market")
        .values[0]
    ) == [0, 0, 0, 0]

    print("Photovoltaic:")
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("PV")
        .values[0]
    ) == [1250, 1250, 1250, 1250]
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("PV")
        .values[0]
    ) == [1250, 1250, 1250, 1250]

    print("Demand:")
    # print(esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('EDemand')) ### Thomas and Stefan:  [2000,1000,1000,1000] correct values
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[0]
        .xs("EDemand")
        .values[0]
    ) == [1250, 1250, 1250, 1250]
    assert list(
        esM.componentModelingDict["SourceSinkModel"]
        .operationVariablesOptimum[1]
        .xs("EDemand")
        .values[0]
    ) == [1250, 1250, 1250, 1250]


def test_stochasticalOpt_withoutSegmentation():
    esM = stochastical_optimization_model()
    # Optimize energy system model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    print("Objective value:")
    print(esM.pyM.Obj())  # year 0: 7545 year 1: 4000  -> Ziel: 11545
    np.testing.assert_almost_equal(esM.pyM.Obj(), 11545)

    # Check capacity results:
    # year 0
    esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs(
        "PV"
    ).values[0]
    np.testing.assert_almost_equal(
        esM.componentModelingDict["SourceSinkModel"]
        .capacityVariablesOptimum[0]
        .xs("PV")
        .values[0],
        1.71232876712329,
    )

    # year 1
    esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[1].xs(
        "PV"
    ).values[0]
    np.testing.assert_almost_equal(
        esM.componentModelingDict["SourceSinkModel"]
        .capacityVariablesOptimum[1]
        .xs("PV")
        .values[0],
        0,
    )


if __name__ == "__main__":
    test_stochasticalOpt_withSegmentation()
    test_stochasticalOpt_withoutSegmentation()
