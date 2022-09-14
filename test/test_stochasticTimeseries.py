#!/usr/bin/env python
# coding: utf-8

# # Test case for perfect foresight approach

# Status: First step towards perfect foresight
# -> one design of energy system
# -> several operation years
# robustness optimizations possible 

import FINE as fn
import numpy as np
import pandas as pd


def test_stochasticTimeSeries():
    
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    numberOfInvestmentPeriods = 2  
    yearsPerInvestmentPeriod = 1

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        mode="stochastic",
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        yearsPerInvestmentPeriod=yearsPerInvestmentPeriod,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    # Sources
    # Electricity market
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
                    0.5e3,
                    0.5e3,
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
            commodityRevenueTimeSeries=revenues,
        )
    )  # eur/kWh

    # Photovoltaic
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
    ).T 

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
    # Optimize energy system model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    print("Objective value:")
    print(esM.pyM.Obj())
    np.testing.assert_almost_equal(
        esM.pyM.Obj(), 7135
    )  # capacity costs only taken for one year
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
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    numberOfInvestmentPeriods = 2  
    yearsPerInvestmentPeriod = 1

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand","PerfectLand2"},
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        mode="stochastic",
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        yearsPerInvestmentPeriod=yearsPerInvestmentPeriod,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    # Sources
    # Electricity market
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
    ).T
    maxpurchase[1] = pd.DataFrame(
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
        index=["PerfectLand","PerfectLand2"],
    ).T

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

    # Photovoltaic
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
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
        index=["PerfectLand","PerfectLand2"],
    ).T

    demand = {}
    demand[0]=pd.DataFrame(columns=["PerfectLand","PerfectLand2"])
    demand[0].loc[0,"PerfectLand2"]=0
    demand[0].loc[1,"PerfectLand2"]=2e3
    demand[0].loc[2,"PerfectLand2"]=2e3
    demand[0].loc[3,"PerfectLand2"]=2e3
    demand[0].loc[0,"PerfectLand"]=2e3
    demand[0].loc[1,"PerfectLand"]=1e3
    demand[0].loc[2,"PerfectLand"]=1e3
    demand[0].loc[3,"PerfectLand"]=1e3
    demand[1]=pd.DataFrame(columns=["PerfectLand","PerfectLand2"])
    demand[1].loc[0,"PerfectLand2"]=0
    demand[1].loc[1,"PerfectLand2"]=2e3
    demand[1].loc[2,"PerfectLand2"]=2e3
    demand[1].loc[3,"PerfectLand2"]=2e3
    demand[1].loc[0,"PerfectLand"]=2e3
    demand[1].loc[1,"PerfectLand"]=1e3
    demand[1].loc[2,"PerfectLand"]=1e3
    demand[1].loc[3,"PerfectLand"]=1e3
    
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
    esM.getOptimizationSummary("SourceSinkModel", outputLevel=1)
    esM.getOptimizationSummary("TransmissionModel", outputLevel=1)
    esM.getOptimizationSummary("StorageModel", outputLevel=1)
    
    # Optimize energy system model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    print("Objective value:")
    print(esM.pyM.Obj())
    np.testing.assert_almost_equal(
        esM.pyM.Obj(), 7135
    )  # capacity costs only taken for one year
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

if __name__ == "__main__":
    test_stochasticTimeSeries()
