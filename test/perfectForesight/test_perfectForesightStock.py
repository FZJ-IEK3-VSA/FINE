import FINE as fn
import numpy as np
import pandas as pd
import pytest


def test_Stock_wrongStockYears():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        startYear=2020,
        numberOfInvestmentPeriods=6,
        yearsPerInvestmentPeriod=1,
        mode="perfectForesight",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    with pytest.raises(ValueError, match=r".*stockCommissioning should be initialized.*"):
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=4e6,
            investPerCapacity=2 * 2190,
            opexPerCapacity=0,
            interestRate=0.02,
            opexPerOperation= 0.01,
            economicLifetime=5,
            technicalLifetime=6,
            stockCommissioning={2015:2,2017:0,2018:5}
        )


def test_perfectForesightStock():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    investmentPeriodList_for_testing=[2020,2021,2022,2023,2024,2025]

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        startYear=2020,
        numberOfInvestmentPeriods=6,
        yearsPerInvestmentPeriod=1,
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
    costs[2020] = pd.DataFrame(
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
    costs[2021] = pd.DataFrame(
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
    costs[2022]=costs[2021]
    costs[2023]=costs[2021]
    costs[2024]=costs[2021]
    costs[2025]=costs[2021]

    revenues = {}
    revenues[2020] = pd.DataFrame(
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
    revenues[2021] = pd.DataFrame(
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
    revenues[2022]=revenues[2021]
    revenues[2023]=revenues[2021]
    revenues[2024]=revenues[2021]
    revenues[2025]=revenues[2021]

    maxpurchase = {}
    maxpurchase[2020] = pd.DataFrame(
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
    maxpurchase[2021] = pd.DataFrame(
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
    maxpurchase[2022]=maxpurchase[2021]
    maxpurchase[2023]=maxpurchase[2021]
    maxpurchase[2024]=maxpurchase[2021]
    maxpurchase[2025]=maxpurchase[2021]

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
    PVoperationRateMax[2020] = pd.DataFrame(
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
    PVoperationRateMax[2021] = pd.DataFrame(
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
    PVoperationRateMax[2022]=PVoperationRateMax[2021]
    PVoperationRateMax[2023]=PVoperationRateMax[2021]
    PVoperationRateMax[2024]=PVoperationRateMax[2021]
    PVoperationRateMax[2025]=PVoperationRateMax[2021]

    
    
    # different opexPerOperation per investmentperiod
    PVopexPerOperation = {}
    PVopexPerOperation[2020] = 0.01
    PVopexPerOperation[2021] = 0.02
    PVopexPerOperation[2022]=PVopexPerOperation[2021]
    PVopexPerOperation[2023]=PVopexPerOperation[2021]
    PVopexPerOperation[2024]=PVopexPerOperation[2021]
    PVopexPerOperation[2025]=PVopexPerOperation[2021]

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
            interestRate=0.02,
            opexPerOperation=PVopexPerOperation,  # 0.01,
            economicLifetime=5,
            technicalLifetime=6,
            stockCommissioning={2018:1,2019:0.5}
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
    revenuesDemand[2020] = pd.DataFrame(
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
    revenuesDemand[2021] = pd.DataFrame(
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
    revenuesDemand[2022]=revenuesDemand[2021]
    revenuesDemand[2023]=revenuesDemand[2021]
    revenuesDemand[2024]=revenuesDemand[2021]
    revenuesDemand[2025]=revenuesDemand[2021]

    demand = {}
    demand[2020] = pd.DataFrame(
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
    demand[2021] = pd.DataFrame(
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
    demand[2022]=demand[2021]
    demand[2023]=demand[2021]
    demand[2024]=demand[2021]
    demand[2025]=demand[2021]

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
    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    print("Objective value:")
    print(esM.pyM.Obj())    # 44655?
    
    # check 
    assert esM.getOptimizationSummary("SourceSinkModel").keys() != investmentPeriodList_for_testing
    
    # check 
    PV_cap_year0=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[2020].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year0 ,1.71232876712329 )
    PV_cap_year1=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[2021].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year1 ,1.71232876712329 )
    PV_cap_year5=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[2025].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year5 , 1.1415525114155252)
    
    raise ValueError("Currently we dont know the correct results for linked ip's")
    np.testing.assert_almost_equal(esM.pyM.Obj(),11545)

    # Check capacity results:
    # year 0
    esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs("PV").values[0]
    np.testing.assert_almost_equal(esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs("PV").values[0]  ,1.71232876712329 )
    
    # year 1
    esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[2021].xs("PV").values[0]
    np.testing.assert_almost_equal(esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[2021].xs("PV").values[0]  ,0 )


if __name__ == "__main__":
    test_perfectForesightStock()
    