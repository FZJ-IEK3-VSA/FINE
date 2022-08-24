#!/usr/bin/env python
# coding: utf-8

# # Test case for perfect foresight approach

# Status: Second step towards perfect foresight
# -> two designs of energy system
# -> several operation years


import FINE as fn
import numpy as np
import pandas as pd


def test_perfectForesight_linked_costs():
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
        numberOfInvestmentPeriods=6,
        yearsPerInvestmentPeriod=1,
        mode="perfectForesight",
        lengthUnit="km",
        verboseLogLevel=2,
    )

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
    PVoperationRateMax[1]=PVoperationRateMax[0]
    PVoperationRateMax[2]=PVoperationRateMax[0]
    PVoperationRateMax[3]=PVoperationRateMax[0]
    PVoperationRateMax[4]=PVoperationRateMax[0]
    PVoperationRateMax[5]=PVoperationRateMax[0]

    
    
    # different opexPerOperation per investmentperiod
    PVopexPerOperation = {}
    PVopexPerOperation[0] = 0.01
    PVopexPerOperation[1] = 0.01
    PVopexPerOperation[2]=PVopexPerOperation[1]
    PVopexPerOperation[3]=PVopexPerOperation[1]
    PVopexPerOperation[4]=PVopexPerOperation[1]
    PVopexPerOperation[5]=PVopexPerOperation[1]

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            # capacityMax=4e6,
            investPerCapacity=1,
            opexPerCapacity=0,
            interestRate=0.02,  # formerly 0
            opexPerOperation=0, #0.01, #PVopexPerOperation,  # 0.01,
            economicLifetime=6,
            technicalLifetime=6
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
    revenuesDemand[2]=revenuesDemand[1]
    revenuesDemand[3]=revenuesDemand[1]
    revenuesDemand[4]=revenuesDemand[1]
    revenuesDemand[5]=revenuesDemand[1]

    demand = {}
    demand[0] = pd.DataFrame(
        [
            np.array(
                [
                    2190,
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
                    1095,
                    1e3,
                    1e3,
                    1e3,
                ]
            )
        ],
        index=["PerfectLand"],
    ).T  # second investmentperiod
    demand[2]=demand[1]
    demand[3]=demand[1]
    demand[4]=demand[1]
    demand[5]=demand[1]

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
            # commodityRevenueTimeSeries=revenuesDemand,
        )
    )

    # Optimize energy system model
    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    print("Objective value:")
    print(esM.pyM.Obj()) 
    
    # 
    PV_cap_year0=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year0 ,1.71232876712329 )
    PV_cap_year1=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[1].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year1 ,1.71232876712329 )
    PV_cap_year5=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[5].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year5 , 1.1415525114155252)
    
    raise ValueError("Currently we dont know the correct results for linked ip's")
    np.testing.assert_almost_equal(esM.pyM.Obj(),11545)

    # Check capacity results:
    # year 0
    esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs("PV").values[0]
    np.testing.assert_almost_equal(esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs("PV").values[0]  ,1.71232876712329 )
    
    # year 1
    esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[1].xs("PV").values[0]
    np.testing.assert_almost_equal(esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[1].xs("PV").values[0]  ,0 )


# _df_tac=pd.DataFrame()
# for i in range(0,6):
#     _df_tac.loc[i,"TAC"]=esM.getOptimizationSummary("SourceSinkModel")[i].loc["PV","TAC"].iloc[0,0]
#     _df_tac.loc[i,"Capacity"]=esM.getOptimizationSummary("SourceSinkModel")[i].loc["PV","capacity"].iloc[0,0]
#     _df_tac.loc[i,"capexCap"]=esM.getOptimizationSummary("SourceSinkModel")[i].loc["PV","capexCap"].iloc[0,0]
#     _df_tac.loc[i,"invest"]=esM.getOptimizationSummary("SourceSinkModel")[i].loc["PV","invest"].iloc[0,0]
#     _df_tac.loc[i,"opexOp"]=esM.getOptimizationSummary("SourceSinkModel")[i].loc["PV","opexOp"].iloc[0,0]
#     #print(esM.getOptimizationSummary("SourceSinkModel")[i].loc["PV","TAC"].iloc[0,0])


if __name__ == "__main__":
    test_perfectForesight_linked_costs()
