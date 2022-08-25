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
        yearsPerInvestmentPeriod=2,
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
    PVoperationRateMax = pd.DataFrame(
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
    PVopexPerOperation = {0:0.01, 2:0.01, 4:0.01, 6:0.02, 8:0.02, 10:0.02,}

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            investPerCapacity=100,
            #opexPerCapacity=1,
            interestRate=0.02,
            #opexPerOperation=PVopexPerOperation,
            economicLifetime=4,
            technicalLifetime=4
        )
    )

    # Sinks

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
    demand[2] = pd.DataFrame(
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
    demand[4]=demand[2]
    demand[6]=demand[2]
    demand[8]=demand[2]
    demand[10]=demand[2]

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # Optimize energy system model
    esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    print("Objective value:")
    print(esM.pyM.Obj()) 
    
    # 
    PV_cap_year0=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[0].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year0 ,2.5)
    PV_cap_year4=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[4].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year1 ,1.25)
    PV_cap_year8=esM.componentModelingDict["SourceSinkModel"].capacityVariablesOptimum[8].xs("PV").values[0]
    np.testing.assert_almost_equal(PV_cap_year5 ,1.25)
    
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
