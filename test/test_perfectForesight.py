#!/usr/bin/env python
# coding: utf-8

# # Test case for perfect foresight approach

# Status: Working with FINE w/o Perfect Foresight
# Status: No errors with perfect foresight version, but output not working correctly --> sourceSink.py (line 650+)
# Status: Obviously not doing perfect foresight yet, required expansions will come in the future

# 1. Import required packages and set input data path

import FINE as fn
import numpy as np
import pandas as pd

def test_perfectForesight():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    numberOfInvestmentPeriods = 1
    yearsPerInvestmentPeriod=1

    # Create an energy system model instance 
    esM = fn.EnergySystemModel(locations={'PerfectLand'}, 
                                commodities={'electricity'}, 
                                numberOfTimeSteps=numberOfTimeSteps,
                                commodityUnitsDict={'electricity': r'kW$_{el}$'},
                                hoursPerTimeStep=hoursPerTimeStep, costUnit='1 Euro', 
                                numberOfInvestmentPeriods=numberOfInvestmentPeriods,
                                yearsPerInvestmentPeriod=yearsPerInvestmentPeriod,
                                lengthUnit='km', 
                                verboseLogLevel=2)
               
    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    
    # Sources

    # Electricity market
    costs = pd.DataFrame([np.array([ 1,1,1,1,])],
                            index = ['PerfectLand']).T
    revenues = pd.DataFrame([np.array([ 0., 0., 0., 0.,])],
                            index = ['PerfectLand']).T
    maxpurchase = pd.DataFrame([np.array([4e3, 4e3, 4e3, 4e3,])],
                            index = ['PerfectLand']).T
    esM.add(fn.Source(esM=esM, name='Electricity market', commodity='electricity', 
                        hasCapacityVariable=False, operationRateMax = maxpurchase,
                        commodityCostTimeSeries = costs,  
                        commodityRevenueTimeSeries = revenues,  
                        )) # eur/kWh

    # Photovoltaic
    PVoperationRateMax = pd.DataFrame([np.array([0.4, 0.4, 0.6, 0.6,])],
                            index = ['PerfectLand']).T
    esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                  operationRateMax=PVoperationRateMax,
                  capacityMax=4e6,
                  investPerCapacity=2*2190, opexPerCapacity=0, interestRate=0,
                  economicLifetime=1))

    # Sinks

    ### Industry site
    demand = pd.DataFrame([np.array([1e3, 1e3, 1e3, 1e3,])],
                    index = ['PerfectLand']).T
    esM.add(fn.Sink(esM=esM, name='EDemand', commodity='electricity', hasCapacityVariable=False,
                    operationRateFix = demand,
                    ))

    # Optimize energy system model    
    
    #esM.cluster(numberOfTypicalPeriods=4, numberOfTimeStepsPerPeriod=1)

    esM.optimize(timeSeriesAggregation=False, solver = 'gurobi')
    print(esM.pyM.Obj())
    print('Electricity Market:')
    print(esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Electricity market'))

    print('Photovoltaic:')
    print(esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('PV'))

    print('Demand:')
    print(esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('EDemand'))



if __name__ == "__main__":
    test_perfectForesight()
