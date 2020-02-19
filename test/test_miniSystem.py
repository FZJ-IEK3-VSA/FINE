#!/usr/bin/env python
# coding: utf-8

# # Workflow for a multi-regional energy system
# 
# In this application of the FINE framework, a multi-regional energy system is modeled and optimized.
# 
# All classes which are available to the user are utilized and examples of the selection of different parameters within these classes are given.
# 
# The workflow is structures as follows:
# 1. Required packages are imported and the input data path is set
# 2. An energy system model instance is created
# 3. Commodity sources are added to the energy system model
# 4. Commodity conversion components are added to the energy system model
# 5. Commodity storages are added to the energy system model
# 6. Commodity transmission components are added to the energy system model
# 7. Commodity sinks are added to the energy system model
# 8. The energy system model is optimized
# 9. Selected optimization results are presented
# 

# 1. Import required packages and set input data path

import FINE as fn
import numpy as np
import pandas as pd

def test_miniSystem():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance 
    esM = fn.EnergySystemModel(locations={'ElectrolyzerLocation', 'IndustryLocation'}, 
                                commodities={'electricity', 'hydrogen'}, 
                                numberOfTimeSteps=numberOfTimeSteps,
                                commodityUnitsDict={'electricity': r'kW$_{el}$', 'hydrogen': r'kW$_{H_{2},LHV}$'},
                                hoursPerTimeStep=hoursPerTimeStep, costUnit='1 Euro', 
                                lengthUnit='km', 
                                verboseLogLevel=2)

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep


    ### Buy electricity at the electricity market
    costs = pd.DataFrame([np.array([ 0.05, 0., 0.1, 0.051,]),np.array([0., 0., 0., 0.,])],
                            index = ['ElectrolyzerLocation', 'IndustryLocation']).T
    revenues = pd.DataFrame([np.array([ 0., 0.01, 0., 0.,]),np.array([0., 0., 0., 0.,])],
                            index = ['ElectrolyzerLocation', 'IndustryLocation']).T
    maxpurchase = pd.DataFrame([np.array([1e6, 1e6, 1e6, 1e6,]),np.array([0., 0., 0., 0.,])],
                            index = ['ElectrolyzerLocation', 'IndustryLocation']).T * hoursPerTimeStep
    esM.add(fn.Source(esM=esM, name='Electricity market', commodity='electricity', 
                        hasCapacityVariable=False, operationRateMax = maxpurchase,
                        commodityCostTimeSeries = costs,  
                        commodityRevenueTimeSeries = revenues,  
                        )) # eur/kWh

    ### Electrolyzers
    esM.add(fn.Conversion(esM=esM, name='Electroylzers', physicalUnit=r'kW$_{el}$',
                          commodityConversionFactors={'electricity':-1, 'hydrogen':0.7},
                          hasCapacityVariable=True, 
                          investPerCapacity=500, # euro/kW
                          opexPerCapacity=500*0.025, 
                          interestRate=0.08,
                          economicLifetime=10))

    ### Hydrogen filled somewhere
    esM.add(fn.Storage(esM=esM, name='Pressure tank', commodity='hydrogen',
                       hasCapacityVariable=True, capacityVariableDomain='continuous',
                       stateOfChargeMin=0.33, 
                       investPerCapacity=0.5, # eur/kWh
                       interestRate=0.08,
                       economicLifetime=30))

    ### Hydrogen pipelines
    esM.add(fn.Transmission(esM=esM, name='Pipelines', commodity='hydrogen',
                            hasCapacityVariable=True,
                            investPerCapacity=0.177, 
                            interestRate=0.08, 
                            economicLifetime=40))

    ### Industry site
    demand = pd.DataFrame([np.array([0., 0., 0., 0.,]), np.array([6e3, 6e3, 6e3, 6e3,]),],
                    index = ['ElectrolyzerLocation', 'IndustryLocation']).T * hoursPerTimeStep
    esM.add(fn.Sink(esM=esM, name='Industry site', commodity='hydrogen', hasCapacityVariable=False,
                    operationRateFix = demand,
                    ))

    # 8. Optimize energy system model
    
    
    #esM.cluster(numberOfTypicalPeriods=4, numberOfTimeStepsPerPeriod=1)

    esM.optimize(timeSeriesAggregation=False, solver = 'glpk')


    # test if solve fits to the original results
    testresults = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Electricity market')
    np.testing.assert_array_almost_equal(testresults.values, [np.array([1.877143e+07,  3.754286e+07,  0.0,  1.877143e+07]),],decimal=-3)

    # test if the summary fits to the expected summary
    summary = esM.getOptimizationSummary("SourceSinkModel")
    # of cost
    np.testing.assert_almost_equal(summary.loc[('Electricity market','commodCosts','[1 Euro/a]'),'ElectrolyzerLocation'],
        costs['ElectrolyzerLocation'].mul(np.array([1.877143e+07,  3.754286e+07,  0.0,  1.877143e+07])).sum(), decimal=0)
    # and of revenues
    np.testing.assert_almost_equal(summary.loc[('Electricity market','commodRevenues','[1 Euro/a]'),'ElectrolyzerLocation'],
        revenues['ElectrolyzerLocation'].mul(np.array([1.877143e+07,  3.754286e+07,  0.0,  1.877143e+07])).sum(), decimal=0)

if __name__ == "__main__":
    test_miniSystem()