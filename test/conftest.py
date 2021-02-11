
import pytest

import FINE as fn
import numpy as np
import pandas as pd


@pytest.fixture
def minimal_test_esM():
    """Returns minimal instance of esM"""

    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance 
    esM = fn.EnergySystemModel(locations={'ElectrolyzerLocation', 'IndustryLocation'}, 
                                commodities={'electricity', 'hydrogen'}, 
                                numberOfTimeSteps=numberOfTimeSteps,
                                commodityUnitsDict={'electricity': r'kW$_{el}$', 'hydrogen': r'kW$_{H_{2},LHV}$'},
                                hoursPerTimeStep=hoursPerTimeStep, costUnit='1 Euro', 
                                lengthUnit='km', 
                                verboseLogLevel=1)

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
    esM.add(fn.Conversion(esM=esM, name='Electrolyzers', physicalUnit=r'kW$_{el}$',
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

    return esM


@pytest.fixture
def dsm_test_esM():
    """
    Generate a simple energy system model with one node, two fixed generators and one load time series
    for testing demand side management functionality.
    """
    # load without dsm
    now = pd.Timestamp.now().round('h')
    number_of_time_steps = 28
    #t_index = pd.date_range(now, now + pd.DateOffset(hours=number_of_timeSteps - 1), freq='h')
    t_index = range(number_of_time_steps)
    load_without_dsm = pd.Series([80.] * number_of_time_steps, index=t_index)

    timestep_up = 10
    timestep_down = 20
    load_without_dsm[timestep_up:timestep_down] += 40.

    time_shift = 3
    cheap_capacity = 100.
    expensive_capacity = 20.

    # set up energy model
    esM = fn.EnergySystemModel(locations={'location'},
                               commodities={'electricity'},
                               numberOfTimeSteps=number_of_time_steps,
                               commodityUnitsDict={'electricity': r'MW$_{el}$'},
                               hoursPerTimeStep=1, costUnit='1 Euro',
                               lengthUnit='km',
                               verboseLogLevel=2)
    esM.add(fn.Source(esM=esM, name='cheap', commodity='electricity', hasCapacityVariable=False,
                      operationRateMax=pd.Series(cheap_capacity, index=t_index), opexPerOperation=25))
    esM.add(fn.Source(esM=esM, name='expensive', commodity='electricity', hasCapacityVariable=False,
                      operationRateMax=pd.Series(expensive_capacity, index=t_index), opexPerOperation=50))
    esM.add(fn.Source(esM=esM, name='back-up', commodity='electricity', hasCapacityVariable=False,
                      operationRateMax=pd.Series(1000, index=t_index), opexPerOperation=1000))

    return esM, load_without_dsm, timestep_up, timestep_down, time_shift, cheap_capacity