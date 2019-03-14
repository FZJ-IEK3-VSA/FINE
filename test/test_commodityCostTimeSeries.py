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
import os
import pandas as pd
import numpy as np
import xlrd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..','examples','Multi-regional Energy System Workflow_Test_190312'))
from getData import getData

def test_commodityCostTimeSeries():
    cwd = os.getcwd()
    data = getData()

    # read in original results
    results = pd.Series.from_csv(os.path.join(os.path.dirname(__file__),'..','examples','Multi-regional Energy System Workflow_Test_190312','totalTransmission.csv'), index_col = [0,1,2])

    # 2. Create an energy system model instance 
    locations = {'cluster_0', 'cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5', 'cluster_6', 'cluster_7'}
    commodityUnitDict = {'electricity': r'GW$_{el}$', 'methane': r'GW$_{CH_{4},LHV}$', 'biogas': r'GW$_{biogas,LHV}$',
                         'CO2': r'Mio. t$_{CO_2}$/h', 'hydrogen': r'GW$_{H_{2},LHV}$'}
    commodities = {'electricity', 'hydrogen', 'methane', 'biogas', 'CO2'}
    numberOfTimeSteps=8760
    hoursPerTimeStep=1

    esM = fn.EnergySystemModel(locations=locations, commodities=commodities, numberOfTimeSteps=8760,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='1e9 Euro', lengthUnit='km', verboseLogLevel=0)

    CO2_reductionTarget = 1


    # 3. Add commodity sources to the energy system model
    ## 3.1. Electricity sources
    ### Wind onshore

    esM.add(fn.Source(esM=esM, name='Wind (onshore)', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=data['Wind (onshore), operationRateMax'],
                      capacityMax=data['Wind (onshore), capacityMax'],
                      investPerCapacity=1.1, opexPerCapacity=1.1*0.02, interestRate=0.08,
                      economicLifetime=20))

    data['Wind (onshore), operationRateMax'].sum()


    ### Wind offshore

    esM.add(fn.Source(esM=esM, name='Wind (offshore)', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=data['Wind (offshore), operationRateMax'],
                      capacityMax=data['Wind (offshore), capacityMax'],
                      investPerCapacity=2.3, opexPerCapacity=2.3*0.02, interestRate=0.08,
                      economicLifetime=20))

    data['Wind (offshore), operationRateMax'].sum()

    ### PV

    esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=data['PV, operationRateMax'], capacityMax=data['PV, capacityMax'],
                      investPerCapacity=0.65, opexPerCapacity=0.65*0.02, interestRate=0.08,
                      economicLifetime=25))

    data['PV, operationRateMax'].sum()

    ### Exisisting run-of-river hydroelectricity plants

    esM.add(fn.Source(esM=esM, name='Existing run-of-river plants', commodity='electricity',
                      hasCapacityVariable=True,
                      operationRateFix=data['Existing run-of-river plants, operationRateFix'], tsaWeight=0.01,
                      capacityFix=data['Existing run-of-river plants, capacityFix'],
                      investPerCapacity=0, opexPerCapacity=0.208))

    ## 3.2. Methane (natural gas and biogas)
    ### Natural gas        
    esM.add(fn.Source(esM=esM, name='Natural gas purchase', commodity='methane',
                      hasCapacityVariable=False, commodityCostTimeSeries=data['Natural Gas, commodityCostTimeSeries']))

    ### Biogas
    esM.add(fn.Source(esM=esM, name='Biogas purchase', commodity='biogas',
                      operationRateMax=data['Biogas, operationRateMax'], hasCapacityVariable=False,
                      commodityCostTimeSeries=data['Biogas, commodityCostTimeSeries']))


    ## 3.3 CO2
    ### CO2

    esM.add(fn.Source(esM=esM, name='CO2 from enviroment', commodity='CO2',
                      hasCapacityVariable=False, commodityLimitID='CO2 limit', yearlyLimit=366*(1-CO2_reductionTarget)))


    # 4. Add conversion components to the energy system model

    ### Combined cycle gas turbine plants

    esM.add(fn.Conversion(esM=esM, name='CCGT plants (methane)', physicalUnit=r'GW$_{el}$',
                          commodityConversionFactors={'electricity':1, 'methane':-1/0.625, 'CO2':201*1e-6/0.625},
                          hasCapacityVariable=True,
                          investPerCapacity=0.65, opexPerCapacity=0.021, interestRate=0.08,
                          economicLifetime=33))


    ### New combined cycle gas turbine plants for biogas

    esM.add(fn.Conversion(esM=esM, name='New CCGT plants (biogas)', physicalUnit=r'GW$_{el}$',
                          commodityConversionFactors={'electricity':1, 'biogas':-1/0.635},
                          hasCapacityVariable=True, 
                          investPerCapacity=0.7, opexPerCapacity=0.021, interestRate=0.08,
                          economicLifetime=33))


    ### New combined cycly gas turbines for hydrogen

    esM.add(fn.Conversion(esM=esM, name='New CCGT plants (hydrogen)', physicalUnit=r'GW$_{el}$',
                          commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.6},
                          hasCapacityVariable=True, 
                          investPerCapacity=0.7, opexPerCapacity=0.021, interestRate=0.08,
                          economicLifetime=33))

    ### Electrolyzers

    esM.add(fn.Conversion(esM=esM, name='Electroylzers', physicalUnit=r'GW$_{el}$',
                          commodityConversionFactors={'electricity':-1, 'hydrogen':0.7},
                          hasCapacityVariable=True, 
                          investPerCapacity=0.5, opexPerCapacity=0.5*0.025, interestRate=0.08,
                          economicLifetime=10))

    ### rSOC

    capexRSOC=1.5

    esM.add(fn.Conversion(esM=esM, name='rSOEC', physicalUnit=r'GW$_{el}$', linkedConversionCapacityID='rSOC',
                          commodityConversionFactors={'electricity':-1, 'hydrogen':0.6},
                          hasCapacityVariable=True, 
                          investPerCapacity=capexRSOC/2, opexPerCapacity=capexRSOC*0.02/2, interestRate=0.08,
                          economicLifetime=10))

    esM.add(fn.Conversion(esM=esM, name='rSOFC', physicalUnit=r'GW$_{el}$', linkedConversionCapacityID='rSOC',
                          commodityConversionFactors={'electricity':1, 'hydrogen':-1/0.6},
                          hasCapacityVariable=True, 
                          investPerCapacity=capexRSOC/2, opexPerCapacity=capexRSOC*0.02/2, interestRate=0.08,
                          economicLifetime=10))


    # 5. Add commodity storages to the energy system model
    ## 5.1. Electricity storage
    ### Lithium ion batteries

    esM.add(fn.Storage(esM=esM, name='Li-ion batteries', commodity='electricity',
                       hasCapacityVariable=True, chargeEfficiency=0.95,
                       cyclicLifetime=10000, dischargeEfficiency=0.95, selfDischarge=1-(1-0.03)**(1/(30*24)),
                       chargeRate=1, dischargeRate=1, doPreciseTsaModeling=False,
                       investPerCapacity=0.151, opexPerCapacity=0.002, interestRate=0.08,
                       economicLifetime=22))

    ## 5.2. Hydrogen storage
    ### Hydrogen filled salt caverns

    esM.add(fn.Storage(esM=esM, name='Salt caverns (hydrogen)', commodity='hydrogen',
                       hasCapacityVariable=True, capacityVariableDomain='continuous',
                       capacityPerPlantUnit=133,
                       chargeRate=1/470.37, dischargeRate=1/470.37, sharedPotentialID='Existing salt caverns',
                       stateOfChargeMin=0.33, stateOfChargeMax=1, capacityMax=data['Salt caverns (hydrogen), capacityMax'],
                       investPerCapacity=0.00011, opexPerCapacity=0.00057, interestRate=0.08,
                       economicLifetime=30))


    ## 5.3. Methane storage
    ### Methane filled salt caverns

    esM.add(fn.Storage(esM=esM, name='Salt caverns (biogas)', commodity='biogas',
                       hasCapacityVariable=True, capacityVariableDomain='continuous',
                       capacityPerPlantUnit=443,
                       chargeRate=1/470.37, dischargeRate=1/470.37, sharedPotentialID='Existing salt caverns',
                       stateOfChargeMin=0.33, stateOfChargeMax=1, capacityMax=data['Salt caverns (methane), capacityMax'],
                       investPerCapacity=0.00004, opexPerCapacity=0.00001, interestRate=0.08,
                       economicLifetime=30))


    ## 5.4 Pumped hydro storage
    ### Pumped hydro storage

    esM.add(fn.Storage(esM=esM, name='Pumped hydro storage', commodity='electricity',
                       chargeEfficiency=0.88, dischargeEfficiency=0.88,
                       hasCapacityVariable=True, selfDischarge=1-(1-0.00375)**(1/(30*24)),
                       chargeRate=0.16, dischargeRate=0.12, capacityFix=data['Pumped hydro storage, capacityFix'],
                       investPerCapacity=0, opexPerCapacity=0.000153))


    # 6. Add commodity transmission components to the energy system model
    ## 6.1. Electricity transmission
    ### AC cables

    esM.add(fn.LinearOptimalPowerFlow(esM=esM, name='AC cables', commodity='electricity',
                                      hasCapacityVariable=True, capacityFix=data['AC cables, capacityFix'],
                                      reactances=data['AC cables, reactances']))

    ### DC cables

    esM.add(fn.Transmission(esM=esM, name='DC cables', commodity='electricity', losses=data['DC cables, losses'],
                            distances=data['DC cables, distances'],
                            hasCapacityVariable=True, capacityFix=data['DC cables, capacityFix']))


    ## 6.2 Methane transmission
    ### Methane pipeline

    esM.add(fn.Transmission(esM=esM, name='Pipelines (biogas)', commodity='biogas', 
                            distances=data['Pipelines, distances'],
                            hasCapacityVariable=True, hasIsBuiltBinaryVariable=False, bigM=300,
                            locationalEligibility=data['Pipelines, eligibility'],
                            capacityMax=data['Pipelines, eligibility']*15, sharedPotentialID='pipelines',
                            investPerCapacity=0.000037, investIfBuilt=0.000314,
                            interestRate=0.08, economicLifetime=40))

    ## 6.3 Hydrogen transmission
    ### Hydrogen pipelines

    esM.add(fn.Transmission(esM=esM, name='Pipelines (hydrogen)', commodity='hydrogen',
                            distances=data['Pipelines, distances'],
                            hasCapacityVariable=True, hasIsBuiltBinaryVariable=False, bigM=300,
                            locationalEligibility=data['Pipelines, eligibility'],
                            capacityMax=data['Pipelines, eligibility']*15, sharedPotentialID='pipelines',
                            investPerCapacity=0.000177, investIfBuilt=0.00033,
                            interestRate=0.08, economicLifetime=40))

    # 7. Add commodity sinks to the energy system model
    ## 7.1. Electricity sinks
    ### Electricity demand

    esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                    hasCapacityVariable=False, operationRateFix=data['Electricity demand, operationRateFix']))

    ## 7.2. Hydrogen sinks
    ### Fuel cell electric vehicle (FCEV) demand

    FCEV_penetration=0.5
    esM.add(fn.Sink(esM=esM, name='Hydrogen demand', commodity='hydrogen', hasCapacityVariable=False,
                    operationRateFix=data['Hydrogen demand, operationRateFix']*FCEV_penetration))

    ## 7.3. CO2 sinks
    ### CO2 exiting the system's boundary

    esM.add(fn.Sink(esM=esM, name='CO2 to enviroment', commodity='CO2',
                    hasCapacityVariable=False, commodityLimitID='CO2 limit', yearlyLimit=366*(1-CO2_reductionTarget)))

    # 8. Optimize energy system model

    esM.cluster(numberOfTypicalPeriods=14)

    esM.optimize(timeSeriesAggregation=True, optimizationSpecs='OptimalityTol=1e-3 cuts=0 method=2')


    # test if here solved fits with original results
    testresults = esM.componentModelingDict["TransmissionModel"].operationVariablesOptimum.sum(axis=1)
    np.testing.assert_array_almost_equal(testresults.values, results.values,decimal=2)

if __name__ == "__main__":
    test_commodityCostTimeSeries()