#!/usr/bin/env python
# coding: utf-8

import FINE as fn         # Provides objects and functions to model an energy system 
import pandas as pd       # Used to manage data in tables
import numpy as np        # Used to generate random input data
import os


def test_industrialSite():
    # # Model an energy system

    # Input parameters
    locations = {'industry_0'}
    commodityUnitDict = {'electricity': r'MW$_{el}$', 'hydrogen': r'MW$_{H_{2},LHV}$'}
    commodities = {'electricity', 'hydrogen'}
    numberOfTimeSteps, hoursPerTimeStep = 24*6, 1/6 #8760, 1 # 52560, 1/6
    costUnit, lengthUnit = '1e3 Euro', 'km'

    # Code
    esM = fn.EnergySystemModel(locations=locations, commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps, commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=hoursPerTimeStep, costUnit=costUnit, lengthUnit=lengthUnit, verboseLogLevel=0)


    # ## Add source component

    data = pd.read_excel('test/generationTimeSeries_e825103.xlsx')
    # data = pd.read_excel(os.path.dirname(os.path.abspath(__file__)) + '/generationTimeSeries_e825103.xlsx')
    data = data.iloc[0:numberOfTimeSteps]

    operationRateMax = pd.DataFrame(data['e825103_2017_2.3MW_faults9'],index=range(numberOfTimeSteps)) # Dataset with least missing data
    operationRateMax.columns = ['industry_0']

    # Input parameters
    name, commodity ='Wind turbines', 'electricity'
    hasCapacityVariable = True
    capacityFix = pd.Series([10], index=['industry_0']) # 10 MW_el = 0.01 GW_el
    investPerCapacity, opexPerCapacity = 0, 30 # 30 €/kW = 30 1e6€/GW = 30 1e3€/MW
    interestRate, economicLifetime = 0.08, 20

    # Code
    esM.add(fn.Source(esM=esM, name=name, commodity=commodity, hasCapacityVariable=hasCapacityVariable,
        operationRateMax=operationRateMax, capacityFix=capacityFix, investPerCapacity=investPerCapacity,
        opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

    # ## Add conversion components

    esM.add(fn.Conversion(esM=esM, name='PEMEC', physicalUnit=r'MW$_{el}$',
                        commodityConversionFactors={'electricity':-1, 'hydrogen':0.67},
                        hasCapacityVariable=True, 
                        investPerCapacity=2300, opexPerCapacity=12.5, interestRate=0.08, # for 2018 CAPEX
                        economicLifetime=5))

    esM.add(fn.ConversionFancy(esM=esM, name='AEC', physicalUnit=r'MW$_{el}$',
                        commodityConversionFactors={'electricity':-1, 'hydrogen':0.64},
                        commodityConversionFactorsPartLoad={'electricity':-1, 'hydrogen': lambda x: 0.5*(x-2)**3 + (x-2)**2 + 0.0001},
                        hasCapacityVariable=True, 
                        bigM=99,
                        investPerCapacity=1300, opexPerCapacity=18, interestRate=0.08, # for 2018 CAPEX
                        economicLifetime=9))

    # ## Add storage components

    esM.add(fn.Storage(esM=esM, name='Hydrogen tank (gaseous)', commodity='hydrogen',
                    hasCapacityVariable=True, capacityVariableDomain='continuous',
                    capacityPerPlantUnit=1,
                    chargeRate=1, dischargeRate=1, sharedPotentialID=None,
                    stateOfChargeMin=0.06, stateOfChargeMax=1,
                    investPerCapacity=0.004, opexPerCapacity=0.004*0.02, interestRate=0.08,
                    economicLifetime=20))


    # ### Industrial hydrogen demand

    operationRateFix = pd.DataFrame(2*np.ones(numberOfTimeSteps)*(hoursPerTimeStep), columns=['industry_0']) # constant hydrogen demand of 2 MW_GH2: ATTN
    esM.add(fn.Sink(esM=esM, name='Hydrogen demand', commodity='hydrogen', hasCapacityVariable=False,
                    operationRateFix=operationRateFix))


    # # Optimize energy system model

    # Time Series Clustering
    # esM.cluster(numberOfTypicalPeriods=numberOfTypicalPeriods,numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod)

    # Input parameters
    timeSeriesAggregation=False
    solver='glpk'

    # Optimize
    esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)

    # # Postprocessing

    # Print the discretization variables of the AEC
    testresults = esM.componentModelingDict["ConversionFancyModel"].discretizationPointVariablesOptimun
    print(testresults)

    # Write results to excel file
    # fn.writeOptimizationOutputToExcel(esM, outputFileName='scenarioOutput_industrialSite')

if __name__ == "__main__":
    test_miniSystem()