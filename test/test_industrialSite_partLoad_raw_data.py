#!/usr/bin/env python
# coding: utf-8

import FINE as fn         # Provides objects and functions to model an energy system 
import pandas as pd       # Used to manage data in tables
import numpy as np        # Used to generate random input data
import matplotlib.pyplot as plt
import os

def test_industrialSite():
    # # Model an energy system

    # Input parameters
    locations = {'industry_0'}
    commodityUnitDict = {'electricity': r'MW$_{el}$', 'hydrogen': r'MW$_{H_{2},LHV}$','heat':r'MW$_{heat}$}'}
    commodities = {'electricity', 'hydrogen','heat'}
    numberOfTimeSteps, hoursPerTimeStep = 24*6, 1/6 #8760, 1 # 52560, 1/6
    costUnit, lengthUnit = '1e3 Euro', 'km'

    # Code
    esM = fn.EnergySystemModel(locations=locations, commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps, commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=hoursPerTimeStep, costUnit=costUnit, lengthUnit=lengthUnit, verboseLogLevel=0)

    # ## Add source component
    data = pd.read_excel(os.path.join(os.path.dirname(__file__), '_testInputFiles', 
                                       'generationTimeSeries_e825103.xlsx'))

    data = data.iloc[0:numberOfTimeSteps]

    operationRateFix = pd.DataFrame(data['e825103_2017_2.3MW_faults9'],index=range(numberOfTimeSteps)) # Dataset with least missing data
    operationRateFix.columns = ['industry_0']

    # Input parameters
    name, commodity ='Wind turbines', 'electricity'
    hasCapacityVariable = True
    capacityMax = pd.Series([10], index=['industry_0']) # 10 MW_el = 0.01 GW_el
    investPerCapacity, opexPerCapacity = 0, 30 # 30 €/kW = 30 1e6€/GW = 30 1e3€/MW
    interestRate, economicLifetime = 0.08, 20

    esM.add(fn.Source(esM=esM, name=name, commodity=commodity, hasCapacityVariable=hasCapacityVariable,
        operationRateFix=operationRateFix, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
        opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

    # ## Add conversion components

    esM.add(fn.Conversion(esM=esM, name='PEMEC', physicalUnit=r'MW$_{el}$',
                        commodityConversionFactors={'electricity':-1, 'hydrogen':0.67},
                        hasCapacityVariable=True, 
                        investPerCapacity=2300, opexPerCapacity=12.5, interestRate=0.08, # for 2018 CAPEX
                        economicLifetime=5))

    func = lambda x: 0.5*(x-2)**3 + (x-2)**2 + 0.0001

    esM.add(fn.ConversionPartLoad(esM=esM, name='AEC', physicalUnit=r'MW$_{el}$',
                        commodityConversionFactors={'electricity':-1, 'hydrogen':0.64},
                        commodityConversionFactorsPartLoad={'electricity':-1, 'hydrogen': func},
                        # commodityConversionFactorsPartLoad=pwl,
                        nSegments=2,
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


    ### Industrial hydrogen demand
    operationRateFix = pd.DataFrame(2*np.ones(numberOfTimeSteps)*(hoursPerTimeStep), columns=['industry_0']) # constant hydrogen demand of 2 MW_GH2: 
    esM.add(fn.Sink(esM=esM, name='Hydrogen demand', commodity='hydrogen', hasCapacityVariable=False,
                    operationRateFix=operationRateFix))

    # Heat output
    esM.add(fn.Sink(esM=esM, name='Heat output', commodity='heat', hasCapacityVariable=False))

    # Input parameters
    timeSeriesAggregation=False
    solver='glpk'

    # Optimize
    esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)

if __name__ == "__main__":
    test_industrialSite()