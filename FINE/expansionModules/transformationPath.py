"""
Last edited: February 06, 2020

@author: Theresa Groß
"""

import pandas as pd 
from FINE import utils
from FINE.IOManagement import standardIO
import copy

def optimizeMyopic(esM, startYear, endYear=None, nbOfSteps=None, nbOfRepresentedYears=None,
                    timeSeriesAggregation=True, numberOfTypicalPeriods = 7, numberOfTimeStepsPerPeriod=24,
                    logFileName='', threads=3, solver='gurobi', timeLimit=None, 
                    optimizationSpecs='', warmstart=False, CO2ReductionTargets=None):
    """
    Optimization function for myopic approach. For each optimization run, the newly installed capacities
    will be given as a stock (with capacityFix) to the next optimization run.

    :param esM: EnergySystemModel instance representing the energy system which should be optimized by considering the
                transformation pathway (myopic foresight).
    :type esM: esM - EnergySystemModel class instance

    :param startYear: year of the first optimization
    :type startYear: int

    **Default arguments:**

    :param endYear: year of the last optimization
    :type endYear: int

    :param nbOfSteps: number of optimization runs excluding the start year 
            (minimum number of optimization runs is 2: one optimization for the start year and one for the end year).
        |br| * the default value is None
    :type nbOfSteps: int or None 
                    
    :param noOfRepresentedYears: number of years represented by one optimization run
        |br| * the default value is None
    :type nbOfRepresentedYears: int or None

    :param timeSeriesAggregation: states if the optimization of the energy system model should be done with
        (a) the full time series (False) or
        (b) clustered time series data (True).
        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param numberOfTypicalPeriods: states the number of typical periods into which the time series data
        should be clustered. The number of time steps per period must be an integer multiple of the total
        number of considered time steps in the energy system. This argument is used if timeSeriesAggregation is set to True.
        Note: Please refer to the tsam package documentation of the parameter noTypicalPeriods for more
        information.
        |br| * the default value is 7
    :type numberOfTypicalPeriods: strictly positive integer

    :param numberOfTimeStepsPerPeriod: states the number of time steps per period
        |br| * the default value is 24
    :type numberOfTimeStepsPerPeriod: strictly positive integer

    :param CO2ReductionTargets: specifies the CO2 reduction targets for all optimization periods. 
        If specified, the length of the list must equal the number of optimization steps, and an object of the sink class 
        which counts the CO2 emission is required. 
        |br| * the default value is None
    :type CO2ReductionTargets: list of strictly positive integer or None

    Last edited: February 06, 2020
    |br| @author: Theresa Gross, Felix Kullmann
    """                              
                
    nbOfSteps, nbOfRepresentedYears = utils.checkAndSetTimeHorizon(startYear, endYear, nbOfSteps, nbOfRepresentedYears)
    print('Number of optimization steps: ', nbOfSteps, '(+ 1 (base year))')
    print('Number of years represented by one optimization: ', nbOfRepresentedYears)
    mileStoneYear = startYear
    utils.checkCO2ReductionsTargets(CO2ReductionTargets, nbOfSteps)

    for step in range(0,nbOfSteps+1):
        mileStoneYear = startYear + step*nbOfRepresentedYears
        logFileName = 'log_'+str(mileStoneYear)
        # First optimization: Optimize start year for first stock
        if timeSeriesAggregation:
            esM.cluster(numberOfTypicalPeriods=numberOfTypicalPeriods, numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod)

        esM.optimize(declaresOptimizationProblem=True, timeSeriesAggregation=timeSeriesAggregation, 
                        logFileName=logFileName, threads=threads, solver=solver, timeLimit=timeLimit, 
                        optimizationSpecs=optimizationSpecs, warmstart=False)

        standardIO.writeOptimizationOutputToExcel(esM, outputFileName='ESM'+str(mileStoneYear), optSumOutputLevel=2, optValOutputLevel=1)
                    
        # Get first stock (installed capacities within the start year)
        esM = getStock(esM, mileStoneYear, nbOfRepresentedYears)

    return esM

def getStock(esM, mileStoneYear, nbOfRepresentedYears):
    '''
    Function for determining the stock of all considered technologies for the next optimization period. 
    If the technical lifetime is expired, the fixed capacities of the concerned components are set to 0.

    :param mileStoneYear: Last year of the optimization period
    :type mileStoneYear: int

    :param nbOfRepresentativeYears: Number of years within one optimization period.
    :type nbOfRepresentativeYears: int

    Last edited: December 17, 2019
    |br| @author: Theresa Gross, Felix Kullmann
    ''' 
    for mdl in esM.componentModelingDict.keys():
        compValues = esM.componentModelingDict[mdl].getOptimalValues('capacityVariablesOptimum')['values']
        if compValues is not None:
            for comp in compValues.index.get_level_values(0).unique():
                if 'stock' not in esM.componentModelingDict[mdl].componentsDict[comp].name:
                    stockName = comp+'_stock'+'_'+str(mileStoneYear)
                    stockComp = copy.copy(esM.componentModelingDict[mdl].componentsDict[comp])
                    stockComp.name = stockName
                    stockComp.lifetime = esM.componentModelingDict[mdl].componentsDict[comp].technicalLifetime
                    if getattr(stockComp, 'capacityFix') is None:
                        if isinstance(compValues.loc[comp], pd.DataFrame):
                            stockComp.capacityFix = utils.preprocess2dimData(compValues.loc[comp].fillna(value=-1), discard=False)
                        else:
                            stockComp.capacityFix = compValues.loc[comp]
                            stockComp.capacityMin, stockComp.capacityMax = None, None
                        esM.add(stockComp)
                elif 'stock' in esM.componentModelingDict[mdl].componentsDict[comp].name:
                    esM.componentModelingDict[mdl].componentsDict[comp].lifetime -= nbOfRepresentedYears
                    if any(getattr(esM.componentModelingDict[mdl].componentsDict[comp],'lifetime') <= 0):
                        setattr(esM.componentModelingDict[mdl].componentsDict[comp], 'capacityFix', pd.Series(0, index=getattr(esM,'locations')))
    return esM