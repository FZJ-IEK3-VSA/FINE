"""
Last edited: May 13 2018

@author: Lara Welder
"""

import warnings
import FINE.utils as utils
from tsam.timeseriesaggregation import TimeSeriesAggregation
import pandas as pd
import pyomo.environ as pyomo
import pyomo.opt as opt
import time
import datetime


class EnergySystemModel:
    """
    Energy system model class
    """

    def __init__(self, locations, commodities, numberOftimeSteps=8760, hoursPerTimeStep=1,
                 powerUnit='GW', energyUnit='GWh', costUnit='1e9 Euro', lengthUnit='km'):
        """
        Doc
        """
        # Check correctness of inputs
        for sets in [locations, commodities]:
            utils.isSetOfStrings(sets)
        utils.isStrictlyPositiveInt(numberOftimeSteps)
        utils.isStrictlyPositiveNumber(hoursPerTimeStep)
        for string in [powerUnit, costUnit, energyUnit, lengthUnit]:
            utils.isString(string)

        # Energy system specific parameters
        self._locations, self._commodities, self._componentNames = locations, commodities, {}

        # Time series parameters
        self._years = numberOftimeSteps * hoursPerTimeStep / 8760.0
        self._totalTimeSteps = list(range(numberOftimeSteps))
        self._timeStepsPerPeriod = list(range(numberOftimeSteps))
        self._periods = [0]
        self._typicalPeriods = [0]
        self._interPeriodTimeSteps = list(range(int(len(self._totalTimeSteps) / len(self._timeStepsPerPeriod)) + 1))
        self._periodsOrder = [0]
        self._periodOccurrences = [1]
        self._hoursPerTimeStep = hoursPerTimeStep
        self._isTimeSeriesDataClustered = False

        # Unit parameters
        self._costUnit = costUnit
        self._powerUnit = powerUnit
        self._timeUnit = 'h'
        self._energyUnit = energyUnit
        self._lengthUnit = lengthUnit

        # Component specific parameters
        self._componentModelingDict = {}

        # Optimization parameters
        self._pyM = pyomo.ConcreteModel()
        self._runtime = 0
        self._optimizationSpecs = 'LogToConsole=1 OptimalityTol=1e-6'
        self._solver = 'gurobi'
        self._timeLimit = None
        self._threads = 0
        self._jobName = 'job'

    def add(self, component):
        """ Function for adding components to the energy system model """
        component.addToESM(self)

    def cluster(self, numberOfTypicalPeriods=7, numberOfTimeStepsPerPeriod=24, clusterMethod='hierarchical',
                sortValues=True, extremePeriodMethod='None', **kwargs):
        """
        Clusters all time series data which is specified in the energy system
        structure and stores the clustered data in the components of the energy
        system

        Default arguments:
            numberOfTypicalPeriods - integer, strictly positive (> 0)
                states the number of typical periods for the time series
                aggregation, must be smaller than the number of time slices
                (=totalNumberOfHours/hoursPerTimeSlices)
                * the default value is 7

            numberOfTimeSlicesPerPeriod - integer, strictly positive (> 0)
                states the number of time slices within a period, must be a
                divisor of 8760
                * the default value is 24
        """
        # Store time series aggregation parameters in class instance
        self._typicalPeriods = list(range(numberOfTypicalPeriods))
        self._timeStepsPerPeriod = list(range(numberOfTimeStepsPerPeriod))
        self._interPeriodTimeSteps = list(range(int(len(self._totalTimeSteps) / len(self._timeStepsPerPeriod)) + 1))

        # Format data to the input requirements of the tsam package
        timeSeriesData, weightDict = [], {}
        for mdlName, mdl in self._componentModelingDict.items():
            for compName, comp in mdl._componentsDict.items():
                compTimeSeriesData, compWeightDict = comp.getDataForTimeSeriesAggregation()
                if compTimeSeriesData is not None:
                    timeSeriesData.append(compTimeSeriesData), weightDict.update(compWeightDict)
        timeSeriesData = pd.concat(timeSeriesData, axis=1)
        timeSeriesData.index = pd.date_range('2050-01-01 00:30:00', periods=len(self._totalTimeSteps),
                                             freq=(str(self._hoursPerTimeStep) + 'H'), tz='Europe/Berlin')

        # Cluster data with tsam package
        clusterClass = TimeSeriesAggregation(timeSeries=timeSeriesData, noTypicalPeriods=numberOfTypicalPeriods,
                                             hoursPerPeriod=numberOfTimeStepsPerPeriod*self._hoursPerTimeStep,
                                             extremePeriodMethod=extremePeriodMethod, clusterMethod=clusterMethod,
                                             sortValues=sortValues, **kwargs)
        clusterClass.prepareEnersysInput()
        # Save cluster day order as DataFrame and compute and store occurrence of each typical period type
        self._periodsOrder = clusterClass.clusterOrder
        self._periodOccurrences = [(self._periodsOrder == p).sum()/self._years for p in self._typicalPeriods]

        # Convert clustered data to DataFrame
        data = pd.DataFrame.from_dict(clusterClass.clusterPeriodDict)

        # Store clustered data in components
        for mdlName, mdl in self._componentModelingDict.items():
            for compName, comp in mdl._componentsDict.items():
                comp.setAggregatedTimeSeriesData(data)

        # Set cluster flag to true
        self._isTimeSeriesDataClustered = True

    def optimize(self, timeSeriesAggregation=False, jobName='job', threads=0, solver='gurobi', timeLimit=None,
                 optimizationSpecs='LogToConsole=1 OptimalityTol=1e-6', warmstart=False, tsamSpecs=None):
        timeStart = time.time()
        self._jobName = jobName
        self._threads = threads
        self._solver = solver
        self._timeLimit = timeLimit
        self._optimizationSpecs = optimizationSpecs

        self._pyM = pyomo.ConcreteModel()
        pyM = self._pyM
        pyM.hasTSA = timeSeriesAggregation

        if timeSeriesAggregation and tsamSpecs is not None:
            self.cluster(**tsamSpecs)
        elif timeSeriesAggregation and not self._isTimeSeriesDataClustered and tsamSpecs is None :
            warnings.warn('The time series flag indicates that not all time series data might be clustered.\n' +
                          'Clustering time series data with default values:')
            self.cluster()

        for mdlName, mdl in self._componentModelingDict.items():
            for comp in mdl._componentsDict.values():
                comp.setTimeSeriesData(pyM.hasTSA)

        # Initialize time sets
        if not pyM.hasTSA:
            # Reset timeStepsPerPeriod in case it was overwritten by the clustering function
            self._timeStepsPerPeriod = self._totalTimeSteps

            def initTimeSet(pyM):
                return ((p, t) for p in self._periods for t in self._timeStepsPerPeriod)

            def initInterTimeStepsSet(pyM):
                return ((p, t) for p in self._periods for t in range(len(self._timeStepsPerPeriod) + 1))
        else:
            def initTimeSet(pyM):
                return ((p, t) for p in self._typicalPeriods for t in self._timeStepsPerPeriod)

            def initInterTimeStepsSet(pyM):
                return ((p, t) for p in self._typicalPeriods for t in range(len(self._timeStepsPerPeriod) + 1))
        pyM.timeSet = pyomo.Set(dimen=2, initialize=initTimeSet)
        pyM.interTimeStepsSet = pyomo.Set(dimen=2, initialize=initInterTimeStepsSet)

        # Create shared potential dictionary
        potentialDict = {}
        for mdlName, mdl in self._componentModelingDict.items():
            for compName, comp in mdl._componentsDict.items():
                if comp._sharedPotentialID is not None:
                    for loc in self._locations:
                        if comp._capacityMax[loc] != 0:
                            potentialDict.setdefault((comp._sharedPotentialID, loc), []).append(compName)
        pyM.sharedPotentialDict = potentialDict

        # Declare component specific sets, variables and constraints
        for key, comp in self._componentModelingDict.items():
            _t = time.time()
            print('Declaring sets, variables and constraints for', key)
            print('\tdeclaring sets... '), comp.declareSets(self, pyM)
            print('\tdeclaring variables... '), comp.declareVariables(self, pyM),
            print('\tdeclaring constraints... '), comp.declareComponentConstraints(self, pyM)
            print("\t\t(%.4f" % (time.time() - _t), "sec)")

        _t = time.time()
        print('Declaring shared potential constraint...')

        def sharedPotentialConstraint(pyM, ID, loc):
            # sum up percentage of maximum capacity for each component and location & ensure that the total <= 100%
            return sum(comp.getSharedPotentialContribution(pyM, ID, loc)
                       for comp in self._componentModelingDict.values()) <= 1
        pyM.ConstraintSharedPotentials = \
            pyomo.Constraint(pyM.sharedPotentialDict.keys(), rule=sharedPotentialConstraint)
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        _t = time.time()
        print('Declaring commodity balances...')

        def initLocationCommoditySet(pyM):
            return ((loc, commod) for loc in self._locations for commod in self._commodities
                    if any([comp.hasOpVariablesForLocationCommodity(self, loc, commod)
                            for comp in self._componentModelingDict.values()]))
        pyM.locationCommoditySet = pyomo.Set(dimen=2, initialize=initLocationCommoditySet)

        def commodityBalanceConstraint(pyM, loc, commod, p, t):
            return sum(comp.getCommodityBalanceContribution(pyM, commod, loc, p, t)
                       for comp in self._componentModelingDict.values()) == 0
        pyM.commodityBalanceConstraint = pyomo.Constraint(pyM.locationCommoditySet, pyM.timeSet,
                                                          rule=commodityBalanceConstraint)
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        _t = time.time()
        print('Declaring objective function...')

        def objective(pyM):
            TAC = sum(comp.getObjectiveFunctionContribution(self, pyM) for comp in self._componentModelingDict.values())
            return TAC
        pyM.Obj = pyomo.Objective(rule=objective)
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        optimizer = opt.SolverFactory(solver)
        if self._timeLimit is not None:
            optimizer.options['timelimit'] = timeLimit
        optimizer.set_options('Threads=' + str(threads) + ' logfile=' + jobName + ' ' + optimizationSpecs)
        solver_info = optimizer.solve(pyM, warmstart=warmstart, tee=True)

        print(solver_info.solver())
        print(solver_info.problem())
        print("\t\t(%.4f" % (time.time() - _t), "sec)")
        _t = time.time()

        results = {}
        if solver_info.solver.termination_condition == opt.TerminationCondition.infeasibleOrUnbounded or \
                        solver_info.solver.termination_condition == opt.TerminationCondition.infeasible or \
                        solver_info.solver.termination_condition == opt.TerminationCondition.unbounded:
            print('Optimization problem is ' + str(
                solver_info.solver.termination_condition) + '. No output is generated.')
        else:
            if not (solver_info.solver.status == opt.SolverStatus.ok and
                    solver_info.solver.termination_condition == opt.TerminationCondition.optimal):
                warnings.warn('Output is generated for a non-optimal solution.')
            print("TODO: Generating output...")
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        self._runtime = time.time() - timeStart
