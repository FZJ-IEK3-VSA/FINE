"""
Last edited: July 27 2018
|br| @author: Lara Welder
"""

import FINE.utils as utils
from FINE.component import Component
from tsam.timeseriesaggregation import TimeSeriesAggregation
import pandas as pd
import pyomo.environ as pyomo
import pyomo.opt as opt
import time
import warnings


class EnergySystemModel:
    """
    EnergySystemModel class

    The functionality provided by the EnergySystemModel class is fourfold:
    * With it, the **basic structure** (spatial and temporal resolution, considered commodities) of
      the investigated energy system is defined.
    * It serves as a **container for all components** investigated in the energy system model. These components,
      namely sources and sinks, conversion options, storage options, and transmission options
      (in the core module), can be added to an EnergySystemModel instance.
    * It provides the core functionality of **modeling and optimizing the energy system** based on the specified
      structure and components on the one hand and of specified simulation parameters on the other hand,.
    * It **stores optimization results** which can then be post-processed with other modules.

    The parameter which are stored in an instance of the class refer to:
    * the modeled spatial representation of the energy system (**locations, lengthUnit**)
    * the modeled temporal representation of the energy system (**totalTimeSteps, hoursPerTimeStep,
      years, periods, periodsOrder, periodsOccurrences, timeStepsPerPeriod, interPeriodTimeSteps,
      isTimeSeriesDataClustered, typicalPeriods, tsaInstance, timeUnit**)
    * the considered commodities in the energy system (**commodities, commoditiyUnitsDict**)
    * the considered components in the energy system (**componentNames, componentModelingDict, costUnit**)
    * optimization related parameters (**pyM, solverSpecs**)
      all parameters are marked as protected (thus they all begin with an underscore) and are set when an class
      instance is initiated, components are added or user accessible functions are called.

    Instances of this class provide function for
    * adding components and their respective modeling classes (**add**)
    * clustering the time series data of all added components using the time series aggregation package tsam, cf.
      https://github.com/FZJ-IEK3-VSA/tsam (**cluster**)
    * optimizing the specified energy system (**optimize**), for which a pyomo discrete model instance is build
      and filled with
      (0) basic time sets,
      (1) sets, variables and constraints contributed by the component modeling classes,
      (2) basic, component overreaching constraints, and
      (3) an objective function.
      The pyomo instance is then optimized by a specified solver and the optimization results processed once
      available.
    * getting components and their attributes (**getComponent, getCompAttr**)

    Last edited: July 27, 2018
    |br| @author: Lara Welder
    """

    def __init__(self, locations, commodities, commoditiyUnitsDict, numberOfTimeSteps=8760, hoursPerTimeStep=1,
                 costUnit='1e9 Euro', lengthUnit='km'):
        """
        Constructor for creating an EnergySystemModel class instance

        **Required arguments:**

        :param locations: locations considered in the energy system
        :type locations: set of strings

        :param commodities: commodities considered in the energy system
        :type commodities: set of strings

        :param commoditiyUnitsDict: dictionary which assigns each commodity a quantitative unit per time
            (e.g. GW_el, GW_H2, Mio.t_CO2/h). The dictionary is used for results output.
            Note for advanced users: the scale of these units can influence the numerical stability of the
            optimization solver, cf. http://files.gurobi.com/Numerics.pdf where a reasonable range of model
            coefficients is suggested
        :type commoditiyUnitsDict: dictionary of strings

        **Default arguments:**

        :param numberOfTimeSteps: number of time steps considered when modeling the energy system (for each
            time step, or each representative time step, variables and constraints are constituted). Together
            with the hours per time step, the total number of hours considered can be derived. The total
            number of hours is again used for scaling the arising costs to the arising total annual costs (TAC),
            which are minimized during optimization.
            |br| * the default value is 8760
        :type totalNumberOfHours: strictly positive integer

        :param hoursPerTimeStep: hours per time step
            |br| * the default value is 1
        :type totalNumberOfHours: strictly positive float

        :param costUnit: cost unit of all cost related values in the energy system. This value sets the unit of
            all cost parameters which are given as an input to the EnergySystemModel instance (i.e. for the
            invest per capacity or the cost per operation).
            Note for advanced users: the scale of this unit can influence the numerical stability of the
            optimization solver, cf. http://files.gurobi.com/Numerics.pdf where a reasonable range of model
            coefficients is suggested
            |br| * the default value is '10^9 Euro' (billion euros), which can be a suitable scale for national
                energy systems.
        :type costUnit: string

        :param lengthUnit: length unit for all length related values in the energy system
            Note for advanced users: the scale of this unit can influence the numerical stability of the
            optimization solver, cf. http://files.gurobi.com/Numerics.pdf where a reasonable range of model
            coefficients is suggested.
            |br| * the default value is 'km' (kilometers)
        :type lengthUnit: string

        Last edited: July 27, 2018
        |br| @author: Lara Welder
        """

        # Check correctness of inputs
        utils.checkEnergySystemModelInput(locations, commodities, commoditiyUnitsDict, numberOfTimeSteps,
                                          hoursPerTimeStep, costUnit, lengthUnit)

        ################################################################################################################
        #                                        Spatial resolution parameters                                         #
        ################################################################################################################

        # The locations (set of string) name the considered location in an energy system model instance. The parameter
        # is used throughout the build of the energy system model to validate inputs and declare relevant sets,
        # variables and constraints.
        # The length unit refers to length measure referred throughout the model.
        self._locations, self._lengthUnit = locations, lengthUnit

        ################################################################################################################
        #                                            Time series parameters                                            #
        ################################################################################################################

        # The totalTimeSteps (list, ranging from 0 to the total numberOfTimeSteps-1) refers to the total number of time
        # steps considered when modeling the specified energy system. The parameter is used for validating time series
        # data input and for setting other time series parameters when modeling a full temporal resolution.
        # The hoursPerTimeStep parameter (float > 0) refers to the temporal length of a time step in the totalTimeSteps
        # From the numberOfTimeSteps and the hoursPerTimeStep the numberOfYears parameter is computed.
        self._totalTimeSteps, self._hoursPerTimeStep = list(range(numberOfTimeSteps)), hoursPerTimeStep
        self._numberOfYears = numberOfTimeSteps * hoursPerTimeStep / 8760.0

        # The periods parameter (list, [0] when considering a full temporal resolution, range of [0, ...,
        # totalNumberOfTimeSteps/numberOfTimeStepsPerPeriod] when applying time series aggregation) represents the
        # the periods considered when modeling the energy system. Only one period exists when considering the full
        # temporal resolution. When applying time series aggregation, the full time series are broken down into
        # periods to which a typical period is assigned to.
        # These periods have an order which is stored in the periodsOrder parameter (list, [0] when considering a full
        # temporal resolution, [typicalPeriod(0), ..., typicalPeriod(totalNumberOfTimeSteps/numberOfTimeStepsPerPeriod)]
        # when applying time series aggregation).
        # The occurrences of these periods are stored in the periodsOccurrences parameter (list, [1] when considering a
        # full temporal resolution, [occurrences(0), ..., occurrences(numberOfTypicalPeriods-1)) when applying time
        # series aggregation).
        self._periods, self._periodsOrder, self._periodOccurrences = [0], [0], [1]
        self._timeStepsPerPeriod = list(range(numberOfTimeSteps))
        self._interPeriodTimeSteps = list(range(int(len(self._totalTimeSteps) / len(self._timeStepsPerPeriod)) + 1))

        # The isTimeSeriesDataClustered is used to check data consistency. It is set to True is the class' cluster
        # function is called. It set to False if a new component is added.
        # If the cluster function is called, the typicalPeriods parameters is set from None to
        # [0, ..., numberOfTypicalPeriods-1] and if specified the resulting TimeSeriesAggregation instance is stored
        # in the tsaInstance parameter (default None)
        # The time unit refers to time measure referred throughout the model. Currently, it has to be an hour 'h'.
        self._isTimeSeriesDataClustered, self._typicalPeriods, self._tsaInstance = False, None, None
        self._timeUnit = 'h'

        ################################################################################################################
        #                                        Commodity specific parameters                                         #
        ################################################################################################################

        # The commodities parameter is a set of strings which describes what commodities are considered in the energy
        # system and hence which commodity balances need to be considered in the energy system model and its
        # optimization.
        # The commodityUnitsDict is a dictionary which assigns each considered commodity (string) a unit (string)
        # which is can be used by results output functions.
        self._commodities = commodities
        self._commoditiyUnitsDict = commoditiyUnitsDict

        ################################################################################################################
        #                                        Component specific parameters                                         #
        ################################################################################################################

        # The componentNames parameter is a set of strings in which all in the EnergySystemModel instance considered
        # components are stored. It is used to check that all components have unique indices.
        # The componentModelingDict is a dictionary (modelingClass name: modelingClass instance) in which the in the
        # energy system considered modeling classes are stored (in which again the components modeled with the
        # modelingClass as well as the equations to model them with are stored)
        # The costUnit parameter (string) is the parameter in which all cost input parameter have to be specified
        self._componentNames = {}
        self._componentModelingDict = {}
        self._costUnit = costUnit

        ################################################################################################################
        #                                           Optimization parameters                                            #
        ################################################################################################################

        # The pyM parameter (None when the EnergySystemModel is initialized otherwise a Concrete Pyomo instance which
        # stores parameters, sets, variables, constraints and objective required for the optimization set up and
        # solving)
        # The solverSpecs parameter is a dictionary (string: param) which stores different parameters that were used
        # when solving the last optimization problem. The parameter are: solver (string, solver which was used to solve
        # the optimization problem), optimizationSpecs (string representing **kwargs for the solver), hasTSA (boolean,
        # indicating if time series aggregation is used for the optimization), runtime (positive float, runtime of the
        # optimization run in seconds), timeLimit (positive float or None, if specified indicates the maximum allowed
        # runtime of the solver), threads (positive int, number of threads used for optimization, can depend on solver),
        # logFileName (string, name of logfile)
        self._pyM = None
        self._solverSpecs = {'solver': '', 'optimizationSpecs': '', 'hasTSA': False, 'runtime': 0, 'timeLimit': None,
                             'threads': 0, 'logFileName': ''}

    def add(self, component):
        """
        Function for adding a component and, if required its respective modeling class to the EnergySystemModel instance

        :param componen: the component to be added
        :type component: An object which inherits from the FINE Component class
        """
        if not issubclass(type(component), Component):
            raise TypeError('The added component has to inherit from the FINE class Component.')
        component.addToEnergySystemModel(self)

    def getComponent(self, componentName):
        """
        Function which returns a component of the energy system

        :param componentName: name of the component that should be returned
        :type componentName: string

        :returns: the component which has the name componentName
        :rtype: Component
        """
        if componentName not in self._componentNames.keys():
            raise ValueError('The component ' + componentName + ' cannot be found in the energy system model.\n' +
                             'The components considered in the model are: ' + str(self._componentNames.keys()))
        modelingClass = self._componentNames[componentName]
        return self._componentModelingDict[modelingClass]._componentsDict[componentName]

    def getComponentAttribute(self, componentName, attributeName):
        """
        Function which returns an attribute of a component considered in the energy system

        :param componentName: name of the component from which the attribute should be obtained
        :type componentName: string

        :param attributeName: name of the attributed that should be returned
        :type attributeName: string

        :returns: the attribute specified by the attributeName of the component with the name componentName
        :rtype: depends on the specified attribute
        """
        return getattr(self.getComponent(componentName), attributeName)

    def getOptimizationSummary(self, modelingClass, outputLevel=0):
        """
        Function which returns an attribute of a component considered in the energy system

        :param modelingClass: name of the modeling class from which the optimization summary should be obtained
        :type modelingClass: string

        :param outputLevel: states the level of detail of the output summary:
            - if equal to 0 if the full optimization summary is returned
            - if equal to 1 rows in which all values are NaN (not a number) are dropped
            - if equal to 2 rows in which all values are NaN or 0 (not a number) are dropped
            |br| * the default value is True
        :type outputLevel: integer (0, 1 or 2)

        :returns: the attribute specified by the attributeName of the component with the name componentName
        :rtype: depends on the specified attribute
        """
        if outputLevel == 0:
            return self._componentModelingDict[modelingClass]._optSummary
        elif outputLevel == 1:
            return self._componentModelingDict[modelingClass]._optSummary.dropna(how='all')
        else:
            df = self._componentModelingDict[modelingClass]._optSummary.dropna(how='all')
            return df.loc[((df != 0) & (~df.isnull())).any(axis=1)]

    def createOutputAsNetCDF(self, output='output.nc', year=2050, freq='H', saveLocationWKT=False, locationSource=None):
        
    ds = nc.Dataset(output, mode='w')
    ds.createDimension('locations', size=len(self._locations))
    ds.createDimension('time', size=len(self._totalTimeSteps))

    ds.createGroup('/operationTimeSeries/')
    for compMod in self._componentModelingDict.keys():
        tsD = ds.createGroup('/operationTimeSeries/{}'.format(compMod[:-8]))
        if compMod == 'StorageModeling': df = self._componentModelingDict[compMod]._stateOfChargeOperationVariablesOptimum
        else: df = self._componentModelingDict[compMod]._operationVariablesOptimum

        if compMod =='TransmissionModeling':colNames = ['{}:{}:{}'.format(i,j,k) for i,j,k in zip(df.index.get_level_values(2), 
                                                                                                  df.index.get_level_values(1),
                                                                                                  df.index.get_level_values(0))]
        else:colNames = ['{}:{}'.format(i,j) for i,j in zip(df.index.get_level_values(1), df.index.get_level_values(0))]

        dimName = '{}_cols'.format(compMod[:-8])
        tsD.createDimension(dimName, size=len(colNames))

        var = tsD.createVariable('{}_operation'.format(compMod[:-8]), 'f', dimensions=('time', dimName,), zlib=True)
        var.description = 'Big table containing operation data for all technologies and regions under {} class'.format(compMod)
        var.longname = 'Some numbers with time series'
        var[:]= df.T.values

        var = tsD.createVariable('{}_cols'.format(compMod[:-8]), str, dimensions=(dimName,))
        var.description = 'Columns of {} class that are zipped to be used in big table of the class'.format(compMod)
        var.longname = '{} columns for big table'.format(compMod)
        for i in range(len(colNames)): var[i]= colNames[i]

        for levelNo in range(len(df.index.levels)):
            var = tsD.createVariable('{}_cols_lev{}'.format(compMod[:-8],levelNo), str, dimensions=(dimName,))
            var.description = 'Level_{} index to be used in multiIndex'.format(levelNo)
            var.longname = 'Level_{} index of big table'.format(levelNo)
            for i in range(len(df.index.get_level_values(levelNo))): var[i]= df.index.get_level_values(levelNo)[i]

    ds.createGroup('/capacityVariables/')
    for compMod in self._componentModelingDict.keys():
        cvD = ds.createGroup('/capacityVariables/{}'.format(compMod[:-8]))
        df = self._componentModelingDict[compMod]._capacityVariablesOptimum

        dil = pd.Series()
        if compMod =='TransmissionModeling':
            for loc1 in df.columns: 
                for tech in df.index.get_level_values(0):
                    for loc2 in df.loc[tech].index:
                        capVar = df.loc[tech].loc[loc2][loc1]
                        if not np.isnan(capVar): dil['{}:{}:{}'.format(loc1,loc2,tech)]=capVar   
        else:
            for loc1 in df.columns: 
                for tech in df.index.get_level_values(0):
                    capVar = df.loc[tech][loc1]
                    if not np.isnan(capVar): dil['{}:{}'.format(loc1,tech)]=capVar
        dil.sort_index(inplace=True)    

        dimName = '{}_capVar_ix'.format(compMod[:-8])
        cvD.createDimension(dimName, size=len(dil.index))
        
        var = cvD.createVariable('{}_capVar'.format(compMod[:-8]), 'f', dimensions=(dimName,), zlib=True)
        var.description = 'Not important'
        var.longname = 'Some numbers with time series'
        var[:]= dil.values

        var = cvD.createVariable('{}_capVar_cols'.format(compMod[:-8]), str, dimensions=(dimName,))
        var.description = 'Not important'
        var.longname = 'Some numbers with time series'
        for i in range(len(dil.index)): var[i]= dil.index[i]

    #this part is shortened as much as possible.Do not try to change it...
    ds.createGroup('/costComponents/')
    for compMod in self._componentModelingDict.keys():
        ccD = ds.createGroup('/costComponents/{}'.format(compMod[:-8]))   

        data = self._componentModelingDict[compMod]._optSummary
        s = data.index.get_level_values(1) == 'TAC'
        df = data.loc[s].sum(level=0)
        df.sort_index(inplace=True) 

        colDim ='{}_TACcols'.format(compMod[:-8])
        indexDim= '{}_TACixs'.format(compMod[:-8])
        ccD.createDimension(colDim, size=len(df.columns))
        ccD.createDimension(indexDim, size=len(df.index))

        var = ccD.createVariable('{}_TAC'.format(compMod[:-8]), 'f', dimensions=(indexDim, colDim,), zlib=True)
        var.description = 'Not important'
        var.longname = 'Some numbers with time series'
        var.unit = '{}/a'.format(self._costUnit)
        var[:]= df.values

        var = ccD .createVariable('{}_TACcols'.format(compMod[:-8]), str, dimensions=(colDim,))
        var.longname = 'Some numbers with time series'
        for i in range(len(df.columns)): var[i]= df.columns[i]

        var = ccD .createVariable('{}_TACixs'.format(compMod[:-8]), str, dimensions=(indexDim,))
        var.longname = 'Some numbers with time series'
        for i in range(len(df.index)): var[i]= df.index[i]

    ds.close()

    def cluster(self, numberOfTypicalPeriods=7, numberOfTimeStepsPerPeriod=24, clusterMethod='hierarchical',
                sortValues=True, storeTSAinstance=False, **kwargs):
        """
        Clusters the time series data of all components considered in the EnergySystemModel instance and then
        stores the clustered data in the respective components. For the clustering itself, the tsam package is
        used (cf. https://github.com/FZJ-IEK3-VSA/tsam).

        **Default arguments:**

        :param numberOfTypicalPeriods: states the number of typical periods into which the time series data
            should be clustered. The number of typical periods multiplied with the number of time steps per
            period must be an integer divisor of the total number of considered time steps in the energy system.
            Note: Please refer to the tsam package documentation of the parameter noTypicalPeriods for more
            information.
            |br| * the default value is 7
        :type numberOfTypicalPeriods: strictly positive integer

        :param numberOfTimeStepsPerPeriod: states the number of time steps per period
            |br| * the default value is 24
        :type numberOfTimeStepsPerPeriod: strictly positive integer

        :param clusterMethod: states the method which is used in the tsam package for clustering the time series
            data. Note: Please refer to the tsam package documentation of the parameter clusterMethod for more
            information.
            |br| * the default value is 'hierarchical'
        :type clusterMethod: string

        :param sortValues: states if the algorithm in the tsam package should use
            (a) the sorted duration curves (-> True) or
            (b) the original profiles (-> False)
            of the time series data within a period for clustering. Note: Please refer to the tsam package
            documentation of the parameter sortValues for more information.
            |br| * the default value is True
        :type clusterMethod: boolean

        :param storeTSAinstance: states if the TimeSeriesAggregation instance create during clustering should be
            stored in the EnergySystemModel instance.
            |br| * the default value is False
        :type storeTSAinstance: boolean

        Moreover, additional keyword arguments for the TimeSeriesAggregation instance can be added (facilitated
        by **kwargs). As an example: it might be useful to add extreme periods to the clustered typical periods
        (cf. https://github.com/FZJ-IEK3-VSA/tsam).

        Last edited: August 10, 2018
        |br| @author: Lara Welder
        """

        # Check input arguments which have to fit the temporal representation of the energy system
        utils.checkClusteringInput(numberOfTypicalPeriods, numberOfTimeStepsPerPeriod, len(self._totalTimeSteps))

        timeStart = time.time()
        print('\nClustering time series data with', numberOfTypicalPeriods, 'typical periods and',
              numberOfTimeStepsPerPeriod, 'time steps per period...')

        # Format data to fit the input requirements of the tsam package:
        # (a) append the time series data from all components stored in all initialized modeling classes to a pandas
        #     data frame with unique column names
        # (b) thereby collect the weights which should be considered for each time series as well in a dictionary
        timeSeriesData, weightDict = [], {}
        for mdlName, mdl in self._componentModelingDict.items():
            for compName, comp in mdl._componentsDict.items():
                compTimeSeriesData, compWeightDict = comp.getDataForTimeSeriesAggregation()
                if compTimeSeriesData is not None:
                    timeSeriesData.append(compTimeSeriesData), weightDict.update(compWeightDict)
        timeSeriesData = pd.concat(timeSeriesData, axis=1)
        timeSeriesData.index = pd.date_range('2050-01-01 00:30:00', periods=len(self._totalTimeSteps),
                                             freq=(str(self._hoursPerTimeStep) + 'H'), tz='Europe/Berlin')

        # Cluster data with tsam package (the reindex_axis call is here for reproducibility of TimeSeriesAggregation
        # call)
        timeSeriesData = timeSeriesData.reindex_axis(sorted(timeSeriesData.columns), axis=1)
        clusterClass = TimeSeriesAggregation(timeSeries=timeSeriesData, noTypicalPeriods=numberOfTypicalPeriods,
                                             hoursPerPeriod=numberOfTimeStepsPerPeriod*self._hoursPerTimeStep,
                                             clusterMethod=clusterMethod, sortValues=sortValues, weightDict=weightDict,
                                             **kwargs)

        # Convert the clustered data to a pandas DataFrame and store the respective clustered time series data in the
        # associated components
        data = pd.DataFrame.from_dict(clusterClass.clusterPeriodDict)
        for mdlName, mdl in self._componentModelingDict.items():
            for compName, comp in mdl._componentsDict.items():
                comp.setAggregatedTimeSeriesData(data)

        # Store time series aggregation parameters in class instance
        if storeTSAinstance:
            self._tsaInstance = clusterClass
        self._typicalPeriods = clusterClass.clusterPeriodIdx
        self._timeStepsPerPeriod = list(range(numberOfTimeStepsPerPeriod))
        self._periods = list(range(int(len(self._totalTimeSteps) / len(self._timeStepsPerPeriod))))
        self._interPeriodTimeSteps = list(range(int(len(self._totalTimeSteps) / len(self._timeStepsPerPeriod)) + 1))
        self._periodsOrder = clusterClass.clusterOrder
        self._periodOccurrences = [(self._periodsOrder == p).sum()/self._numberOfYears for p in self._typicalPeriods]

        # Set cluster flag to true (used to ensure consistently clustered time series data)
        self._isTimeSeriesDataClustered = True
        print("\t\t(%.4f" % (time.time() - timeStart), "sec)\n")

    def optimize(self, timeSeriesAggregation=False, logFileName='job', threads=3, solver='gurobi', timeLimit=None,
                 optimizationSpecs='LogToConsole=1 OptimalityTol=1e-6', warmstart=False):
        """
        Optimizes the specified energy system, for which a pyomo discrete model instance is build and filled
        with
        (0) basic time sets,
        (1) sets, variables and constraints contributed by the component modeling classes,
        (2) basic, component overreaching constraints, and
        (3) an objective function.
        The pyomo instance is then optimized by the specified solver and the optimization results are further
        processed.

        **Default arguments:**

        :param timeSeriesAggregation: states if the optimization of the energy system model should be done with
            (a) the full time series (False) or
            (b) clustered time series data (True).
            If: the argument is True, the time series data was previously clustered, and no further tsamSpecs
            are declared, the clustered time series data from the last cluster function call is used. Otherwise
            the time series data is clustered within the optimize function, using, if specified, the tsamSpecs
            argument.
            |br| * the default value is False
        :type timeSeriesAggregation: boolean

        :param logFileName: logFileName is used for naming the log file of the optimization solver output.
            If the logFileName is given as an absolute path (i.e. logFileName = os.path.join(os.getcwd(),
            'Results', 'logFileName.txt')) the log file will be stored in the specified directory. Otherwise
            it will be by default stored in the directory where the executing python script is called.
            |br| * the default value is 'job'
        :type logFileName: string

        :param threads: number of computational threads used for solving the optimization (solver dependent
            input). If gurobi is selected as the solver: a value of 0 results in using all available threads. If
            a value larger than the available number of threads are chosen, the value will reset to the maximum
            number of threads.
            |br| * the default value is 3
        :type threads: positive integer

        :param solver: specifies which solver should solve the optimization problem (which of course has to be
            installed on the machine on which the model is run).
            |br| * the default value is 'gurobi'
        :type solver: string

        :param timeLimit: if not specified as None, indicates the maximum solve time of the optimization problem
            in seconds (solver dependent input). The use of this parameter is suggested when running models in
            runtime restricted environments (such as clusters with job submission systems). If the runtime
            limitation is triggered before an optimal solution is available, the best solution obtained up
            until then (if available) is processed.
            |br| * None
        :type timeLimit: strictly positive integer or None

        :param optimizationSpecs: specifies parameters for the optimization solver (see the respective solver
            documentation for more information)
            |br| * 'LogToConsole=1 OptimalityTol=1e-6'
        :type timeLimit: string

        :param warmstart: specifies if a warm start of the optimization should be considered
            (not supported by all solvers).
            |br| * the default value is False
        :type warmstart: boolean

        Last edited: August 10, 2018
        |br| @author: Lara Welder
        """
        # Get starting time of the optimization to later on obtain the total run time of the optimize function call
        timeStart = time.time()

        # Check correctness of inputs
        # Check input arguments which have to fit the temporal representation of the energy system
        utils.checkOptimizeInput(timeSeriesAggregation, self._isTimeSeriesDataClustered, logFileName, threads, solver,
                                 timeLimit, optimizationSpecs, warmstart)

        # Store keyword arguments in the EnergySystemModel instance
        self._solverSpecs['logFileName'], self._solverSpecs['threads'] = logFileName, threads
        self._solverSpecs['solver'], self._solverSpecs['timeLimit'] = solver, timeLimit
        self._solverSpecs['optimizationSpecs'], self._solverSpecs['hasTSA'] = optimizationSpecs, timeSeriesAggregation

        ################################################################################################################
        #                           Initialize mathematical model (ConcreteModel) instance                             #
        ################################################################################################################

        # Initialize a pyomo ConcreteModel which will be used to store the mathematical formulation of the model.
        # The ConcreteModel instance is stored in the EnergySystemModel instance, which makes it available for
        # post-processing or debugging. A pyomo Suffix with the name dual is declared to make dual values associated
        # to the model's constraints available after optimization.
        self._pyM = pyomo.ConcreteModel()
        pyM = self._pyM
        pyM.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT)

        ################################################################################################################
        #                              Set and initialize basic time parameters and sets                               #
        ################################################################################################################

        # Store the information if aggregated time series data is considered for modeling the energy system in the pyomo
        # model instance and set the time series which is again considered for modeling in all components accordingly
        pyM.hasTSA = timeSeriesAggregation
        for mdl in self._componentModelingDict.values():
            for comp in mdl._componentsDict.values():
                comp.setTimeSeriesData(pyM.hasTSA)

        # Set the time set and the inter time steps set. The time set is a set of tuples. A tuple consists of two
        # entries, the first one indicating an index of a period and the second one indicating a time step inside that
        # period. If time series aggregation is not considered only one period (period 0) exists and the time steps
        # range from 0 up until the specified number of total time steps - 1. Otherwise the time set is initialized for
        # each typical period (0 ... numberOfTypicalPeriods-1) and the number of time steps per period (0 ...
        # numberOfTimeStepsPerPeriod-1).
        # The inter time steps set is a set of tuples as well, which again consist of two values. The first value again
        # indicates the period, however the second one now refers to a point in time right before or after a time step
        # (or between two time steps). Hence, the second value reaches values from (0 ... numberOfTimeStepsPerPeriod).
        if not pyM.hasTSA:
            # Reset timeStepsPerPeriod in case it was overwritten by the clustering function
            self._timeStepsPerPeriod = self._totalTimeSteps
            self._interPeriodTimeSteps = list(range(int(len(self._totalTimeSteps) /
                                                        len(self._timeStepsPerPeriod)) + 1))
            self._periods = [0]
            self._periodsOrder = [0]
            self._periodOccurrences = [1]

            # Define sets
            def initTimeSet(pyM):
                return ((p, t) for p in self._periods for t in self._timeStepsPerPeriod)

            def initInterTimeStepsSet(pyM):
                return ((p, t) for p in self._periods for t in range(len(self._timeStepsPerPeriod) + 1))
        else:
            print('Time series aggregation specifications:\nNumber of typical periods:', len(self._typicalPeriods),
                  ', number of time steps per periods:', len(self._timeStepsPerPeriod))

            # Define sets
            def initTimeSet(pyM):
                return ((p, t) for p in self._typicalPeriods for t in self._timeStepsPerPeriod)

            def initInterTimeStepsSet(pyM):
                return ((p, t) for p in self._typicalPeriods for t in range(len(self._timeStepsPerPeriod) + 1))

        # Initialize sets
        pyM.timeSet = pyomo.Set(dimen=2, initialize=initTimeSet)
        pyM.interTimeStepsSet = pyomo.Set(dimen=2, initialize=initInterTimeStepsSet)

        ################################################################################################################
        #                         Declare component specific sets, variables and constraints                           #
        ################################################################################################################

        for key, mdl in self._componentModelingDict.items():
            _t = time.time()
            print('Declaring sets, variables and constraints for', key)
            print('\tdeclaring sets... '), mdl.declareSets(self, pyM)
            print('\tdeclaring variables... '), mdl.declareVariables(self, pyM),
            print('\tdeclaring constraints... '), mdl.declareComponentConstraints(self, pyM)
            print("\t\t(%.4f" % (time.time() - _t), "sec)")

        ################################################################################################################
        #                              Declare cross-componential sets and constraints                                 #
        ################################################################################################################

        _t = time.time()

        # Declare shared potential constraints, i.e. if a maximum potential of salt caverns has to be shared by
        # salt caverns storing methane and salt caverns storing hydrogen.
        print('Declaring shared potential constraint...')

        # Create shared potential dictionary (maps a shared potential ID and a location to components who share the
        # potential)
        potentialDict = {}
        for mdl in self._componentModelingDict.values():
            for compName, comp in mdl._componentsDict.items():
                if comp._sharedPotentialID is not None:
                    [potentialDict.setdefault((comp._sharedPotentialID, loc), []).append(compName)
                     for loc in self._locations if comp._capacityMax[loc] != 0]
        pyM.sharedPotentialDict = potentialDict

        # Define and initialize constraints for each instance and location where components have to share an available
        # potential. Sum up the relative contributions to the shared potential and ensure that the total share is
        # <= 100%. For this, get the contributions to the shared potential for the corresponding ID and
        # location from each modeling class.
        def sharedPotentialConstraint(pyM, ID, loc):
            return sum(mdl.getSharedPotentialContribution(pyM, ID, loc)
                       for mdl in self._componentModelingDict.values()) <= 1
        pyM.ConstraintSharedPotentials = \
            pyomo.Constraint(pyM.sharedPotentialDict.keys(), rule=sharedPotentialConstraint)
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        _t = time.time()

        # Declare commodity balance constraints (balance constraint for each component, location and time step)
        print('Declaring commodity balances...')

        # Declare and initialize a set that states for which location and commodity the commodity balance constraints
        # are non trivial (i.e. not 0 == 0; trivial constraints raise errors in pyomo).
        def initLocationCommoditySet(pyM):
            return ((loc, commod) for loc in self._locations for commod in self._commodities
                    if any([mdl.hasOpVariablesForLocationCommodity(self, loc, commod)
                            for mdl in self._componentModelingDict.values()]))
        pyM.locationCommoditySet = pyomo.Set(dimen=2, initialize=initLocationCommoditySet)

        # Declare and initialize commodity balance constraints by checking for each location and commodity in the
        # locationCommoditySet and for each period and time step within the period if the commodity source and sink
        # terms add up to zero. For this, get the contribution to commodity balance from each modeling class
        def commodityBalanceConstraint(pyM, loc, commod, p, t):
            return sum(mdl.getCommodityBalanceContribution(pyM, commod, loc, p, t)
                       for mdl in self._componentModelingDict.values()) == 0
        pyM.commodityBalanceConstraint = pyomo.Constraint(pyM.locationCommoditySet, pyM.timeSet,
                                                          rule=commodityBalanceConstraint)
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        ################################################################################################################
        #                                         Declare objective function                                           #
        ################################################################################################################

        _t = time.time()

        print('Declaring objective function...')

        # Declare objective function by obtaining the contributions to the objective function from all modeling classes
        # Currently, the only objective function which can be selected is the sum of the total annual cost of all
        # components.
        def objective(pyM):
            TAC = sum(mdl.getObjectiveFunctionContribution(self, pyM) for mdl in self._componentModelingDict.values())
            return TAC
        pyM.Obj = pyomo.Objective(rule=objective)
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        ################################################################################################################
        #                                  Solve the specified optimization problem                                    #
        ################################################################################################################

        _t = time.time()

        # Set which solver should solve the specified optimization problem
        optimizer = opt.SolverFactory(solver)

        # Set, if specified, the time limit
        if self._solverSpecs['timeLimit'] is not None:
            optimizer.options['timelimit'] = timeLimit

        # Set the specified solver options
        optimizer.set_options('Threads=' + str(threads) + ' logfile=' + logFileName + ' ' + optimizationSpecs)

        # Solve optimization problem. The optimization output is stored in the pyM model and the solver information
        # is printed.
        solver_info = optimizer.solve(pyM, warmstart=warmstart, tee=True)
        print(solver_info.solver())
        print(solver_info.problem())
        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        ################################################################################################################
        #                                      Post-process optimization output                                        #
        ################################################################################################################

        _t = time.time()

        # Post-process the optimization output by differentiating between different solver statuses and termination
        # conditions. First, check if the model is infeasible or unbounded. In this case, no output is generated.
        status, termCondition = solver_info.solver.status, solver_info.solver.termination_condition
        if status == opt.SolverStatus.error or status == opt.SolverStatus.aborted or status == opt.SolverStatus.unknown:
            print('Solver status:  ' + str(status) + ', termination condition:  ' + str(termCondition) +
                  '. No output is generated.')
        else:
            # Otherwise the
            if solver_info.solver.termination_condition == opt.TerminationCondition.infeasibleOrUnbounded or \
               solver_info.solver.termination_condition == opt.TerminationCondition.infeasible or \
               solver_info.solver.termination_condition == opt.TerminationCondition.unbounded:
                print('Optimization problem is ' +
                      str(solver_info.solver.termination_condition) + '. No output is generated.')
            else:
                # if the solver status is not okay (hence either has a warning, an error, was aborted or has an unknown
                # status)
                if not solver_info.solver.termination_condition == opt.TerminationCondition.optimal:
                    warnings.warn('Output is generated for a non-optimal solution.')
                print("Processing optimization output...")
                # Declare component specific sets, variables and constraints
                for key, mdl in self._componentModelingDict.items():
                    __t = time.time()
                    mdl.setOptimalValues(self, pyM)
                    print('\tfor', key, '...', "\t(%.4f" % (time.time() - __t), "sec)")

        print("\t\t(%.4f" % (time.time() - _t), "sec)")

        # Store the runtime of the optimize function call in the EnergySystemModel instance
        self._solverSpecs['runtime'] = time.time() - timeStart
