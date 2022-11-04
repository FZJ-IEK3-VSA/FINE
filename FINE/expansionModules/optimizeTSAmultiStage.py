from FINE import utils
import FINE as fn
import numpy as np


def optimizeTSAmultiStage(
    esM,
    declaresOptimizationProblem=True,
    relaxIsBuiltBinary=False,
    numberOfTypicalPeriods=30,
    numberOfTimeStepsPerPeriod=24,
    clusterMethod="hierarchical",
    logFileName="",
    threads=3,
    solver="gurobi",
    timeLimit=None,
    optimizationSpecs="",
    warmstart=False,
):
    """
    Call the optimize function for a temporally aggregated MILP (so the model has to include
    hasIsBuiltBinaryVariables in all or some components). Fix the binary variables and run it again
    without temporal aggregation. Furthermore, a LP with relaxed binary variables can be solved to
    obtain both, an upper and lower bound for the fully resolved MILP.

    **Required arguments:**

    :param esM: energy system model to which the component should be added. Used for unit checks.
    :type esM: EnergySystemModel instance from the FINE package

    **Default arguments:**

    :param declaresOptimizationProblem: states if the optimization problem should be declared (True) or not (False).

        (a) If true, the declareOptimizationProblem function is called and a pyomo ConcreteModel instance is built.
        (b) If false a previously declared pyomo ConcreteModel instance is used.

        |br| * the default value is True
    :type declaresOptimizationProblem: boolean

    :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
        bound of the problem.
        |br| * the default value is False
    :type declaresOptimizationProblem: boolean

    :param numberOfTypicalPeriods: states the number of typical periods into which the time series data
        should be clustered. The number of time steps per period must be an integer multiple of the total
        number of considered time steps in the energy system.

        .. note::
            Please refer to the tsam package documentation of the parameter noTypicalPeriods for more
            information.

        |br| * the default value is 30
    :type numberOfTypicalPeriods: strictly positive integer

    :param numberOfTimeStepsPerPeriod: states the number of time steps per period
        |br| * the default value is 24
    :type numberOfTimeStepsPerPeriod: strictly positive integer

    :param clusterMethod: states the method which is used in the tsam package for clustering the time series
        data. Options are for example 'averaging','k_means','exact k_medoid' or 'hierarchical'.

        .. note::
            Please refer to the tsam package documentation of the parameter clusterMethod for more information.

        |br| * the default value is 'hierarchical'
    :type clusterMethod: string

    :param logFileName: logFileName is used for naming the log file of the optimization solver output
        if gurobi is used as the optimization solver.
        If the logFileName is given as an absolute path (e.g. logFileName = os.path.join(os.getcwd(),
        'Results', 'logFileName.txt')) the log file will be stored in the specified directory. Otherwise,
        it will be stored by default in the directory where the executing python script is called.
        |br| * the default value is 'job'
    :type logFileName: string

    :param threads: number of computational threads used for solving the optimization (solver dependent
        input) if gurobi is used as the solver. A value of 0 results in using all available threads. If
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
        |br| * the default value is None
    :type timeLimit: strictly positive integer or None

    :param optimizationSpecs: specifies parameters for the optimization solver (see the respective solver
        documentation for more information). Example: 'LogToConsole=1 OptimalityTol=1e-6'
        |br| * the default value is an empty string ('')
    :type timeLimit: string

    :param warmstart: specifies if a warm start of the optimization should be considered
        (not always supported by the solvers).
        |br| * the default value is False
    :type warmstart: boolean
    """
    lowerBound = None

    if relaxIsBuiltBinary:
        esM.optimize(
            declaresOptimizationProblem=True,
            timeSeriesAggregation=False,
            relaxIsBuiltBinary=True,
            logFileName="relaxedProblem",
            threads=threads,
            solver=solver,
            timeLimit=timeLimit,
            optimizationSpecs=optimizationSpecs,
            warmstart=warmstart,
        )
        lowerBound = esM.objectiveValue

    esM.aggregateTemporally(
        numberOfTypicalPeriods=numberOfTypicalPeriods,
        numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod,
        segmentation=False,
        clusterMethod=clusterMethod,
        solver=solver,
        sortValues=True,
        rescaleClusterPeriods=True,
        representationMethod=None,
    )

    esM.optimize(
        declaresOptimizationProblem=True,
        timeSeriesAggregation=True,
        relaxIsBuiltBinary=False,
        logFileName="firstStage",
        threads=threads,
        solver=solver,
        timeLimit=timeLimit,
        optimizationSpecs=optimizationSpecs,
        warmstart=warmstart,
    )

    # Set the binary variables to the values resulting from the first optimization step
    fn.fixBinaryVariables(esM)

    esM.optimize(
        declaresOptimizationProblem=True,
        timeSeriesAggregation=False,
        relaxIsBuiltBinary=False,
        logFileName="secondStage",
        threads=threads,
        solver=solver,
        timeLimit=timeLimit,
        optimizationSpecs=optimizationSpecs,
        warmstart=False,
    )
    upperBound = esM.objectiveValue

    if lowerBound is not None:
        delta = upperBound - lowerBound
        gap = delta / upperBound
        esM.lowerBound, esM.upperBound = lowerBound, upperBound
        esM.gap = gap
        print(
            "The real optimal value lies between "
            + str(round(lowerBound, 2))
            + " and "
            + str(round(upperBound, 2))
            + " with a gap of "
            + str(round(gap * 100, 2))
            + "%."
        )


def fixBinaryVariables(esM):
    """
    Search for the optimized binary variables and set them as fixed.

    :param esM: energy system model to which the component should be added. Used for unit checks.
    :type esM: EnergySystemModel instance from the FINE package
    """
    for mdl in esM.componentModelingDict.keys():
        compValues = esM.componentModelingDict[mdl].getOptimalValues(
            name="isBuiltVariablesOptimum", ip=0
        )["values"]
        if compValues is not None:
            for comp in compValues.index.get_level_values(0).unique():
                values = utils.preprocess2dimData(
                    compValues.loc[comp]
                    .fillna(value=-1)
                    .round(decimals=0)
                    .astype(np.int64),
                    discard=False,
                )
                esM.componentModelingDict[mdl].componentsDict[comp].isBuiltFix = values
