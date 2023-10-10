from FINE import utils
from FINE.IOManagement import standardIO
import pandas as pd
import copy


def optimizeSimpleMyopic(
    esM,
    startYear,
    endYear=None,
    nbOfSteps=None,
    nbOfRepresentedYears=None,
    timeSeriesAggregation=True,
    numberOfTypicalPeriods=7,
    numberOfTimeStepsPerPeriod=24,
    clusterMethod="hierarchical",
    logFileName="",
    threads=3,
    solver="gurobi",
    timeLimit=None,
    optimizationSpecs="",
    warmstart=False,
    CO2Reference=366,
    CO2ReductionTargets=None,
    saveResults=True,
    trackESMs=True,
):
    """
    Optimization function for myopic approach. For each optimization run, the newly installed capacities
    will be given as a stock (with capacityFix) to the next optimization run.

    :param esM: EnergySystemModel instance representing the energy system which should be optimized by considering the
                transformation pathway (myopic foresight).
    :type esM: esM - EnergySystemModel instance

    :param startYear: year of the first optimization
    :type startYear: int

    **Default arguments:**

    :param endYear: year of the last optimization
    :type endYear: int

    :param nbOfSteps: number of optimization runs excluding the start year (minimum number
        of optimization runs is 2: one optimization for the start year and one for the end year).
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

    :param clusterMethod: states the method which is used in the tsam package for clustering the time series
        data. Options are for example 'averaging','k_means','exact k_medoid' or 'hierarchical'.

        .. note::
            Please refer to the tsam package documentation of the parameter clusterMethod for more information.

        |br| * the default value is 'hierarchical'
    :type clusterMethod: string

    :param CO2Reference: gives the reference value of the CO2 emission to which the reduction should be applied to.
        The default value refers to the emissions of 1990 within the electricity sector (366kt CO2_eq)
        |br| * the default value is 366
    :type CO2Reference: float

    :param CO2ReductionTargets: specifies the CO2 reduction targets for all optimization periods.
        If specified, the length of the list must equal the number of optimization steps, and an object of the sink class
        which counts the CO2 emission is required.
        |br| * the default value is None
    :type CO2ReductionTargets: list of strictly positive integer or None

    :param saveResults: specifies if the results are saves in excelfiles or not.
        |br| * the default value is True
    :type saveResults: boolean

    :param trackESMs: specifies if the energy system model instances of each model run should be stored in a dictionary or not.
        It´s not recommended to track the ESMs if the model is quite big.
        |br| * the default value is True
    :type trackESMs: boolean

    :returns: myopicResults: Store all optimization outputs in a dictionary for further analyses. If trackESMs is set to false,
        nothing is returned.
    :rtype: dict of all optimized instances of the EnergySystemModel class or None.
    """
    if esM.numberOfInvestmentPeriods != 1:
        raise ValueError(
            "Myopic is based on single year optimizations. "
            + "numberOfInvestmentPeriods must be 1"
        )
    nbOfSteps, nbOfRepresentedYears = utils.checkAndSetTimeHorizon(
        startYear, endYear, nbOfSteps, nbOfRepresentedYears
    )
    utils.checkSinkCompCO2toEnvironment(esM, CO2ReductionTargets)
    utils.checkCO2ReductionTargets(CO2ReductionTargets, nbOfSteps)
    print("Number of optimization runs: ", nbOfSteps + 1)
    print("Number of years represented by one optimization: ", nbOfRepresentedYears)
    mileStoneYear = startYear
    if trackESMs:
        myopicResults = dict()

    for step in range(0, nbOfSteps + 1):
        mileStoneYear = startYear + step * nbOfRepresentedYears
        logFileName = "log_" + str(mileStoneYear)
        utils.setNewCO2ReductionTarget(esM, CO2Reference, CO2ReductionTargets, step)

        # Optimization
        if timeSeriesAggregation:
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
            timeSeriesAggregation=timeSeriesAggregation,
            logFileName=logFileName,
            threads=threads,
            solver=solver,
            timeLimit=timeLimit,
            optimizationSpecs=optimizationSpecs,
            warmstart=False,
        )

        if saveResults:
            standardIO.writeOptimizationOutputToExcel(
                esM,
                outputFileName="ESM" + str(mileStoneYear),
                optSumOutputLevel=2,
                optValOutputLevel=1,
            )

        if trackESMs:
            tmp = esM
            del (
                tmp.pyM
            )  # Delete pyomo instance from esM before copying it (pyomo instances cannot be copied)
            myopicResults.update({"ESM_" + str(mileStoneYear): copy.deepcopy(tmp)})

        # Get stock if not all optimizations are done
        if step != nbOfSteps + 1:
            esM = getStock(esM, mileStoneYear, nbOfRepresentedYears)
    if trackESMs:
        return myopicResults
    else:
        return None


def getStock(esM, mileStoneYear, nbOfRepresentedYears):
    """
    Function for determining the stock of all considered technologies for the next optimization period.
    If the technical lifetime is expired, the fixed capacities of the concerned components are set to 0.

    :param mileStoneYear: Last year of the optimization period
    :type mileStoneYear: int

    :param nbOfRepresentativeYears: Number of years within one optimization period.
    :type nbOfRepresentativeYears: int

    :return: EnergySystemModel instance including the installed capacities of the previous optimization runs.
    :rtype: EnergySystemModel instance
    """
    for mdl in esM.componentModelingDict.keys():
        compValues = esM.componentModelingDict[mdl].getOptimalValues(
            "capacityVariablesOptimum", ip=0
        )["values"]
        if compValues is not None:
            for comp in compValues.index.get_level_values(0).unique():
                if (
                    "stock"
                    not in esM.componentModelingDict[mdl].componentsDict[comp].name
                ):
                    stockName = comp + "_stock" + "_" + str(mileStoneYear)
                    stockComp = copy.deepcopy(
                        esM.componentModelingDict[mdl].componentsDict[comp]
                    )
                    stockComp.name = stockName
                    stockComp.lifetime = (
                        esM.componentModelingDict[mdl]
                        .componentsDict[comp]
                        .technicalLifetime
                        - nbOfRepresentedYears
                    )
                    # If lifetime is shorter than number of represented years, skip component
                    if any(getattr(stockComp, "lifetime") <= 0):
                        continue

                    # If capacities are installed, set the values as capacityFix.
                    if getattr(stockComp, "capacityFix") is None:
                        if isinstance(compValues.loc[comp], pd.DataFrame):
                            stockComp.processedCapacityFix = {}
                            stockComp.processedCapacityFix[
                                0
                            ] = utils.preprocess2dimData(
                                compValues.loc[comp].fillna(value=-1), discard=False
                            )
                        else:
                            # NOTE: Values of capacityMin and capacityMax are not overwritten.
                            # CapacityFix values set the capacity fix and fulfills the boundary constraints (capacityMin <= capacityFix <= capacityMax)
                            stockComp.processedCapacityFix = {}
                            stockComp.processedCapacityFix[0] = compValues.loc[comp]
                    esM.add(stockComp)

                elif (
                    "stock" in esM.componentModelingDict[mdl].componentsDict[comp].name
                ):
                    esM.componentModelingDict[mdl].componentsDict[
                        comp
                    ].lifetime -= nbOfRepresentedYears
                    # If lifetime is exceeded, remove component from the energySystemModel instance
                    if any(
                        getattr(
                            esM.componentModelingDict[mdl].componentsDict[comp],
                            "lifetime",
                        )
                        <= 0
                    ):
                        esM.removeComponent(comp)

    return esM
