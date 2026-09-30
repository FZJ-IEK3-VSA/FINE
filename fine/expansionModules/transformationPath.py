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
    """Raise NotImplementedError. optimizeSimpleMyopic has been replaced by
    fine.expansionModules.rollingHorizon.rollingHorizonOptimization with
    numberOfInvestmentPeriodsForRollingHorizon=1, which covers the same
    myopic foresight use case (see issue #640).

    :raises NotImplementedError: always. Use rollingHorizonOptimization instead.
    """
    raise NotImplementedError(
        "optimizeSimpleMyopic has been removed and is no longer maintained. "
        "Use fine.expansionModules.rollingHorizon.rollingHorizonOptimization with "
        "numberOfInvestmentPeriodsForRollingHorizon=1 instead, which covers the same "
        "myopic foresight use case (see issue #640). Note that CO2ReductionTargets are "
        "not supported there anymore; express them as a per-investment-period "
        "balanceLimit passed to rollingHorizonOptimization instead."
    )
