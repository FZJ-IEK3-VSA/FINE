import fine as fn
from fine import utils
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
import copy
from pathlib import Path


def rollingHorizonOptimization(
    esM,
    scenario_name,
    resultExportPath,
    numberOfInvestmentPeriodsForRollingHorizon,
    timeSeriesAggregation=True,
    numberOfTypicalPeriods=7,
    numberOfTimeStepsPerPeriod=24,
    numberOfSegments=16,
    clusterMethod="hierarchical",
    solver="gurobi",
    optimizationSpecs="",
):
    """If numberOfInvestmentPeriodsForRollingHorizon == numberOfInvestmentPeriods -> Perfect Foresight, If numberOfInvestmentPeriodsForRollingHorizon == 1 -> Foresight, else Rolling Horizon."""
    if esM.rollingHorizonStartYear is None:
        esM.rollingHorizonStartYear = esM.startYear

    # checks for data input
    if esM.numberOfInvestmentPeriods < 2:
        raise ValueError("At least two investmentperiods required for rolling horizon.")
    if esM.numberOfInvestmentPeriods <= numberOfInvestmentPeriodsForRollingHorizon:
        raise ValueError(
            "There must be at least one more investment period in the "
            "transformation pathway than in the rolling horizon interval"
        )

    utils.isStrictlyPositiveInt(numberOfInvestmentPeriodsForRollingHorizon)
    utils.isStrictlyPositiveNumber(numberOfInvestmentPeriodsForRollingHorizon)

    # 0. set up rolling horizon intervals
    interval = esM.investmentPeriodInterval
    rollingHorizonIntervals = [
        list(
            range(
                start,
                start + interval * numberOfInvestmentPeriodsForRollingHorizon,
                interval,
            )
        )
        for start in esM.investmentPeriodNames
        if start + interval * (numberOfInvestmentPeriodsForRollingHorizon - 1)
        in esM.investmentPeriodNames
    ]

    # extract all information of original esM
    esmDict, compDict = fn.dictIO.exportToDict(esM)

    print("Starting rolling horizon optimization.")

    esM_results = {}
    persistedStock = {
        classname: {
            comp: copy.deepcopy(compDict[classname][comp]["stockCommissioning"])
            for comp in compDict[classname]
        }
        for classname in compDict
    }
    for rollingHorizonYears in rollingHorizonIntervals:
        print(
            f"Initizializing rolling horizon optimization for {rollingHorizonYears}..."
        )
        # 1. Analyse components and create dicts for adding them
        rollingHorizonCompDict = copy.deepcopy(dict(compDict))
        for classname in rollingHorizonCompDict:
            for comp in rollingHorizonCompDict[classname]:
                # restore accumulated stock from previous iterations
                rollingHorizonCompDict[classname][comp]["stockCommissioning"] = (
                    copy.deepcopy(persistedStock[classname][comp])
                )
                # 1.1 update stockCommissioning
                # first rolling horizon requires no changes, just external stock,
                # further years needs internal optimization results
                if rollingHorizonYears != rollingHorizonIntervals[0]:
                    # get previous year
                    previousYear = rollingHorizonYears[0] - interval
                    # get model class from previous rolling horizon optimization
                    mdl_class = esM_results[previousYear].componentNames[comp]
                    # get commissioning results of previous rolling horizon
                    previousCommissioning = (
                        esM_results[previousYear]
                        .getOptimizationSummary(mdl_class, ip=previousYear)
                        .loc[comp, "commissioning"]
                    )
                    previousCommissioningLocation = previousCommissioning.loc[
                        previousCommissioning.index[0]
                    ].T

                    # add commissioning of previous runs as stock, if there was commissioning
                    if round(previousCommissioningLocation.sum(), 5) > 0:
                        # a) if no stock previously existed, create new structure
                        if (
                            rollingHorizonCompDict[classname][comp][
                                "stockCommissioning"
                            ]
                            is None
                        ):
                            rollingHorizonCompDict[classname][comp][
                                "stockCommissioning"
                            ] = {}
                            rollingHorizonCompDict[classname][comp][
                                "stockCommissioning"
                            ][previousYear] = previousCommissioningLocation
                        # b) else add to structure
                        else:
                            rollingHorizonCompDict[classname][comp][
                                "stockCommissioning"
                            ][previousYear] = previousCommissioningLocation

                        # c) delete "too" old stock, as it will make
                        # problems with setup of parameters otherwise
                        stockData = rollingHorizonCompDict[classname][comp][
                            "stockCommissioning"
                        ]
                        technicalLifetime = rollingHorizonCompDict[classname][comp][
                            "technicalLifetime"
                        ]
                        outdatedStockYears = [
                            x
                            for x in stockData.keys()
                            if x < rollingHorizonYears[0] - technicalLifetime.max()
                        ]
                        for outdatedStockYear in outdatedStockYears:
                            rollingHorizonCompDict[classname][comp][
                                "stockCommissioning"
                            ].pop(outdatedStockYear)

                # persist updated stock for next iteration
                persistedStock[classname][comp] = copy.deepcopy(
                    rollingHorizonCompDict[classname][comp]["stockCommissioning"]
                )

                if (
                    rollingHorizonCompDict[classname][comp]["stockCommissioning"]
                    is not None
                ):
                    stockYears = list(
                        rollingHorizonCompDict[classname][comp][
                            "stockCommissioning"
                        ].keys()
                    )
                else:
                    stockYears = []

                # get data for rolling horizon years from perfect foresight model
                for parameter_name, parameter_value in rollingHorizonCompDict[
                    classname
                ][comp].items():
                    # stock commissioning
                    if parameter_name == "stockCommissioning":
                        continue
                    # 1.2 commodity conversion factors
                    if parameter_name == "commodityConversionFactors":
                        firstKey = list(parameter_value.keys())[0]
                        # check for ip dependendy
                        if firstKey in esM.investmentPeriodNames:
                            # filter for years of rolling horizon time frame
                            new_parameter_value = {
                                key: value
                                for (key, value) in parameter_value.items()
                                if key in rollingHorizonYears
                            }
                            rollingHorizonCompDict[classname][comp][parameter_name] = (
                                new_parameter_value
                            )
                        # check for (commis, ip dependency)
                        elif isinstance(firstKey, tuple):
                            # filter for correct operation years
                            _new_parameter_value = {
                                (commisYear, opYear): value
                                for (
                                    (commisYear, opYear),
                                    value,
                                ) in parameter_value.items()
                                if opYear in rollingHorizonYears
                            }
                            # filter out years before the modelyears
                            # without commissioning
                            new_parameter_value = _new_parameter_value.copy()
                            for commisYear, opYear in _new_parameter_value.keys():
                                if (
                                    commisYear < rollingHorizonYears[0]
                                    and commisYear not in stockYears
                                ):
                                    new_parameter_value.pop((commisYear, opYear))

                            rollingHorizonCompDict[classname][comp][parameter_name] = (
                                new_parameter_value
                            )
                        else:
                            pass
                    # 1.3 other parameter which are yearly dependent
                    elif isinstance(parameter_value, dict):
                        if "PerOperation" in parameter_name:
                            relevantYears = rollingHorizonYears
                        else:
                            relevantYears = rollingHorizonYears + stockYears
                        # filter for years of rolling horizon time frame
                        new_parameter_value = {
                            _year: value
                            for (_year, value) in parameter_value.items()
                            if _year in relevantYears
                        }
                        rollingHorizonCompDict[classname][comp][parameter_name] = (
                            new_parameter_value
                        )
                    # 1.4 other parameters, which do not change over time
                    else:
                        pass

        # 2. init esm with new startYear and numberOfInvestmentPeriods and add components
        rollingHorizonEsmDict = esmDict.copy()
        rollingHorizonEsmDict["startYear"] = rollingHorizonYears[0]
        rollingHorizonEsmDict["numberOfInvestmentPeriods"] = (
            numberOfInvestmentPeriodsForRollingHorizon
        )
        for param, value in rollingHorizonEsmDict.items():
            if (
                isinstance(value, dict)
                and list(value.keys()) == esM.investmentPeriodNames
            ):
                rollingHorizonEsmDict[param] = {
                    _year: _value
                    for (_year, _value) in value.items()
                    if _year in rollingHorizonYears
                }
        rollingHorizonEsm = fn.EnergySystemModel(**rollingHorizonEsmDict)
        # add components per class
        for classname in rollingHorizonCompDict:
            for comp in rollingHorizonCompDict[classname]:
                rollingHorizonEsm.add(
                    getattr(fn, classname)(
                        esM=rollingHorizonEsm,
                        **rollingHorizonCompDict[classname][
                            comp
                        ],  # information of component
                    )
                )

        # 3. optimize the rolling horizon esM
        if timeSeriesAggregation:
            rollingHorizonEsm.aggregateTemporally(
                numberOfTypicalPeriods=numberOfTypicalPeriods,
                numberOfTimeStepsPerPeriod=numberOfTimeStepsPerPeriod,
                numberOfSegmentsPerPeriod=numberOfSegments,
                segmentation=True,
                clusterMethod=clusterMethod,
                solver=solver,
                sortValues=True,
                rescaleClusterPeriods=True,
                representationMethod=None,
            )

        rollingHorizonEsm.optimize(
            declaresOptimizationProblem=True,
            timeSeriesAggregation=timeSeriesAggregation,
            solver=solver,
            optimizationSpecs=optimizationSpecs,
        )

        # 4. export optimization summaries
        # For all optimization than the last rolling horizon optimization, export only first year.
        # For the last rolling horizon interval, export all years.
        if rollingHorizonYears != rollingHorizonIntervals[-1]:
            exportYears = [rollingHorizonYears[0]]
        else:
            exportYears = rollingHorizonYears

        for year in exportYears:
            writeOptimizationOutputToExcel(
                rollingHorizonEsm,
                outputFileName=str(
                    Path(resultExportPath) / f"{scenario_name}_rollingHorizon"
                ),
                optSumOutputLevel={
                    "SourceSinkModel": 0,
                    "ConversionModel": 0,
                    "StorageModel": 0,
                    "TransmissionModel": 0,
                    "LOPFModel": 0,
                },
                optValOutputLevel={
                    "SourceSinkModel": 0,
                    "ConversionModel": 0,
                    "StorageModel": 0,
                    "TransmissionModel": 0,
                    "LOPFModel": 0,
                },
                investmentPeriod=year,
            )

        # 5. save esM's
        esM_results[rollingHorizonYears[0]] = rollingHorizonEsm

        print(f"Finished rolling horizon optimization for {rollingHorizonYears}")

    print("Finished rolling horizon optimization.")
    return esM_results
