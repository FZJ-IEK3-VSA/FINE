import warnings

import pandas as pd
from netCDF4 import Dataset

import fine as fn
from fine import utils
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from fine.IOManagement.xarrayIO import (
    writeEnergySystemModelToNetCDF,
    readNetCDFtoEnergySystemModel,
)
import copy
from pathlib import Path


def _cachedGroupExists(netCDFPath, groupPrefix):
    """Check whether a given interval's group is already present in the
    shared rolling horizon netCDF file, without loading it.
    """
    if not netCDFPath.is_file():
        return False
    with Dataset(str(netCDFPath), "r", format="NETCDF4") as rootgrp:
        return groupPrefix in rootgrp.groups


def _cachedIntervalConfigMismatches(
    cachedEsm, rollingHorizonYears, numberOfInvestmentPeriodsForRollingHorizon
):
    """Check a cached interval's own configuration against what this call
    explicitly asked for. A mismatch here (e.g. numberOfInvestmentPeriods
    ForRollingHorizon changed between the interrupted and the resumed run)
    is almost certainly a user error, so it is treated as fatal rather than
    silently re-solved.
    """
    reasons = []
    if cachedEsm.startYear != rollingHorizonYears[0]:
        reasons.append(
            f"cached startYear ({cachedEsm.startYear}) does not match the "
            f"expected interval start year ({rollingHorizonYears[0]})"
        )
    if (
        cachedEsm.numberOfInvestmentPeriods
        != numberOfInvestmentPeriodsForRollingHorizon
    ):
        reasons.append(
            "cached numberOfInvestmentPeriods "
            f"({cachedEsm.numberOfInvestmentPeriods}) does not match "
            f"numberOfInvestmentPeriodsForRollingHorizon "
            f"({numberOfInvestmentPeriodsForRollingHorizon})"
        )
    return reasons


def _stockCommissioningDiffers(freshStock, cachedStock, tolerance=1e-5):
    """Compare the stockCommissioning that was just recomputed for a
    component going into this interval against what a cached interval was
    actually built from. This is what encodes the accumulated result of
    every prior interval in the chain, so a difference here means the
    cache no longer corresponds to the chain currently being computed.
    """
    if freshStock is None and cachedStock is None:
        return False
    if (freshStock is None) != (cachedStock is None):
        return True
    if set(freshStock.keys()) != set(cachedStock.keys()):
        return True
    for year in freshStock:
        freshSeries = pd.Series(freshStock[year]).sort_index()
        cachedSeries = pd.Series(cachedStock[year]).sort_index()
        if not freshSeries.index.equals(cachedSeries.index):
            return True
        if (freshSeries - cachedSeries).abs().max() > tolerance:
            return True
    return False


def _cachedIntervalChainMismatches(cachedEsm, rollingHorizonCompDict):
    """Check whether a cached interval was built from the same rolling
    horizon chain (same components, same accumulated stock) as what was
    just recomputed for it. Unlike a config mismatch, this is expected to
    legitimately happen when resuming after upstream inputs changed or
    after a gap forced an earlier interval to be re-solved differently, so
    callers should treat it as a stale cache to discard and rebuild, not a
    fatal error.
    """
    _, cachedCompDict = fn.dictIO.exportToDict(cachedEsm)

    freshComponents = {
        (classname, comp)
        for classname in rollingHorizonCompDict
        for comp in rollingHorizonCompDict[classname]
    }
    cachedComponents = {
        (classname, comp)
        for classname in cachedCompDict
        for comp in cachedCompDict[classname]
    }
    if freshComponents != cachedComponents:
        return ["cached interval's component set differs from the current esM"]

    reasons = []
    for classname, comp in freshComponents:
        freshStock = rollingHorizonCompDict[classname][comp]["stockCommissioning"]
        cachedStock = cachedCompDict[classname][comp]["stockCommissioning"]
        if _stockCommissioningDiffers(freshStock, cachedStock):
            reasons.append(
                f"stockCommissioning of {classname} '{comp}' differs from "
                "the cached interval"
            )
    return reasons


def _updateStockCommissioningForInterval(
    compEntry,
    classname,
    comp,
    rollingHorizonYears,
    rollingHorizonIntervals,
    interval,
    esM_results,
    persistedStock,
):
    """Update a single component's stockCommissioning for the interval about
    to be built: restore the stock accumulated from prior intervals, fold in
    the previous interval's commissioning (if any), and prune stock older
    than the component's technical lifetime. Mutates compEntry in place and
    updates persistedStock[classname][comp] for the next interval to pick up.
    """
    # restore accumulated stock from previous iterations
    compEntry["stockCommissioning"] = copy.deepcopy(persistedStock[classname][comp])

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
            if compEntry["stockCommissioning"] is None:
                compEntry["stockCommissioning"] = {}
                compEntry["stockCommissioning"][previousYear] = (
                    previousCommissioningLocation
                )
            # b) else add to structure
            else:
                compEntry["stockCommissioning"][previousYear] = (
                    previousCommissioningLocation
                )

            # c) delete "too" old stock, as it will make
            # problems with setup of parameters otherwise
            stockData = compEntry["stockCommissioning"]
            technicalLifetime = compEntry["technicalLifetime"]
            outdatedStockYears = [
                x
                for x in stockData.keys()
                if x < rollingHorizonYears[0] - technicalLifetime.max()
            ]
            for outdatedStockYear in outdatedStockYears:
                compEntry["stockCommissioning"].pop(outdatedStockYear)

    # persist updated stock for next iteration
    persistedStock[classname][comp] = copy.deepcopy(compEntry["stockCommissioning"])


def _filterComponentParametersForInterval(
    compEntry, rollingHorizonYears, stockYears, esM
):
    """Filter a single component's dict entry down to the parameter values
    relevant to this rolling horizon interval (plus, for non-operation-rate
    parameters, its accumulated stock years) so the rebuilt esM only ever
    sees data for years it actually spans. Mutates compEntry in place.
    """
    for parameter_name, parameter_value in compEntry.items():
        # stock commissioning is handled separately, by
        # _updateStockCommissioningForInterval
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
                compEntry[parameter_name] = new_parameter_value
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

                compEntry[parameter_name] = new_parameter_value
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
            compEntry[parameter_name] = new_parameter_value
        # 1.4 other parameters, which do not change over time
        else:
            pass


def _buildIntervalComponentDict(
    compDict,
    rollingHorizonYears,
    rollingHorizonIntervals,
    interval,
    esM_results,
    esM,
    persistedStock,
):
    """Build this interval's component dict from the original esM's exported
    compDict: restore/update each component's accumulated stockCommissioning
    (see _updateStockCommissioningForInterval) and filter every other
    parameter down to the years this interval actually spans (see
    _filterComponentParametersForInterval). Mutates persistedStock in place
    so the next interval picks up the stock this one leaves behind.
    """
    rollingHorizonCompDict = copy.deepcopy(dict(compDict))
    for classname in rollingHorizonCompDict:
        for comp in rollingHorizonCompDict[classname]:
            compEntry = rollingHorizonCompDict[classname][comp]
            _updateStockCommissioningForInterval(
                compEntry,
                classname,
                comp,
                rollingHorizonYears,
                rollingHorizonIntervals,
                interval,
                esM_results,
                persistedStock,
            )
            stockYears = (
                list(compEntry["stockCommissioning"].keys())
                if compEntry["stockCommissioning"] is not None
                else []
            )
            _filterComponentParametersForInterval(
                compEntry, rollingHorizonYears, stockYears, esM
            )
    return rollingHorizonCompDict


def _loadCachedInterval(
    resume,
    mustSolveFresh,
    netCDFPath,
    groupPrefix,
    rollingHorizonYears,
    numberOfInvestmentPeriodsForRollingHorizon,
    rollingHorizonCompDict,
):
    """Try to load this interval from the shared netCDF cache.

    Returns (rollingHorizonEsm, loadedFromCache): (None, False) if resuming
    is off, no cached group exists for this interval, or an earlier interval
    in this run already had to be solved fresh (see rollingHorizonOptimization's
    own cache-safety docs). A structurally mismatched cache raises a
    ValueError; a stale-but-structurally-valid one is discarded with a
    warning and (None, False) is returned so the caller solves it fresh.
    """
    if not (
        resume
        and not mustSolveFresh
        and netCDFPath is not None
        and _cachedGroupExists(netCDFPath, groupPrefix)
    ):
        return None, False

    candidateEsm = readNetCDFtoEnergySystemModel(
        str(netCDFPath), groupPrefix=groupPrefix
    )

    configMismatches = _cachedIntervalConfigMismatches(
        candidateEsm,
        rollingHorizonYears,
        numberOfInvestmentPeriodsForRollingHorizon,
    )
    if configMismatches:
        raise ValueError(
            f"Cached result in group '{groupPrefix}' of {netCDFPath} "
            f"does not match this call's configuration for interval "
            f"{rollingHorizonYears}: {'; '.join(configMismatches)}. "
            "Delete the cache or set resume=False to re-run this interval."
        )

    chainMismatches = _cachedIntervalChainMismatches(
        candidateEsm, rollingHorizonCompDict
    )
    if chainMismatches:
        warnings.warn(
            f"Cached result in group '{groupPrefix}' of {netCDFPath} is "
            f"stale and will be discarded and re-solved: "
            f"{'; '.join(chainMismatches)}."
        )
        return None, False

    print(
        f"Resuming: loading cached result for {rollingHorizonYears} "
        f"from group '{groupPrefix}' of {netCDFPath}"
    )
    return candidateEsm, True


def _buildIntervalEsm(
    esmDict,
    rollingHorizonCompDict,
    rollingHorizonYears,
    numberOfInvestmentPeriodsForRollingHorizon,
    esM,
):
    """Construct (but do not yet optimize) the EnergySystemModel for one
    rolling horizon interval: a copy of the original esM's settings, scoped
    down to this interval's startYear/numberOfInvestmentPeriods and to the
    years it spans, with this interval's components added.
    """
    rollingHorizonEsmDict = esmDict.copy()
    rollingHorizonEsmDict["startYear"] = rollingHorizonYears[0]
    rollingHorizonEsmDict["numberOfInvestmentPeriods"] = (
        numberOfInvestmentPeriodsForRollingHorizon
    )
    for param, value in rollingHorizonEsmDict.items():
        if isinstance(value, dict) and list(value.keys()) == esM.investmentPeriodNames:
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
    return rollingHorizonEsm


def _exportIntervalToNetCDF(rollingHorizonEsm, netCDFPath, groupPrefix):
    """Write one interval's esM into its own group of the shared rolling
    horizon netCDF file. overwriteExisting=False: this call only ever
    touches its own group (groupPrefix); the other intervals' groups in the
    shared file must be left untouched.
    """
    writeEnergySystemModelToNetCDF(
        rollingHorizonEsm,
        outputFilePath=str(netCDFPath),
        overwriteExisting=False,
        groupPrefix=groupPrefix,
    )


def _exportIntervalToExcel(
    rollingHorizonEsm,
    rollingHorizonYears,
    rollingHorizonIntervals,
    resultExportPath,
    scenario_name,
    excelOutputSettings,
):
    """Write one interval's optimization summary to the shared Excel output.
    For every interval except the last, only its first year is exported; the
    last interval exports every year it spans.
    """
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
            investmentPeriod=year,
            **excelOutputSettings,
        )


def rollingHorizonOptimization(
    esM,
    numberOfInvestmentPeriodsForRollingHorizon,
    timeSeriesAggregation=True,
    timeSeriesAggregationSettings=None,
    solver="gurobi",
    optimizationSpecs="",
    optimizeSettings=None,
    writeExcelOutput=False,
    excelOutputSettings=None,
    writeNetCDFOutput=False,
    resume=False,
    resultExportPath=None,
    scenario_name=None,
):
    """If numberOfInvestmentPeriodsForRollingHorizon == 1 -> Myopic Foresight, If numberOfInvestmentPeriodsForRollingHorizon == numberOfInvestmentPeriods -> Perfect Foresight (raises an error), else Rolling Horizon.

    If writeExcelOutput is True, resultExportPath and scenario_name must be set, and the optimization summaries of each rolling horizon interval are written to Excel files there, named after scenario_name.

    If writeNetCDFOutput is True, resultExportPath and scenario_name must be set, and the full esM (input and
    output) of every rolling horizon interval is written to a single shared netCDF file there, named
    "{scenario_name}_rollingHorizon.nc" - one group per interval, keyed by its start year (consistent with
    the single-file output of perfect foresight, unlike writing one file per interval). If resume is False,
    this file is cleared at the start of the call so a fresh run never mixes with stale groups from an
    earlier, unrelated run.

    If resume is True (implies writeNetCDFOutput), resultExportPath and scenario_name must be set. Before
    optimizing an interval, its group in that file is checked for; if present, the interval is loaded from
    there instead of being rebuilt and re-solved. This allows a rolling horizon run to be continued after
    being interrupted, without re-solving already completed intervals. Unlike the resume=False case, the file
    is never cleared up front, since the whole point is to keep prior intervals' groups around.

    Cache safety: a cached interval whose own startYear/numberOfInvestmentPeriods does not match what this
    call expects raises a ValueError (this is treated as a user error, e.g. calling with a different
    numberOfInvestmentPeriodsForRollingHorizon than the interrupted run used). A cached interval whose
    component set or accumulated stockCommissioning does not match what was just recomputed for it from the
    current esM and the (possibly freshly solved) prior interval is instead treated as stale: it is discarded
    with a warning and re-solved. Once any interval in the chain has been solved fresh for either reason,
    every later interval is also solved fresh, even if its group already exists in the file - a cached group
    surviving downstream of a point where the chain was regenerated is never trustworthy, since it was
    necessarily built from a different predecessor.

    :param timeSeriesAggregationSettings: keyword arguments passed directly to
        EnergySystemModel.aggregateTemporally (e.g. numberOfTypicalPeriods, numberOfTimeStepsPerPeriod,
        numberOfSegmentsPerPeriod, segmentation, clusterMethod, sortValues, rescaleClusterPeriods,
        representationMethod, or any further tsam kwarg). Settings not given fall back to
        aggregateTemporally's own defaults; only used if timeSeriesAggregation is True.
        |br| * the default value is None
    :type timeSeriesAggregationSettings: dict or None

    :param optimizeSettings: keyword arguments passed directly to EnergySystemModel.optimize for every
        interval (e.g. relaxIsBuiltBinary, logFileName, threads, timeLimit, warmstart, relevanceThreshold,
        includePerformanceSummary). declaresOptimizationProblem, timeSeriesAggregation, solver and
        optimizationSpecs are already covered by this function's own parameters and must not be repeated
        here. Settings not given fall back to optimize's own defaults.
        |br| * the default value is None
    :type optimizeSettings: dict or None

    :param excelOutputSettings: keyword arguments passed directly to writeOptimizationOutputToExcel for
        every exported interval (e.g. optSumOutputLevel, optValOutputLevel). outputFileName and
        investmentPeriod are already determined by this function and must not be repeated here. Settings
        not given fall back to writeOptimizationOutputToExcel's own defaults; only used if writeExcelOutput
        is True.
        |br| * the default value is None
    :type excelOutputSettings: dict or None
    """
    saveNetCDF = writeNetCDFOutput or resume

    if (writeExcelOutput or saveNetCDF) and resultExportPath is None:
        raise ValueError(
            "resultExportPath must be set if writeExcelOutput, writeNetCDFOutput or resume is True."
        )
    if (writeExcelOutput or saveNetCDF) and scenario_name is None:
        raise ValueError(
            "scenario_name must be set if writeExcelOutput, writeNetCDFOutput or resume is True."
        )

    tsaSettings = timeSeriesAggregationSettings or {}
    optimizeSettings = optimizeSettings or {}
    excelOutputSettings = excelOutputSettings or {}

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

    # all intervals share a single netCDF file, one group per interval (keyed by
    # its start year). A fresh (non-resumed) run starts from a clean file so it
    # never mixes with stale groups left over from an earlier, unrelated run;
    # a resumed run leaves the file untouched so prior intervals' groups survive.
    netCDFPath = None
    if saveNetCDF:
        netCDFPath = Path(resultExportPath) / f"{scenario_name}_rollingHorizon.nc"
        if not resume and netCDFPath.is_file():
            netCDFPath.unlink()

    print("Starting rolling horizon optimization.")

    esM_results = {}
    mustSolveFresh = False
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
        rollingHorizonCompDict = _buildIntervalComponentDict(
            compDict,
            rollingHorizonYears,
            rollingHorizonIntervals,
            interval,
            esM_results,
            esM,
            persistedStock,
        )

        # 2. check for a cached result of this interval (its own group in the
        # shared netCDF file) to resume from. Once any interval has had to be
        # (re-)solved, no later interval may be loaded from cache: a group
        # surviving downstream of a regenerated predecessor was necessarily
        # built from a different chain.
        groupPrefix = str(rollingHorizonYears[0])
        rollingHorizonEsm, loadedFromCache = _loadCachedInterval(
            resume,
            mustSolveFresh,
            netCDFPath,
            groupPrefix,
            rollingHorizonYears,
            numberOfInvestmentPeriodsForRollingHorizon,
            rollingHorizonCompDict,
        )

        if not loadedFromCache:
            mustSolveFresh = True
            # 3. build and optimize the rolling horizon esM
            rollingHorizonEsm = _buildIntervalEsm(
                esmDict,
                rollingHorizonCompDict,
                rollingHorizonYears,
                numberOfInvestmentPeriodsForRollingHorizon,
                esM,
            )
            if timeSeriesAggregation:
                rollingHorizonEsm.aggregateTemporally(solver=solver, **tsaSettings)

            rollingHorizonEsm.optimize(
                declaresOptimizationProblem=True,
                timeSeriesAggregation=timeSeriesAggregation,
                solver=solver,
                optimizationSpecs=optimizationSpecs,
                **optimizeSettings,
            )

        # 4. export optimization summaries
        if saveNetCDF and not loadedFromCache:
            _exportIntervalToNetCDF(rollingHorizonEsm, netCDFPath, groupPrefix)

        if writeExcelOutput:
            _exportIntervalToExcel(
                rollingHorizonEsm,
                rollingHorizonYears,
                rollingHorizonIntervals,
                resultExportPath,
                scenario_name,
                excelOutputSettings,
            )

        # 5. save esM's
        esM_results[rollingHorizonYears[0]] = rollingHorizonEsm

        print(f"Finished rolling horizon optimization for {rollingHorizonYears}")

    print("Finished rolling horizon optimization.")
    return esM_results
