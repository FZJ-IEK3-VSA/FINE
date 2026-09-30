import copy
import inspect
import warnings
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset

import fine as fn
from fine import utils
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from fine.IOManagement.xarrayIO import (
    writeEnergySystemModelToNetCDF,
    readNetCDFtoEnergySystemModel,
)

#: Number of digits a commissioning result is rounded to before it is handed on to the next
#: window as stock. An optimization result carries the full float64 precision, which
#: utils.checkAndSetStock and utils.checkStockCommissioning warn about and round to the same
#: number of digits themselves for their own capacityMax/capacityFix checks.
_STOCK_COMMISSIONING_DIGITS = 10

#: Component parameters whose per-investment-period dict covers the stock years on top of the
#: investment periods, because a component keeps paying for capacity commissioned before the
#: first modeled year (see the ``processedStockYears + esM.investmentPeriods`` arguments of
#: component.py and transmission.py). Every other per-investment-period parameter is validated
#: against the investment periods alone (utils.checkInvestmentPeriodParameters), so handing it
#: a stock year makes rebuilding the component fail.
#:
#: Written out rather than derived, although a component carries the distinction at runtime:
#: on a component that has stock, exactly these parameters' ``processed`` counterparts are
#: keyed by the stock years as well as by the investment periods. That signal is unusable
#: here, because the filter runs against the components of the original pathway model, which
#: normally has no stock at all - the stock is what the windows produce. On a component
#: without stock the counterparts of these parameters are keyed by the investment periods
#: like every other one, so deriving the set from it would answer "no stock years" for all of
#: them and strip the years the previous windows commissioned in. test_rolling_horizon.py
#: checks the list against that runtime signal on a component that does have stock, so that a
#: parameter added to component.py's list is noticed here.
_STOCK_YEAR_PARAMETERS = frozenset(
    {
        "investPerCapacity",
        "investIfBuilt",
        "opexPerCapacity",
        "opexIfBuilt",
        "QPcostScale",
    }
)

#: Arguments each settings dict must not carry, because rollingHorizonOptimization determines
#: them itself, per settings dict of rollingHorizonOptimization.
_RESERVED_SETTINGS = {
    "optimizeSettings": (
        "declaresOptimizationProblem",
        "timeSeriesAggregation",
        "solver",
        "optimizationSpecs",
    ),
    "excelOutputSettings": ("outputFileName", "investmentPeriod"),
    "netCDFOutputSettings": ("outputFilePath", "overwriteExisting", "groupPrefix"),
}


def _checkedSettings(settingsName, settings):
    """Validate one of the pass-through settings dicts of rollingHorizonOptimization.

    :param settingsName: name of the settings parameter, used to look up its reserved
        arguments and to name it in the error message.
    :type settingsName: string

    :param settings: the settings as passed by the caller.
    :type settings: dict or None

    :return: the settings, with None replaced by an empty dict.
    :rtype: dict

    :raises ValueError: if the settings carry an argument rollingHorizonOptimization
        determines itself, which would otherwise fail as a duplicate keyword argument.
    """
    settings = settings or {}
    reserved = sorted(set(settings) & set(_RESERVED_SETTINGS[settingsName]))
    if reserved:
        raise ValueError(
            f"{settingsName} must not contain {reserved}: rollingHorizonOptimization "
            "determines "
            + ("this argument" if len(reserved) == 1 else "these arguments")
            + " itself. Reserved for it are "
            f"{list(_RESERVED_SETTINGS[settingsName])}."
        )
    return settings


def _cachedGroupExists(netCDFPath, groupPrefix):
    """Check whether a given interval's group is already present in the
    shared rolling horizon netCDF file, without loading it.

    :param netCDFPath: path of the shared netCDF file of this rolling horizon run.
    :type netCDFPath: pathlib.Path

    :param groupPrefix: group of the interval to look for, its start year as a string.
    :type groupPrefix: string

    :return: whether the file exists and holds that group.
    :rtype: bool
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

    :param cachedEsm: energy system model read back from the interval's cached group.
    :type cachedEsm: EnergySystemModel instance

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param numberOfInvestmentPeriodsForRollingHorizon: window size this call was given.
    :type numberOfInvestmentPeriodsForRollingHorizon: strictly positive int

    :return: one message per mismatch, empty if the cache matches.
    :rtype: list of string
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


def _stockCommissioningDiffers(freshStock, cachedStock, threshold=1e-5):
    """Compare the stockCommissioning that was just recomputed for a
    component going into this interval against what a cached interval was
    actually built from. This is what encodes the accumulated result of
    every prior interval in the chain, so a difference here means the
    cache no longer corresponds to the chain currently being computed.

    :param freshStock: stockCommissioning just recomputed for this interval.
    :type freshStock: dict of {year: pd.Series} or None

    :param cachedStock: stockCommissioning the cached interval was built from.
    :type cachedStock: dict of {year: pd.Series} or None

    :param threshold: capacity difference below which two stocks count as equal. This
        compares the results of two separate solver runs, so it absorbs solver, not
        floating point, noise.
        |br| * the default value is 1e-5
    :type threshold: positive float

    :return: whether the two stocks differ.
    :rtype: bool
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
        if (freshSeries - cachedSeries).abs().max() > threshold:
            return True
    return False


def _cachedIntervalChainMismatches(
    cachedEsm, rollingHorizonCompDict, stockCommissioningThreshold=1e-5
):
    """Check whether a cached interval was built from the same rolling
    horizon chain (same components, same accumulated stock) as what was
    just recomputed for it. Unlike a config mismatch, this is expected to
    legitimately happen when resuming after upstream inputs changed or
    after a gap forced an earlier interval to be re-solved differently, so
    callers should treat it as a stale cache to discard and rebuild, not a
    fatal error.

    :param cachedEsm: energy system model read back from the interval's cached group.
    :type cachedEsm: EnergySystemModel instance

    :param rollingHorizonCompDict: component dict just built for this interval.
    :type rollingHorizonCompDict: dict

    :param stockCommissioningThreshold: capacity difference below which two stocks count
        as equal, see :func:`_stockCommissioningDiffers`.
        |br| * the default value is 1e-5
    :type stockCommissioningThreshold: positive float

    :return: one message per mismatch, empty if the cache belongs to this chain.
    :rtype: list of string
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
        if _stockCommissioningDiffers(
            freshStock, cachedStock, threshold=stockCommissioningThreshold
        ):
            reasons.append(
                f"stockCommissioning of {classname} '{comp}' differs from "
                "the cached interval"
            )
    return reasons


def _previousCommissioning(previousEsm, comp, previousYear):
    """Read what a component was commissioned with in the previous interval, in the form
    its stockCommissioning takes.

    Read from the commissioning variables rather than from the optimization summary: for a
    2-dim component the summary is split into a locationOut/locationIn matrix, while the
    stock is indexed by the connections themselves ("locationIn_locationOut"). Both are
    written to netCDF, so a resumed interval can be read the same way as a freshly solved
    one.

    :param previousEsm: the previous interval's solved energy system model.
    :type previousEsm: EnergySystemModel instance

    :param comp: name of the component.
    :type comp: string

    :param previousYear: the year of the previous interval to read.
    :type previousYear: int

    :return: the commissioned capacity of every location (1-dim) or connection (2-dim) of
        the energy system model, zero where the component has none, or None for a
        component that has no capacity variable and therefore no commissioning at all.
    :rtype: pd.Series or None
    """
    modelingClass = previousEsm.componentNames[comp]
    commissioning = previousEsm.componentModelingDict[
        modelingClass
    ].commissioningVariablesOptimum
    if commissioning is None:
        return None
    # a model of a single investment period holds the frame itself, one of several holds
    # them per investment period (see ComponentModel._convertOptimalValueNames)
    if isinstance(commissioning, dict):
        commissioning = commissioning.get(previousYear)
    if commissioning is None or comp not in commissioning.index.get_level_values(0):
        return None

    componentCommissioning = commissioning.loc[comp]
    if isinstance(componentCommissioning, pd.DataFrame):
        # 2-dim: a locationIn x locationOut matrix, whose cells without a connection are
        # NaN. As a stock it is indexed by the connections, zero where there is none.
        return utils.preprocess2dimData(componentCommissioning.fillna(0), discard=False)
    # 1-dim: only the locations the component is eligible in are listed, while a stock
    # covers every location of the energy system model, zero where there is none
    return componentCommissioning.reindex(
        sorted(previousEsm.locations), fill_value=0
    ).fillna(0)


def _updateStockCommissioningForInterval(
    compEntry,
    classname,
    comp,
    rollingHorizonYears,
    rollingHorizonIntervals,
    interval,
    esM_results,
    persistedStock,
    stockCommissioningThreshold,
):
    """Update a single component's stockCommissioning for the interval about
    to be built: restore the stock accumulated from prior intervals, fold in
    the previous interval's commissioning (if any), and prune stock older
    than the component's technical lifetime. Mutates compEntry in place and
    updates persistedStock[classname][comp] for the next interval to pick up.

    :param compEntry: the component's own entry of the component dict, i.e.
        compDict[classname][comp] as exported by dictIO.exportToDict.
    :type compEntry: dict

    :param classname: name of the component's class, e.g. "Source".
    :type classname: string

    :param comp: name of the component.
    :type comp: string

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param rollingHorizonIntervals: every interval of this run, in order.
    :type rollingHorizonIntervals: list of list of int

    :param interval: number of years between two investment periods.
    :type interval: strictly positive int

    :param esM_results: the intervals solved (or loaded) so far, keyed by start year.
    :type esM_results: dict of {int: EnergySystemModel instance}

    :param persistedStock: the stock accumulated so far, keyed by class and component
        name. Mutated in place for the next interval.
    :type persistedStock: dict

    :param stockCommissioningThreshold: capacity below which the previous interval's
        commissioning is not carried over as stock at all.
    :type stockCommissioningThreshold: positive float
    """
    # restore accumulated stock from previous iterations
    compEntry["stockCommissioning"] = copy.deepcopy(persistedStock[classname][comp])

    # first rolling horizon requires no changes, just external stock,
    # further years needs internal optimization results
    if rollingHorizonYears != rollingHorizonIntervals[0]:
        # get previous year
        previousYear = rollingHorizonYears[0] - interval
        # get commissioning results of previous rolling horizon
        previousCommissioning = _previousCommissioning(
            esM_results[previousYear], comp, previousYear
        )

        # add commissioning of previous runs as stock, if there was commissioning
        if (
            previousCommissioning is not None
            and previousCommissioning.sum() > stockCommissioningThreshold
        ):
            # a solved commissioning carries the full float64 precision, which the stock
            # checks of utils warn about; round it as they do before storing it
            if compEntry["stockCommissioning"] is None:
                compEntry["stockCommissioning"] = {}
            compEntry["stockCommissioning"][previousYear] = (
                previousCommissioning.astype(float).round(_STOCK_COMMISSIONING_DIGITS)
            )

        # delete "too" old stock, as it will make problems with setup of parameters
        # otherwise. Independent of whether this interval added any commissioning: an
        # entry that has outlived its technical lifetime must not survive just because
        # nothing new was built in the previous interval.
        if compEntry["stockCommissioning"] is not None:
            technicalLifetime = compEntry["technicalLifetime"]
            outdatedStockYears = [
                x
                for x in compEntry["stockCommissioning"].keys()
                if x < rollingHorizonYears[0] - technicalLifetime.max()
            ]
            for outdatedStockYear in outdatedStockYears:
                compEntry["stockCommissioning"].pop(outdatedStockYear)
            # None, not an empty dict, is how a component without stock is described;
            # utils.checkAndSetStock cannot read an empty one
            if not compEntry["stockCommissioning"]:
                compEntry["stockCommissioning"] = None

    # persist updated stock for next iteration
    persistedStock[classname][comp] = copy.deepcopy(compEntry["stockCommissioning"])


def _filterComponentParametersForInterval(
    compEntry, rollingHorizonYears, stockYears, esM
):
    """Filter a single component's dict entry down to the parameter values
    relevant to this rolling horizon interval (plus, for the parameters that
    are given per stock year as well, its accumulated stock years) so the
    rebuilt esM only ever sees data for years it actually spans. Mutates
    compEntry in place.

    :param compEntry: the component's own entry of the component dict, i.e.
        compDict[classname][comp] as exported by dictIO.exportToDict.
    :type compEntry: dict

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param stockYears: years the component holds stock commissioning for.
    :type stockYears: list of int

    :param esM: the original energy system model of the whole pathway.
    :type esM: EnergySystemModel instance
    """
    for parameter_name, parameter_value in compEntry.items():
        # stock commissioning is handled separately, by
        # _updateStockCommissioningForInterval
        if parameter_name == "stockCommissioning":
            continue
        # 1.2 commodity conversion factors
        if parameter_name == "commodityConversionFactors":
            if not parameter_value:
                continue
            firstKey = next(iter(parameter_value))
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
        # 1.3 other parameter which are yearly dependent. A parameter given per investment
        # period covers every investment period of the esM, since
        # utils.checkInvestmentPeriodParameters demands that its keys match them exactly
        # (plus the stock years, for the parameters listed in _STOCK_YEAR_PARAMETERS). A
        # dict that does not is a dict for an unrelated reason - pwlcfParameters, for
        # instance, is keyed by 'etlParameters'/'eosParameters' - and is left untouched
        # rather than filtered down to an empty one.
        elif isinstance(parameter_value, dict) and set(parameter_value).issuperset(
            esM.investmentPeriodNames
        ):
            # Only the parameters describing capacity that was commissioned once and is
            # paid for over its lifetime are given for the stock years as well; every
            # other one is validated against the investment periods alone, so a stock
            # year among its keys would make rebuilding the component fail.
            relevantYears = list(rollingHorizonYears)
            if parameter_name in _STOCK_YEAR_PARAMETERS:
                relevantYears += stockYears
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
    stockCommissioningThreshold,
):
    """Build this interval's component dict from the original esM's exported
    compDict: restore/update each component's accumulated stockCommissioning
    (see _updateStockCommissioningForInterval) and filter every other
    parameter down to the years this interval actually spans (see
    _filterComponentParametersForInterval). Mutates persistedStock in place
    so the next interval picks up the stock this one leaves behind.

    :param compDict: component dict of the whole pathway, as exported by
        dictIO.exportToDict.
    :type compDict: dict

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param rollingHorizonIntervals: every interval of this run, in order.
    :type rollingHorizonIntervals: list of list of int

    :param interval: number of years between two investment periods.
    :type interval: strictly positive int

    :param esM_results: the intervals solved (or loaded) so far, keyed by start year.
    :type esM_results: dict of {int: EnergySystemModel instance}

    :param esM: the original energy system model of the whole pathway.
    :type esM: EnergySystemModel instance

    :param persistedStock: the stock accumulated so far, keyed by class and component
        name. Mutated in place for the next interval.
    :type persistedStock: dict

    :param stockCommissioningThreshold: capacity below which the previous interval's
        commissioning is not carried over as stock at all.
    :type stockCommissioningThreshold: positive float

    :return: the component dict this interval's energy system model is built from.
    :rtype: dict
    """
    rollingHorizonCompDict = copy.deepcopy(dict(compDict))
    for classname in rollingHorizonCompDict:
        constructorArguments = set(
            inspect.getfullargspec(getattr(fn, classname).__init__).args
        )
        for comp in rollingHorizonCompDict[classname]:
            compEntry = rollingHorizonCompDict[classname][comp]
            # For a clustered energy system model, exportToDict adds the aggregated time
            # series (aggregatedOperationRateMax and friends) on top of the constructor
            # arguments it exports otherwise, and the component cannot be rebuilt with
            # them. Recognize them the way exportToDict decides what to export in the
            # first place - by the constructor's own signature - rather than by their
            # name, so that anything else it may add later is dropped as well.
            for parameter_name in [
                parameter_name
                for parameter_name in compEntry
                if parameter_name not in constructorArguments
            ]:
                compEntry.pop(parameter_name)
            _updateStockCommissioningForInterval(
                compEntry,
                classname,
                comp,
                rollingHorizonYears,
                rollingHorizonIntervals,
                interval,
                esM_results,
                persistedStock,
                stockCommissioningThreshold,
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
    netCDFPath,
    groupPrefix,
    rollingHorizonYears,
    numberOfInvestmentPeriodsForRollingHorizon,
    rollingHorizonCompDict,
    stockCommissioningThreshold,
    verboseLogLevel,
):
    """Load one interval from its group of the shared netCDF cache.

    Only called for a group that exists and may still be trusted; whether the cache may
    be used at all is decided by rollingHorizonOptimization (see its own cache-safety
    docs). A structurally mismatched cache raises a ValueError; a stale but structurally
    valid one is discarded with a warning and None is returned, so that the caller solves
    the interval fresh.

    :param netCDFPath: path of the shared netCDF file of this rolling horizon run.
    :type netCDFPath: pathlib.Path

    :param groupPrefix: group of this interval, its start year as a string.
    :type groupPrefix: string

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param numberOfInvestmentPeriodsForRollingHorizon: window size this call was given.
    :type numberOfInvestmentPeriodsForRollingHorizon: strictly positive int

    :param rollingHorizonCompDict: component dict just built for this interval, which the
        cached one must match.
    :type rollingHorizonCompDict: dict

    :param stockCommissioningThreshold: capacity difference below which two stocks count
        as equal, see :func:`_stockCommissioningDiffers`.
    :type stockCommissioningThreshold: positive float

    :param verboseLogLevel: verbosity level of the original esM, see
        :class:`~fine.energySystemModel.EnergySystemModel`.
    :type verboseLogLevel: int (0,1,2)

    :return: the cached energy system model, or None if it turned out to be stale.
    :rtype: EnergySystemModel instance or None

    :raises ValueError: if the cached interval was built for another configuration.
    """
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
        candidateEsm, rollingHorizonCompDict, stockCommissioningThreshold
    )
    if chainMismatches:
        warnings.warn(
            f"Cached result in group '{groupPrefix}' of {netCDFPath} is "
            f"stale and will be discarded and re-solved: "
            f"{'; '.join(chainMismatches)}."
        )
        return None

    utils.output(
        f"Resuming: loading cached result for {rollingHorizonYears} "
        f"from group '{groupPrefix}' of {netCDFPath}",
        verboseLogLevel,
        0,
    )
    return candidateEsm


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

    :param esmDict: energy system model dict of the whole pathway, as exported by
        dictIO.exportToDict.
    :type esmDict: dict

    :param rollingHorizonCompDict: component dict built for this interval.
    :type rollingHorizonCompDict: dict

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param numberOfInvestmentPeriodsForRollingHorizon: window size, i.e. the number of
        investment periods of the model built here.
    :type numberOfInvestmentPeriodsForRollingHorizon: strictly positive int

    :param esM: the original energy system model of the whole pathway.
    :type esM: EnergySystemModel instance

    :return: this interval's energy system model, not yet optimized.
    :rtype: EnergySystemModel instance
    """
    rollingHorizonEsmDict = esmDict.copy()
    rollingHorizonEsmDict["startYear"] = rollingHorizonYears[0]
    rollingHorizonEsmDict["numberOfInvestmentPeriods"] = (
        numberOfInvestmentPeriodsForRollingHorizon
    )
    for param, value in rollingHorizonEsmDict.items():
        # keys, not an ordered list of them: a parameter given per investment period
        # describes the same years no matter which order they were written down in
        if isinstance(value, dict) and set(value.keys()) == set(
            esM.investmentPeriodNames
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
    return rollingHorizonEsm


def _exportIntervalToNetCDF(
    rollingHorizonEsm, netCDFPath, groupPrefix, netCDFOutputSettings
):
    """Write one interval's esM into its own group of the shared rolling
    horizon netCDF file. overwriteExisting=False: this call only ever
    touches its own group (groupPrefix); the other intervals' groups in the
    shared file must be left untouched.

    :param rollingHorizonEsm: the interval's solved energy system model.
    :type rollingHorizonEsm: EnergySystemModel instance

    :param netCDFPath: path of the shared netCDF file of this rolling horizon run.
    :type netCDFPath: pathlib.Path

    :param groupPrefix: group to write this interval to, its start year as a string.
    :type groupPrefix: string

    :param netCDFOutputSettings: further keyword arguments for
        xarrayIO.writeEnergySystemModelToNetCDF.
    :type netCDFOutputSettings: dict
    """
    writeEnergySystemModelToNetCDF(
        rollingHorizonEsm,
        outputFilePath=str(netCDFPath),
        overwriteExisting=False,
        groupPrefix=groupPrefix,
        **netCDFOutputSettings,
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

    :param rollingHorizonEsm: the interval's solved energy system model.
    :type rollingHorizonEsm: EnergySystemModel instance

    :param rollingHorizonYears: investment period years this interval spans.
    :type rollingHorizonYears: list of int

    :param rollingHorizonIntervals: every interval of this run, in order.
    :type rollingHorizonIntervals: list of list of int

    :param resultExportPath: directory the Excel files are written to.
    :type resultExportPath: string

    :param scenario_name: name the Excel files are named after.
    :type scenario_name: string

    :param excelOutputSettings: further keyword arguments for
        standardIO.writeOptimizationOutputToExcel.
    :type excelOutputSettings: dict
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
    solver="None",
    optimizationSpecs="",
    optimizeSettings=None,
    writeExcelOutput=False,
    excelOutputSettings=None,
    writeNetCDFOutput=False,
    netCDFOutputSettings=None,
    resume=False,
    resultExportPath=None,
    scenario_name=None,
    stockCommissioningThreshold=1e-5,
):
    """Optimize an energy system model along a rolling horizon: instead of all investment
    periods at once (perfect foresight), a window of numberOfInvestmentPeriodsForRolling
    Horizon consecutive investment periods is optimized at a time, and the window is then
    moved forward by one investment period. The capacity commissioned in a window is
    handed to the next one as stock, so that each window decides with foresight over its
    own years only.

    A window size of 1 is myopic foresight, i.e. every investment period is optimized on
    its own. A window as long as the pathway would be perfect foresight and is refused;
    use EnergySystemModel.optimize for that.

    Cache safety: a cached interval whose own startYear/numberOfInvestmentPeriods does not match what this
    call expects raises a ValueError (this is treated as a user error, e.g. calling with a different
    numberOfInvestmentPeriodsForRollingHorizon than the interrupted run used). A cached interval whose
    component set or accumulated stockCommissioning does not match what was just recomputed for it from the
    current esM and the (possibly freshly solved) prior interval is instead treated as stale: it is discarded
    with a warning and re-solved. Once any interval in the chain has been solved fresh for either reason,
    every later interval is also solved fresh, even if its group already exists in the file - a cached group
    surviving downstream of a point where the chain was regenerated is never trustworthy, since it was
    necessarily built from a different predecessor.

    Reading the results: the windows overlap, so a year is reported by every window that spans it, and each
    window discounts onto its own first year. Exactly one window is responsible for each year of the pathway:
    every window for its own first year, and the last window for all of the years it spans. Adding up any
    other selection double counts the overlaps. To compare or add up the windows, use the NPVcontributionRH
    row of the optimization summary rather than NPVcontribution: every window is built with
    rollingHorizonStartYear set to the first year of the whole pathway, and NPVcontributionRH is the window's
    NPVcontribution discounted back onto that shared reference year (see
    :class:`~fine.energySystemModel.EnergySystemModel`). Summed over the years described above, it
    reproduces the net present value of the equivalent perfect foresight run for the same commissioning
    decisions. The Excel output of this function already writes exactly those years.

    Costs of a window: capacity is charged as an annuity for the investment periods a window spans, and the
    capacity a window commissions keeps being charged in the following windows, where it arrives as stock, at
    the investPerCapacity of its original commissioning year. No cost is therefore lost between the windows.
    What a window does not see is the cost it commits its successors to beyond its own last year - that is
    what limited foresight means, and it is stronger the smaller the window. As under perfect foresight, cost
    falling beyond the last year of the pathway is not booked at all.

    **Required arguments:**

    :param esM: energy system model of the whole transformation pathway, which the rolling horizon windows
        are built from. It is neither optimized nor modified; every window is an independent copy.
    :type esM: EnergySystemModel instance

    :param numberOfInvestmentPeriodsForRollingHorizon: number of consecutive investment periods optimized
        at a time. Must be smaller than the number of investment periods of esM; 1 is myopic foresight.
    :type numberOfInvestmentPeriodsForRollingHorizon: strictly positive int

    **Default arguments:**

    :param timeSeriesAggregation: states if the optimization of every window should be done with

        (a) the full time series (False) or
        (b) clustered time series data (True), aggregated per window.

        |br| * the default value is True
    :type timeSeriesAggregation: boolean

    :param timeSeriesAggregationSettings: keyword arguments passed directly to
        EnergySystemModel.aggregateTemporally for every window (e.g. n_clusters, period_duration, cluster,
        segments, extremes, preserve_column_means, or any further ETHOS.TSAM keyword argument). The solver
        used for a clustering method that needs one is part of it as well (cluster=ClusterConfig(solver=...)),
        and is deliberately not taken from this function's own solver argument, which selects the solver of
        the optimization itself. Settings not given fall back to aggregateTemporally's own defaults; only
        used if timeSeriesAggregation is True.
        |br| * the default value is None
    :type timeSeriesAggregationSettings: dict or None

    :param solver: specifies which solver should solve the optimization problem of every window (which of
        course has to be installed on the machine on which the model is run). As in
        EnergySystemModel.optimize, 'None' selects an installed solver automatically.
        |br| * the default value is 'None'
    :type solver: string

    :param optimizationSpecs: specifies parameters for the optimization solver (see the respective solver
        documentation for more information). Example: 'LogToConsole=1 OptimalityTol=1e-6'
        |br| * the default value is an empty string ('')
    :type optimizationSpecs: string

    :param optimizeSettings: keyword arguments passed directly to EnergySystemModel.optimize for every
        interval (e.g. relaxIsBuiltBinary, logFileName, threads, timeLimit, warmstart, relevanceThreshold,
        includePerformanceSummary). declaresOptimizationProblem, timeSeriesAggregation, solver and
        optimizationSpecs are already covered by this function's own parameters and must not be repeated
        here. Settings not given fall back to optimize's own defaults.
        |br| * the default value is None
    :type optimizeSettings: dict or None

    :param writeExcelOutput: states if the optimization summary of every window should be written to an
        Excel file in resultExportPath, named after scenario_name and the exported year. For every window
        except the last, its first year is exported; the last window exports every year it spans. Requires
        resultExportPath and scenario_name to be set.
        |br| * the default value is False
    :type writeExcelOutput: boolean

    :param excelOutputSettings: keyword arguments passed directly to writeOptimizationOutputToExcel for
        every exported interval (e.g. optSumOutputLevel, optValOutputLevel). outputFileName and
        investmentPeriod are already determined by this function and must not be repeated here. Settings
        not given fall back to writeOptimizationOutputToExcel's own defaults; only used if writeExcelOutput
        is True.
        |br| * the default value is None
    :type excelOutputSettings: dict or None

    :param writeNetCDFOutput: states if the full esM (input and output) of every window should be written to
        a single shared netCDF file in resultExportPath, named "{scenario_name}_rollingHorizon.nc" - one
        group per window, keyed by its start year (consistent with the single file output of perfect
        foresight, unlike writing one file per window). If resume is False, this file is cleared at the
        start of the call so that a fresh run never mixes with stale groups from an earlier, unrelated run.
        Requires resultExportPath and scenario_name to be set.
        |br| * the default value is False
    :type writeNetCDFOutput: boolean

    :param netCDFOutputSettings: keyword arguments passed directly to
        xarrayIO.writeEnergySystemModelToNetCDF for every written window (e.g. optSumOutputLevel,
        includeShadowPrices, shadowPriceConstraintStr). outputFilePath, overwriteExisting and groupPrefix
        are already determined by this function and must not be repeated here, since they are what makes
        the windows share one file. Every other argument of the writer is reachable, including ones added
        to it later. Settings not given fall back to writeEnergySystemModelToNetCDF's own defaults; only
        used if writeNetCDFOutput or resume is True.

        .. note::
            includeShadowPrices=True currently fails inside writeEnergySystemModelToNetCDF itself, for
            any energy system model and independently of the rolling horizon. It is passed on unchanged;
            it will start working here once the writer does.

        |br| * the default value is None
    :type netCDFOutputSettings: dict or None

    :param resume: states if an interrupted run should be continued instead of started over (implies
        writeNetCDFOutput). Before optimizing a window, its group in the shared netCDF file is checked for;
        if present, the window is loaded from there instead of being rebuilt and re-solved, subject to the
        cache safety rules described above. Unlike the resume=False case, the file is never cleared up
        front, since the whole point is to keep prior windows' groups around. Requires resultExportPath and
        scenario_name to be set.
        |br| * the default value is False
    :type resume: boolean

    :param resultExportPath: directory the Excel and netCDF output is written to. Required if
        writeExcelOutput, writeNetCDFOutput or resume is True.
        |br| * the default value is None
    :type resultExportPath: string or None

    :param scenario_name: name the Excel and netCDF output files are named after. Required if
        writeExcelOutput, writeNetCDFOutput or resume is True.
        |br| * the default value is None
    :type scenario_name: string or None

    :param stockCommissioningThreshold: capacity below which a window's commissioning is not handed to the
        next window as stock at all, and below which two stocks count as equal when a cached window is
        checked for staleness. Both compare capacities across separate solver runs, so this absorbs solver,
        not floating point, noise.
        |br| * the default value is 1e-5
    :type stockCommissioningThreshold: positive float

    :return: the optimized energy system model of every window, keyed by the window's first year. Each of
        them is an ordinary EnergySystemModel and is read like any other solved one; see "Reading the
        results" above for which year to take from which window.
    :rtype: dict of {int: EnergySystemModel instance}
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
    optimizeSettings = _checkedSettings("optimizeSettings", optimizeSettings)
    excelOutputSettings = _checkedSettings("excelOutputSettings", excelOutputSettings)
    netCDFOutputSettings = _checkedSettings(
        "netCDFOutputSettings", netCDFOutputSettings
    )

    # checks for data input. The type checks come first, so that a wrongly typed window
    # size is reported as such instead of as a failing comparison against it.
    utils.isStrictlyPositiveInt(numberOfInvestmentPeriodsForRollingHorizon)
    utils.isPositiveNumber(stockCommissioningThreshold)

    if esM.numberOfInvestmentPeriods < 2:
        raise ValueError("At least two investmentperiods required for rolling horizon.")
    if esM.numberOfInvestmentPeriods <= numberOfInvestmentPeriodsForRollingHorizon:
        raise ValueError(
            "There must be at least one more investment period in the "
            "transformation pathway than in the rolling horizon interval"
        )

    # Settings of the energy system model whose meaning does not survive being cut into
    # windows. They are refused rather than silently reinterpreted per window.
    if esM.stochasticModel:
        raise NotImplementedError(
            "A rolling horizon cannot be applied to a stochastic model: there, the "
            "investment periods are the scenarios of one and the same year, not a "
            "transformation pathway to move a window along."
        )
    if esM.pathwayBalanceLimit is not None:
        raise NotImplementedError(
            "pathwayBalanceLimit is a budget for the whole transformation pathway, so "
            "each rolling horizon window would enforce it again in full. Express the "
            "limit per investment period, as a balanceLimit, instead."
        )
    if esM.annuityPerpetuity:
        raise NotImplementedError(
            "annuityPerpetuity assumes that the last investment period is maintained "
            "forever, which for a rolling horizon window would assume it of the last "
            "year of the window rather than of the pathway."
        )
    pwlcfComponents = sorted(
        compName
        for compName in esM.componentNames
        if getattr(esM.getComponent(compName), "pwlcf", None) is not None
    )
    if pwlcfComponents:
        raise NotImplementedError(
            f"pwlcfParameters is set on {pwlcfComponents}. Endogenous technological "
            "learning accumulates over the whole transformation pathway, while every "
            "rolling horizon window is built and solved on its own, so the learning "
            "curve would restart in each of them."
        )

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

    # Every window inherits the first year of the whole pathway, so that the
    # NPVcontributionRH rows of all windows are discounted onto one shared reference year
    # (a window's own startYear moves along with it and cannot serve as one). It is set on
    # the exported dict, not on esM itself, so that the caller's model is left untouched.
    esmDict["rollingHorizonStartYear"] = (
        esM.rollingHorizonStartYear
        if esM.rollingHorizonStartYear is not None
        else esM.startYear
    )

    # all intervals share a single netCDF file, one group per interval (keyed by
    # its start year). A fresh (non-resumed) run starts from a clean file so it
    # never mixes with stale groups left over from an earlier, unrelated run;
    # a resumed run leaves the file untouched so prior intervals' groups survive.
    netCDFPath = None
    if saveNetCDF:
        netCDFPath = Path(resultExportPath) / f"{scenario_name}_rollingHorizon.nc"
        if not resume and netCDFPath.is_file():
            warnings.warn(
                f"{netCDFPath} already exists and is deleted, so that the results of "
                "this run do not mix with the ones of an earlier run. Set resume=True "
                "to continue that run instead."
            )
            netCDFPath.unlink()

    utils.output("Starting rolling horizon optimization.", esM.verboseLogLevel, 0)

    esM_results = {}
    # Whether a cached interval may still be loaded. Once an interval has had to be
    # solved, no later one may be: a group surviving downstream of a regenerated
    # predecessor was necessarily built from a different chain.
    cacheIsUsable = resume
    persistedStock = {
        classname: {
            comp: copy.deepcopy(compDict[classname][comp]["stockCommissioning"])
            for comp in compDict[classname]
        }
        for classname in compDict
    }
    for rollingHorizonYears in rollingHorizonIntervals:
        utils.output(
            f"Initizializing rolling horizon optimization for {rollingHorizonYears}...",
            esM.verboseLogLevel,
            0,
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
            stockCommissioningThreshold,
        )

        # 2. resume this interval from its own group of the shared netCDF file, if that
        # group exists and may still be trusted
        groupPrefix = str(rollingHorizonYears[0])
        rollingHorizonEsm = None
        if cacheIsUsable and _cachedGroupExists(netCDFPath, groupPrefix):
            rollingHorizonEsm = _loadCachedInterval(
                netCDFPath,
                groupPrefix,
                rollingHorizonYears,
                numberOfInvestmentPeriodsForRollingHorizon,
                rollingHorizonCompDict,
                stockCommissioningThreshold,
                esM.verboseLogLevel,
            )

        if rollingHorizonEsm is None:
            cacheIsUsable = False
            # 3. build and optimize the rolling horizon esM
            rollingHorizonEsm = _buildIntervalEsm(
                esmDict,
                rollingHorizonCompDict,
                rollingHorizonYears,
                numberOfInvestmentPeriodsForRollingHorizon,
                esM,
            )
            if timeSeriesAggregation:
                rollingHorizonEsm.aggregateTemporally(**tsaSettings)

            rollingHorizonEsm.optimize(
                declaresOptimizationProblem=True,
                timeSeriesAggregation=timeSeriesAggregation,
                solver=solver,
                optimizationSpecs=optimizationSpecs,
                **optimizeSettings,
            )

            # 4. export optimization summaries
            if saveNetCDF:
                _exportIntervalToNetCDF(
                    rollingHorizonEsm, netCDFPath, groupPrefix, netCDFOutputSettings
                )

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

        utils.output(
            f"Finished rolling horizon optimization for {rollingHorizonYears}",
            esM.verboseLogLevel,
            0,
        )

    utils.output("Finished rolling horizon optimization.", esM.verboseLogLevel, 0)
    return esM_results
