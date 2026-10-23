"""Sequential stochastic optimization: a rolling horizon whose windows are multi-stage
stochastic programs.

Every window is solved with :func:`optimizeMultiStageStochastic` on a scenario tree built for
it. The decisions of the window's first stage are implemented, i.e. handed to the following
windows as stock, and the window moves on by the length of the first stage. The result is
one implemented pathway, decided under uncertainty at every step.

The windows are built like those of :func:`rollingHorizon.rollingHorizonOptimization` and reuse
its helpers for reading a window's commissioning, for giving passed-on stock its lead time and
for scoping the parameters to a window. Unlike there, a window implements several investment
periods at once, so the stock is handed over by this module itself.
"""

import copy
import inspect
from dataclasses import dataclass

import pandas as pd

import fine as fn
from fine import utils
from fine.enums import Dimension
from fine.expansionModules import multiStageStochasticProgram as mssp
from fine.expansionModules.rollingHorizon import (
    _STOCK_COMMISSIONING_DIGITS,
    _buildIntervalEsm,
    _expandLeadTimeForPassedOnStock,
    _filterComponentParametersForInterval,
    _previousCommissioning,
    _roundedStockLeadTimeInYears,
)

#: Arguments of optimizeMultiStageStochastic that this module determines itself.
_RESERVED_MSSP_SETTINGS = (
    "baseModel",
    "tree",
    "stageInvestmentPeriods",
    "processResults",
)

#: Arguments of EnergySystemModel.optimize that this module determines itself for a
#: deterministic window.
_RESERVED_OPTIMIZE_SETTINGS = (
    "declaresOptimizationProblem",
    "timeSeriesAggregation",
    "solver",
)

_END_OF_HORIZON_OPTIONS = ("extend", "deterministic")


@dataclass
class WindowResult:
    """The outcome of one window of a sequential stochastic optimization.

    :param startYear: first year of the window
    :type startYear: int

    :param years: investment period years the window spans
    :type years: list of int

    :param stageYears: years of every stage of the window's scenario tree
    :type stageYears: list of lists of int

    :param implementedYears: years whose decisions were implemented, i.e. passed on to the
        following windows as stock
    :type implementedYears: list of int

    :param deterministic: whether the window was solved deterministically (the last window
        with ``endOfHorizon="deterministic"``)
    :type deterministic: boolean

    :param tree: the scenario tree the window was solved with
    :type tree: ScenarioTree instance

    :param objectiveValue: the window's optimal objective value: the expected net present
        value over its scenarios, discounted onto the window's first year
    :type objectiveValue: float

    :param implementedCommissioning: the implemented decisions, as
        ``{(modelingClass, componentName, location, year): commissioned capacity}``
    :type implementedCommissioning: dict

    :param results: the window's solved models, as returned by
        :func:`optimizeMultiStageStochastic`. For a deterministic window it holds the single
        solved model, and its first stage covers the whole window. None if the run was made
        with ``keepWindowResults=False``.
    :type results: StochasticOptimizationResults instance or None
    """

    startYear: int
    years: list
    stageYears: list
    implementedYears: list
    deterministic: bool
    tree: object
    objectiveValue: float
    implementedCommissioning: dict
    results: object = None


class SequentialStochasticResults:
    """Container for the outcome of a sequential stochastic optimization.

    :param windows: the windows in the order they were solved
    :type windows: list of WindowResult
    """

    def __init__(self, windows):
        self.windows = windows

    @property
    def implementedCommissioning(self):
        """The implemented pathway: every window's implemented decisions together, as
        ``{(modelingClass, componentName, location, year): commissioned capacity}``. Every
        year up to the last implemented one is covered by exactly one window.
        """
        commissioning = {}
        for window in self.windows:
            commissioning.update(window.implementedCommissioning)
        return commissioning

    def implementedCommissioningTable(self):
        """Return the implemented pathway as a table.

        :return: commissioned capacity, indexed by modeling class, component and location,
            with one column per implemented year
        :rtype: pd.DataFrame
        """
        series = pd.Series(self.implementedCommissioning, dtype=float)
        if series.empty:
            return pd.DataFrame()
        series.index.names = ["modelingClass", "component", "location", "year"]
        return series.unstack("year").sort_index()


@dataclass
class _WindowPlan:
    start: int  # index of the window's first year among the pathway's years
    length: int  # number of investment periods of the window
    implemented: int  # number of investment periods whose decisions are implemented
    deterministic: bool


def _checkedSettings(settingsName, settings, reserved):
    """Return a pass-through settings dict, refusing arguments this module sets itself."""
    settings = dict(settings or {})
    clashes = sorted(set(settings) & set(reserved))
    if clashes:
        raise ValueError(
            f"{settingsName} must not contain {clashes}: sequentialStochasticOptimization "
            f"determines them itself. Reserved for it are {list(reserved)}."
        )
    return settings


def _refuseUnsupportedSettings(esM):
    """Refuse the settings of a pathway that do not survive being cut into windows.

    The same settings rollingHorizonOptimization refuses, for the same reasons.
    """
    if esM.stochasticModel:
        raise NotImplementedError(
            "A sequential stochastic optimization cannot be applied to a model with "
            "stochasticModel=True: there, the investment periods are the scenarios of one "
            "and the same year, not a transformation pathway to move a window along."
        )
    if esM.pathwayBalanceLimit is not None:
        raise NotImplementedError(
            "pathwayBalanceLimit is a budget for the whole transformation pathway, so each "
            "window would enforce it again in full. Express the limit per investment "
            "period, as a balanceLimit, instead."
        )
    if esM.annuityPerpetuity:
        raise NotImplementedError(
            "annuityPerpetuity assumes that the last investment period is maintained "
            "forever, which for a window would assume it of the last year of the window "
            "rather than of the pathway."
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
            "window is built and solved on its own, so the learning curve would restart in "
            "each of them."
        )


def _validatedStages(stageInvestmentPeriods, windowLength):
    """Check that the stages cover the window's investment periods in order."""
    stages = [
        [stage] if isinstance(stage, int) else list(stage)
        for stage in stageInvestmentPeriods
    ]
    if len(stages) < 2:
        raise ValueError(
            "stageInvestmentPeriods must describe at least two stages: a window with a "
            "single stage has no uncertainty to hedge against."
        )
    flattened = [ip for stage in stages for ip in stage]
    if any(not stage for stage in stages) or flattened != list(range(windowLength)):
        raise ValueError(
            "stageInvestmentPeriods must cover the investment periods of a window, "
            f"0 to {windowLength - 1}, exactly once, in ascending order and without empty "
            f"stages. Got {stageInvestmentPeriods}."
        )
    return stages


def _planWindows(numberOfYears, lastIndex, windowLength, stepLength, endOfHorizon):
    """Lay out the windows along the pathway.

    :param numberOfYears: number of investment periods of the pathway model
    :param lastIndex: index of the last year whose decisions are to be implemented
    :param windowLength: number of investment periods of a stochastic window
    :param stepLength: number of investment periods implemented per stochastic window
    :param endOfHorizon: 'extend' or 'deterministic'

    :return: the windows in order
    :rtype: list of _WindowPlan
    """
    windows = []
    if endOfHorizon == "extend":
        for start in range(0, lastIndex + 1, stepLength):
            windows.append(
                _WindowPlan(
                    start,
                    windowLength,
                    min(stepLength, lastIndex + 1 - start),
                    False,
                )
            )
        missing = windows[-1].start + windowLength - numberOfYears
        if missing > 0:
            raise ValueError(
                "With endOfHorizon='extend', every window is a full stochastic window, so "
                "the pathway model must reach beyond lastYear: the last window starts at "
                f"investment period {windows[-1].start} and needs {missing} more "
                "investment period(s) than the model has. Extend the model's data, choose "
                "an earlier lastYear, or use endOfHorizon='deterministic'."
            )
        return windows

    start = 0
    while start + windowLength - 1 <= lastIndex:
        windows.append(_WindowPlan(start, windowLength, stepLength, False))
        start += stepLength
    if not windows:
        raise ValueError(
            f"The pathway up to lastYear spans {lastIndex + 1} investment periods, fewer "
            f"than one window of {windowLength}. Use optimizeMultiStageStochastic for the "
            "whole pathway instead."
        )
    if start <= lastIndex:
        remaining = lastIndex + 1 - start
        windows.append(_WindowPlan(start, remaining, remaining, True))
    return windows


def _handOverStock(
    compEntry,
    classname,
    comp,
    windowYears,
    handOverModel,
    handOverYears,
    interval,
    persistedStock,
    stockCommissioningThreshold,
):
    """Set a component's stockCommissioning for the window about to be built.

    Restores the stock accumulated so far, adds the previous window's implemented
    decisions of every handed-over year, and drops stock that is no longer in operation
    when the window starts. Mutates compEntry and persistedStock[classname][comp].
    """
    stock = copy.deepcopy(persistedStock[classname][comp])
    if handOverModel is not None:
        for year in handOverYears:
            commissioning = _previousCommissioning(handOverModel, comp, year)
            if (
                commissioning is not None
                and commissioning.sum() > stockCommissioningThreshold
            ):
                if stock is None:
                    stock = {}
                # rounded as rollingHorizonOptimization does, since the stock checks of
                # utils warn about the full float64 precision of a solved commissioning
                stock[year] = commissioning.astype(float).round(
                    _STOCK_COMMISSIONING_DIGITS
                )

        # Stock is decommissioned a technical lifetime after it became available, i.e.
        # after the lead time of its year; the rule utils.checkStockYears applies when the
        # window is built, and the one rollingHorizonOptimization prunes by.
        if stock is not None:
            technicalLifetime = pd.Series(
                compEntry["technicalLifetime"], dtype=float
            ).max()
            for year in [
                year
                for year in stock
                if year
                + _roundedStockLeadTimeInYears(
                    compEntry.get("leadTime"), year, interval
                )
                < windowYears[0] - technicalLifetime
            ]:
                stock.pop(year)
            # None, not an empty dict, is how a component without stock is described
            if not stock:
                stock = None

    compEntry["stockCommissioning"] = stock
    persistedStock[classname][comp] = copy.deepcopy(stock)


def _buildWindowComponentDict(
    compDict,
    windowYears,
    handOverModel,
    handOverYears,
    esM,
    persistedStock,
    stockCommissioningThreshold,
):
    """Build a window's component dict from the pathway's exported component dict.

    Mirrors rollingHorizon._buildIntervalComponentDict, except that the decisions of every
    year in handOverYears become stock, not only those of the year before the window.

    :return: the component dict the window's energy system model is built from
    :rtype: dict
    """
    windowCompDict = copy.deepcopy(dict(compDict))
    for classname in windowCompDict:
        constructorArguments = set(
            inspect.getfullargspec(getattr(fn, classname).__init__).args
        )
        for comp, compEntry in windowCompDict[classname].items():
            # exportToDict adds the aggregated time series of a clustered model, which
            # are no constructor arguments
            for parameterName in [
                name for name in compEntry if name not in constructorArguments
            ]:
                compEntry.pop(parameterName)
            _expandLeadTimeForPassedOnStock(compEntry, comp, esM)
            _handOverStock(
                compEntry,
                classname,
                comp,
                windowYears,
                handOverModel,
                handOverYears,
                esM.investmentPeriodInterval,
                persistedStock,
                stockCommissioningThreshold,
            )
            stockYears = (
                list(compEntry["stockCommissioning"])
                if compEntry["stockCommissioning"] is not None
                else []
            )
            _filterComponentParametersForInterval(
                compEntry, windowYears, stockYears, esM
            )
    return windowCompDict


def _solveDeterministicWindow(
    windowEsM, tree, windowStages, msspSettings, optimizeSettings
):
    """Solve a window on the single path of its tree, as an ordinary FINE model."""
    if len(tree.scenarioNames) != 1:
        raise ValueError(
            "The tree of the deterministic last window must have a single path, i.e. one "
            f"leaf, but it has {len(tree.scenarioNames)}: {tree.scenarioNames}."
        )
    windowStages = mssp._validateStageInvestmentPeriods(
        windowEsM, windowStages, tree.stageCount
    )
    mssp._validateTreeValues(windowEsM, tree, windowStages)

    timeSeriesAggregation = msspSettings.get("timeSeriesAggregation", False)
    model = mssp.buildScenarioEnergySystemModel(
        windowEsM,
        tree,
        tree.scenarioNames[0],
        windowStages,
        timeSeriesAggregation=timeSeriesAggregation,
        temporalAggregationSpecs=msspSettings.get("temporalAggregationSpecs"),
    )
    model.optimize(
        declaresOptimizationProblem=False,
        timeSeriesAggregation=timeSeriesAggregation,
        solver=msspSettings.get("solver", "gurobi"),
        **optimizeSettings,
    )
    return mssp.StochasticOptimizationResults(
        tree,
        None,
        {tree.scenarioNames[0]: model},
        model.objectiveValue,
        [list(windowEsM.investmentPeriods)],
        includeIsBuiltBinaryVariables=msspSettings.get(
            "includeIsBuiltBinaryVariables", True
        ),
    )


def realizedValues(esM, parameters, years):
    """Read parameter values of a model in the form a scenario tree node takes them.

    Meant for the root node of a window's tree, which holds the realized data: the window's
    energy system model, as handed to ``buildTree``, carries the realized data of the
    pathway for the window's years.

    :param esM: the energy system model to read from
    :type esM: EnergySystemModel instance

    :param parameters: the parameters to read, as ``{componentName: [parameterName, ...]}``
    :type parameters: dict

    :param years: the investment period years to read them for
    :type years: list of int

    :return: ``{componentName: {parameterName: {year: value}}}``. A parameter given per
        investment period contributes its value of each year, any other its single value.
    :rtype: dict
    """
    values = {}
    for compName, parameterNames in parameters.items():
        component = esM.getComponent(compName)
        for parameterName in parameterNames:
            value = getattr(component, parameterName)
            if isinstance(value, dict):
                missing = [year for year in years if year not in value]
                if missing:
                    raise ValueError(
                        f"'{parameterName}' of component '{compName}' is given per "
                        f"investment period but has no value for {missing}."
                    )
                perYear = {year: copy.deepcopy(value[year]) for year in years}
            else:
                perYear = {year: copy.deepcopy(value) for year in years}
            values.setdefault(compName, {})[parameterName] = perYear
    return values


def evaluateImplementedPathway(
    esM,
    implementedCommissioning,
    solver="gurobi",
    timeSeriesAggregation=False,
    temporalAggregationSpecs=None,
    optimizeSettings=None,
):
    """Solve the pathway with every commissioning decision fixed to the implemented one.

    The result is an ordinary solved energy system model over the implemented years, with
    the realized data of ``esM``: its objective value is the cost of the implemented
    pathway in hindsight, and its operation is the best one possible with that system. It
    can be compared with a perfect foresight or a myopic run of the same model.

    The commissioning of every component with a capacity variable is fixed by
    commissioningFix, in every implemented year and location (zero where nothing was
    commissioned); commissioningMin and commissioningMax are dropped. Binary is-built
    decisions are not fixed, they follow from the commissioning.

    :param esM: energy system model of the whole pathway, holding the realized data. It is
        neither optimized nor modified.
    :type esM: EnergySystemModel instance

    :param implementedCommissioning: the implemented decisions, as a
        SequentialStochasticResults instance or as its ``implementedCommissioning`` dict,
        ``{(modelingClass, componentName, location, year): value}``. Its years must be the
        first investment periods of esM.
    :type implementedCommissioning: SequentialStochasticResults instance or dict

    **Default arguments:**

    :param solver: the solver of the optimization
        |br| * the default value is 'gurobi'
    :type solver: string

    :param timeSeriesAggregation: whether the model is temporally aggregated first
        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param temporalAggregationSpecs: keyword arguments for aggregateTemporally
        |br| * the default value is None
    :type temporalAggregationSpecs: dict or None

    :param optimizeSettings: further keyword arguments for EnergySystemModel.optimize.
        declaresOptimizationProblem, timeSeriesAggregation and solver are set by this
        function.
        |br| * the default value is None
    :type optimizeSettings: dict or None

    :return: the solved energy system model of the implemented years
    :rtype: EnergySystemModel instance
    """
    optimizeSettings = _checkedSettings(
        "optimizeSettings", optimizeSettings, _RESERVED_OPTIMIZE_SETTINGS
    )
    if isinstance(implementedCommissioning, SequentialStochasticResults):
        implementedCommissioning = implementedCommissioning.implementedCommissioning
    years = sorted({key[3] for key in implementedCommissioning})
    pathwayYears = list(esM.investmentPeriodNames)
    if not years or years != pathwayYears[: len(years)]:
        raise ValueError(
            "The implemented decisions must cover the first investment periods of the "
            f"model without gaps, {pathwayYears[:1]} onwards; got the years {years}."
        )

    decisions = {}
    for (_, compName, loc, year), value in implementedCommissioning.items():
        decisions.setdefault(compName, {}).setdefault(year, {})[loc] = value

    esmDict, compDict = fn.dictIO.exportToDict(esM)
    evaluationCompDict = copy.deepcopy(dict(compDict))
    for classname in evaluationCompDict:
        for compName, compEntry in evaluationCompDict[classname].items():
            stockYears = (
                list(compEntry["stockCommissioning"])
                if compEntry.get("stockCommissioning") is not None
                else []
            )
            _filterComponentParametersForInterval(compEntry, years, stockYears, esM)
            if compName not in decisions:
                continue
            perYear = {}
            for year in years:
                values = pd.Series(decisions[compName].get(year, {}), dtype=float)
                if esM.getComponent(compName).dimension == Dimension.ONE:
                    values = values.reindex(sorted(esM.locations), fill_value=0.0)
                # solver noise must neither be negative nor exceed a capacity limit
                perYear[year] = (
                    values.fillna(0.0)
                    .clip(lower=0.0)
                    .round(_STOCK_COMMISSIONING_DIGITS)
                )
            compEntry["commissioningFix"] = perYear
            compEntry["commissioningMin"] = None
            compEntry["commissioningMax"] = None

    evaluationEsM = _buildIntervalEsm(
        esmDict, evaluationCompDict, years, len(years), esM
    )
    if timeSeriesAggregation:
        evaluationEsM.aggregateTemporally(**(temporalAggregationSpecs or {}))
    evaluationEsM.optimize(
        timeSeriesAggregation=timeSeriesAggregation, solver=solver, **optimizeSettings
    )
    return evaluationEsM


def sequentialStochasticOptimization(
    esM,
    windowLength,
    stageInvestmentPeriods,
    buildTree,
    lastYear=None,
    endOfHorizon="extend",
    msspSettings=None,
    optimizeSettings=None,
    stockCommissioningThreshold=1e-5,
    keepWindowResults=True,
):
    """Optimize a transformation pathway as a sequence of multi-stage stochastic programs.

    A window of ``windowLength`` investment periods is solved as a multi-stage stochastic
    program on the scenario tree ``buildTree`` returns for it. The decisions of its first
    stage are implemented: they are handed to the following windows as stock, keyed by the
    year they were taken in, so that their lead time still applies. The next window starts
    right after the first stage, so every window moves the pathway on by the length of the
    first stage, and every year is implemented by exactly one window.

    The pathway model ``esM`` holds the realized data, i.e. what actually happens. It is
    what the windows are cut from, and ``buildTree`` decides which of it a window's tree
    may see: typically the realized data for the first stage and forecasts (such as
    representative weather sequences) for the later stages.

    The windows are built like those of :func:`rollingHorizon.rollingHorizonOptimization`,
    so the same settings are refused (stochasticModel, pathwayBalanceLimit,
    annuityPerpetuity, pwlcfParameters), and a window's NPVcontributionRH is discounted onto
    the first year of the whole pathway. Only commissioning is handed over, not the binary
    is-built decisions, as in rollingHorizonOptimization.

    Under MPI, every process runs the loop. It has to build identical trees, and the
    windows' results hold only the scenarios of the process.

    **Required arguments:**

    :param esM: energy system model of the whole pathway, holding the realized data. It is
        neither optimized nor modified.
    :type esM: EnergySystemModel instance

    :param windowLength: number of investment periods of a window
    :type windowLength: strictly positive int

    :param stageInvestmentPeriods: the stages of a window, as one list of investment period
        indices of the window per stage, e.g. ``[[0, 1, 2], [3, 4, ..., 14]]``. The first
        stage is what a window implements, so its length is also the step from one window
        to the next.
    :type stageInvestmentPeriods: list of lists of int

    :param buildTree: function returning the scenario tree of a window, as a dict of nodes
        or a ScenarioTree (see :class:`ScenarioTree`). Values in the tree are keyed by the
        window's years. It is called as
        ``buildTree(windowEsM, windowYears=..., stageYears=..., realizedYears=...,
        deterministic=...)`` with

        * ``windowEsM``: the window's energy system model, not yet optimized, holding the
          realized data of the window's years and the stock handed over so far
        * ``windowYears``: the years of the window
        * ``stageYears``: the years of every stage, the first stage first
        * ``realizedYears``: the years implemented by the previous windows, i.e. the years
          that have been realized when the window is decided (empty for the first window)
        * ``deterministic``: True for the deterministic last window, whose tree must have a
          single path over the stages in ``stageYears``

        Every node has to give a value for every uncertain parameter, the root included;
        :func:`realizedValues` reads the realized ones from ``windowEsM``.
    :type buildTree: callable

    **Default arguments:**

    :param lastYear: last year whose decisions are implemented. Data beyond it are only
        used to give the last windows a full horizon.
        |br| * the default value is None, i.e. the last investment period of esM
    :type lastYear: int or None

    :param endOfHorizon: how the end of the pathway is handled.

        (a) 'extend': every window is a full stochastic window. The model must therefore
            reach windowLength minus the first stage's length (or more, if the number of
            years is not a multiple of it) investment periods beyond lastYear.
        (b) 'deterministic': stochastic windows are solved as long as a full window fits up
            to lastYear. The remaining years are then solved as one deterministic window,
            on the single path ``buildTree`` returns for it, and all of them are
            implemented.

        |br| * the default value is 'extend'
    :type endOfHorizon: string

    :param msspSettings: keyword arguments passed to optimizeMultiStageStochastic for every
        window (method, solver, solverOptions, mpisppyOptions, includeIsBuiltBinaryVariables,
        timeSeriesAggregation, temporalAggregationSpecs). The solver and the time series
        aggregation settings are also used for the deterministic window. baseModel, tree,
        stageInvestmentPeriods and processResults are set by this function.
        |br| * the default value is None
    :type msspSettings: dict or None

    :param optimizeSettings: keyword arguments passed to EnergySystemModel.optimize for the
        deterministic window (e.g. optimizationSpecs, threads). declaresOptimizationProblem,
        timeSeriesAggregation and solver are set by this function.
        |br| * the default value is None
    :type optimizeSettings: dict or None

    :param stockCommissioningThreshold: capacity below which a window's commissioning is
        not handed over as stock
        |br| * the default value is 1e-5
    :type stockCommissioningThreshold: positive float

    :param keepWindowResults: whether the solved models of every window are kept. Without
        them, only the implemented decisions, the trees and the objective values are kept,
        which saves memory for large models.
        |br| * the default value is True
    :type keepWindowResults: boolean

    :return: the windows and the implemented pathway
    :rtype: SequentialStochasticResults instance
    """
    utils.isStrictlyPositiveInt(windowLength)
    utils.isPositiveNumber(stockCommissioningThreshold)
    if endOfHorizon not in _END_OF_HORIZON_OPTIONS:
        raise ValueError(
            f"endOfHorizon must be one of {list(_END_OF_HORIZON_OPTIONS)}, got "
            f"{endOfHorizon!r}."
        )
    msspSettings = _checkedSettings(
        "msspSettings", msspSettings, _RESERVED_MSSP_SETTINGS
    )
    optimizeSettings = _checkedSettings(
        "optimizeSettings", optimizeSettings, _RESERVED_OPTIMIZE_SETTINGS
    )
    stages = _validatedStages(stageInvestmentPeriods, windowLength)
    stepLength = len(stages[0])
    _refuseUnsupportedSettings(esM)

    years = list(esM.investmentPeriodNames)
    if lastYear is None:
        lastYear = years[-1]
    if lastYear not in years:
        raise ValueError(
            f"lastYear {lastYear} is not an investment period of the model: {years}."
        )
    plans = _planWindows(
        len(years), years.index(lastYear), windowLength, stepLength, endOfHorizon
    )

    esmDict, compDict = fn.dictIO.exportToDict(esM)
    # as in rollingHorizonOptimization: every window discounts its NPVcontributionRH onto
    # the first year of the whole pathway, without modifying esM itself
    esmDict["rollingHorizonStartYear"] = (
        esM.rollingHorizonStartYear
        if esM.rollingHorizonStartYear is not None
        else esM.startYear
    )
    persistedStock = {
        classname: {
            comp: copy.deepcopy(compDict[classname][comp]["stockCommissioning"])
            for comp in compDict[classname]
        }
        for classname in compDict
    }

    windows = []
    handOverModel, handOverYears = None, []
    for plan in plans:
        windowYears = years[plan.start : plan.start + plan.length]
        utils.output(
            f"Building window {windowYears[0]}-{windowYears[-1]}"
            + (" (deterministic)" if plan.deterministic else ""),
            esM.verboseLogLevel,
            0,
        )
        windowCompDict = _buildWindowComponentDict(
            compDict,
            windowYears,
            handOverModel,
            handOverYears,
            esM,
            persistedStock,
            stockCommissioningThreshold,
        )
        if not keepWindowResults and windows:
            windows[-1].results = None
        windowEsM = _buildIntervalEsm(
            esmDict, windowCompDict, windowYears, len(windowYears), esM
        )

        if plan.deterministic:
            windowStages = [list(range(min(stepLength, plan.length)))]
            if plan.length > stepLength:
                windowStages.append(list(range(stepLength, plan.length)))
        else:
            windowStages = stages
        stageYears = [[windowYears[ip] for ip in stage] for stage in windowStages]
        realizedYears = [year for window in windows for year in window.implementedYears]

        tree = buildTree(
            windowEsM,
            windowYears=list(windowYears),
            stageYears=stageYears,
            realizedYears=realizedYears,
            deterministic=plan.deterministic,
        )
        if not isinstance(tree, mssp.ScenarioTree):
            tree = mssp.ScenarioTree(tree)

        if plan.deterministic:
            results = _solveDeterministicWindow(
                windowEsM, tree, windowStages, msspSettings, optimizeSettings
            )
        else:
            results = mssp.optimizeMultiStageStochastic(
                windowEsM,
                tree,
                stageInvestmentPeriods=windowStages,
                processResults=True,
                **msspSettings,
            )

        implementedYears = windowYears[: plan.implemented]
        implementedCommissioning = {
            (modelingClass, compName, loc, windowYears[ip]): value
            for (modelingClass, compName, loc, ip), value in (
                results.firstStageDecisions().items()
            )
            if windowYears[ip] in implementedYears
        }
        windows.append(
            WindowResult(
                startYear=windowYears[0],
                years=list(windowYears),
                stageYears=stageYears,
                implementedYears=list(implementedYears),
                deterministic=plan.deterministic,
                tree=tree,
                objectiveValue=results.objectiveValue,
                implementedCommissioning=implementedCommissioning,
                results=results,
            )
        )
        # equal in all scenarios, so any of them carries the first-stage decisions
        handOverModel = next(iter(results.scenarioModels.values()))
        handOverYears = list(implementedYears)

    if not keepWindowResults:
        windows[-1].results = None
    return SequentialStochasticResults(windows)
