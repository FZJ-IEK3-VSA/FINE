"""Stepwise import/export cost and revenue curves. Dont vary pro region, the same for verey locations. """

import math
import re
import warnings

import numpy as np
import pandas as pd

from fine import utils
from fine.sourceSink import Sink, Source


# Matches the "<name>_segment<i>" naming convention
_SEGMENT_NAME_PATTERN = re.compile(r"^(?P<groupName>.+)_segment(?P<index>\d+)$")


_RESERVED_KWARGS = (
    "hasCapacityVariable",
    "commodityCost",
    "commodityRevenue",
    "commodityCostTimeSeries",
    "commodityRevenueTimeSeries",
    "operationRateMax",
    "operationRateFix",
)


def addStepwiseSource(esM, name, commodity, costCurve, **kwargs):
    """Add a stepwise (piecewise-linear, monotonic) import cost curve to an esM.

    Constructs one :class:`fine.sourceSink.Source` per segment of ``costCurve``, each
    bounded by that segment's width (via ``operationRateMax``) and priced at that
    segment's price (via ``commodityCost``/``commodityCostTimeSeries``). The segment
    components are added to ``esM``; :func:`collapseStepwiseResults` finds them again
    from their ``f"{name}_segment{i}"`` names to re-aggregate their results.

    :param name: name of the stepwise component group. Segment components are named
        ``f"{name}_segment{i}"``.

    :param costCurve: the cost curve, checked and normalized by
        :func:`fine.utils.checkCostCurve` (``isSource=True``: prices must be
        non-decreasing across segments).
    :type costCurve: list of (width, price) tuples, or a pandas DataFrame with a
        2-level column MultiIndex ``(segment, {"width","price"})``, one row per
        timestep - see :func:`fine.utils.checkCostCurve`.

    :param kwargs: any other :class:`fine.sourceSink.Source` constructor argument,
        forwarded to every segment component. 

    :return: the segment component names, in segment order.
    :rtype: list of strings
    """
    return _addStepwiseComponent(esM, name, commodity, costCurve, isSource=True, **kwargs)


def addStepwiseSink(esM, name, commodity, costCurve, **kwargs):
    return _addStepwiseComponent(esM, name, commodity, costCurve, isSource=False, **kwargs)


def _addStepwiseComponent(esM, name, commodity, costCurve, isSource, **kwargs):
    """Shared implementation of :func:`addStepwiseSource`/:func:`addStepwiseSink`."""
    for reserved in _RESERVED_KWARGS: # cannot pass these kwargs to the Source/Sink constructor, they are derived from costCurve
        if reserved in kwargs:
            componentName = "addStepwiseSource" if isSource else "addStepwiseSink"
            raise ValueError(
                f"'{reserved}' cannot be passed to {componentName}; it is derived from "
                "costCurve, or fixed to model a quantity-tiered price."
            )

    widths, prices, isTimeVarying = utils.checkCostCurve(esM, costCurve, isSource)

    if not hasattr(esM, "checkNamesStepwise"): #to ensure no <name>_segment<i> othan those created by addStepwiseSource/Sink already exists in the esM
        esM.checkNamesStepwise = []
    foreignSegments = [
        compName
        for compName, modelingClassName in esM.componentNames.items()
        if modelingClassName == "SourceSinkModel"
        and (match := _SEGMENT_NAME_PATTERN.match(compName)) is not None
        and match.group("groupName") not in esM.checkNamesStepwise
    ]
    if foreignSegments and esM.verboseLogLevel < 2:
        warnings.warn(
            f"{foreignSegments} follow the '<name>_segment<i>' naming of stepwise segments "
            "but were not created by addStepwiseSource/addStepwiseSink in this session. "
            "fn.collapseStepwiseResults(esM) will still merge them as stepwise segments "
            "under '<name>'. Rename them if they are not Source/Sink stepwise segments."

        )

    noSegments = len(widths) # width is list of floats or list of pd.Series, prices is list of floats or list of pd.Series
    priceParam = "commodityCost" if isSource else "commodityRevenue"
    priceTimeSeriesParam = (
        "commodityCostTimeSeries" if isSource else "commodityRevenueTimeSeries"
    )
    componentClass = Source if isSource else Sink

    segmentNames = []
    for i in range(noSegments):
        segmentName = f"{name}_segment{i}"
        segmentKwargs = dict(kwargs)
        segmentKwargs["hasCapacityVariable"] = False

        if not isTimeVarying:
            width = widths[i]
            if not math.isinf(width):
                segmentKwargs["operationRateMax"] = pd.DataFrame(
                    {loc: [width] * esM.numberOfTimeSteps for loc in esM.locations}
                )
            segmentKwargs[priceParam] = prices[i]
        else:
            widthSeries, priceSeries = widths[i], prices[i]
            if not np.isinf(widthSeries.to_numpy()).any(): # if the segment has a finite width at any time step, set operationRateMax to that width series
                segmentKwargs["operationRateMax"] = pd.DataFrame(
                    {loc: widthSeries.to_numpy() for loc in esM.locations},
                    index=widthSeries.index,
                )
            segmentKwargs[priceTimeSeriesParam] = pd.DataFrame(
                {loc: priceSeries.to_numpy() for loc in esM.locations},
                index=priceSeries.index,
            )

        esM.add(componentClass(esM, segmentName, commodity, **segmentKwargs))
        segmentNames.append(segmentName)

    esM.checkNamesStepwise.append(name)

    if esM.verboseLogLevel < 2:
        warnings.warn(
            f"'{name}' was added as {noSegments} separate Source/Sink components "
            f"({', '.join(segmentNames)}). FINE's native esM.getOptimizationSummary/"
            "getOptimalValues will show these segments split apart until "
            "fn.collapseStepwiseResults(esM) is called after optimizing, "
            f"which combines them into a single row under '{name}'. Use "
            "fn.plotStepwiseOperation to plot the combined operation."
        )

    return segmentNames


def _findStepwiseGroups(esM):
    """Return ``{groupName: segmentNames}`` for every stepwise group of ``esM``, found
    from the ``f"{groupName}_segment{i}"`` names of its Source/Sink components."""
    segmentsByGroup = {}
    for compName, modelingClassName in esM.componentNames.items():
        if modelingClassName != "SourceSinkModel":
            continue
        match = _SEGMENT_NAME_PATTERN.match(compName)
        if match is None:
            continue
        segmentsByGroup.setdefault(match.group("groupName"), []).append(
            (int(match.group("index")), compName)
        )
    return {
        groupName: [compName for _, compName in sorted(indexedNames)]
        for groupName, indexedNames in segmentsByGroup.items()
    }


def _collapseOptSummary(summary, groups):
    """Collapse the segment rows of one investment period's optimization summary.

    "summary" is the DataFrame stored in "ComponentModel._optSummary" for one investment
    period ("_optSummary[ipName]"), indexed by "(Component, Property, Unit)". For every
    group in "groups" (result from _findStepwiseGroups):

    - the rows of the group's segments are summed property by property ("TAC", "operation"...)
    - these rows are replaced by a single row named after the group name (e.g. "gridImport").
    - the rest stays the same

    :return: a new DataFrame with the same format as "summary", where the segment rows of
        each group are replaced by the summed rows named after the group, sorted by index.
        If no segment rows are found (e.g. already collapsed), "summary" itself is returned.
    """
    allSegmentNames = {n for segmentNames in groups.values() for n in segmentNames}
    remaining = summary.loc[
        ~summary.index.get_level_values("Component").isin(allSegmentNames)
    ]
    collapsedGroups = []
    for groupName, segmentNames in groups.items():
        ownRows = summary.loc[
            summary.index.get_level_values("Component").isin(segmentNames)
        ]
        if ownRows.empty:
            continue
        aggregated = ownRows.groupby(level=["Property", "Unit"]).sum()
        collapsedGroups.append(pd.concat({groupName: aggregated}, names=["Component"]))
    if not collapsedGroups:
        return summary
    combined = pd.concat([remaining, *collapsedGroups])
    return combined.reorder_levels(["Component", "Property", "Unit"]).sort_index()


def _collapseOperation(values, groups):
    """Collapse the segment rows of one investment period's operation time series, following 
    the same logic as :func:`_collapseOptSummary`.
    """
    if values is None:
        return values
    allSegmentNames = {n for segmentNames in groups.values() for n in segmentNames}
    remaining = values.loc[~values.index.get_level_values(0).isin(allSegmentNames)]
    collapsedGroups = []
    for groupName, segmentNames in groups.items():
        ownRows = values.loc[values.index.get_level_values(0).isin(segmentNames)]
        if ownRows.empty:
            continue
        collapsed = ownRows.groupby(level=1).sum()
        collapsed.index = pd.MultiIndex.from_product(
            [[groupName], collapsed.index], names=["component", "location"]
        )
        collapsedGroups.append(collapsed)
    if not collapsedGroups:
        return values
    return pd.concat([remaining, *collapsedGroups]).sort_index()



def collapseStepwiseResults(esM):
    """Permanently collapse every stepwise group's results, in place. This relies on :func:`_findStepwiseGroups`.
     This is the function to call right after ``esM.optimize()``, or on an esM reloaded from netCDF.

    .. note::
        The groups are found from the component names only, so that this also works on an
        esM reloaded from netCDF, where the names are the only information left. As a
        consequence, every Source/Sink component named ``<name>_segment<i>`` is treated as
        a stepwise segment and merged under ``<name>``, even if it was not created by
        :func:`addStepwiseSource`/:func:`addStepwiseSink`. Do not use this naming for
        other Source/Sink components. :func:`addStepwiseSource`/:func:`addStepwiseSink`
        warn about such components, but only about those already added at that point.

    :return: ``["SourceSinkModel"]`` if the esM has stepwise groups, ``[]`` otherwise.
    """
    groups = _findStepwiseGroups(esM)
    if not groups:
        return []
    model = esM.componentModelingDict["SourceSinkModel"]
    for ipName in esM.investmentPeriodNames:
        if ipName in model._optSummary:
            model._optSummary[ipName] = _collapseOptSummary(model._optSummary[ipName], groups)
        if ipName in model._operationVariablesOptimum:
            model._operationVariablesOptimum[ipName] = _collapseOperation(
                model._operationVariablesOptimum[ipName], groups
            )
    model._convertOptimalValueNames(esM)
    return ["SourceSinkModel"]

