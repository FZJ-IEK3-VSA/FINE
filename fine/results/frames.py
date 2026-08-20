"""Frame shaping helpers for the result post-processing.

Everything here runs after the solver and touches nothing but pandas frames and scalars -
no pyomo, no component model, no EnergySystemModel. That makes these functions testable
on hand-built frames, without solving a model first. The modeling classes reach them
through thin adapters on :class:`fine.results.componentResults.ComponentResultsMixin`.
"""

from functools import lru_cache

import numpy as np
import pandas as pd

from fine.enums import Dimension


@lru_cache(maxsize=8)
def _connectionLocationPairs(locations):
    """Build the connection pairs for a hashable location set (cached).

    Returns an immutable tuple rather than the mapping itself: the cache is global, so a
    mutable value would be shared by every model and energy system with the same locations,
    and a mutation in one place would reach all of them.
    """
    ordered = sorted(locations)
    return tuple((l1 + "_" + l2, (l1, l2)) for l1 in ordered for l2 in ordered)


def connectionLocationMap(locations):
    """Build the ``"locIn_locOut" -> (locIn, locOut)`` map used to split 2-dim connections.

    The pairs are cached per location set, so the O(locations^2) key construction does not
    repeat on every summary or export call. Each call still gets its own dict, so callers
    cannot affect one another.

    :param locations: the energy system's locations.
    :type locations: iterable of strings

    :return: mapping of connection key to its ``(locationIn, locationOut)`` pair.
    :rtype: dict
    """
    return dict(_connectionLocationPairs(frozenset(locations)))


def economicSummaryUnits(costUnit):
    """Property -> unit string for the derived economic summary rows.

    Single source for these units; consumed by both the optimization summary and the
    export, so a unit cannot drift between the two.

    :param costUnit: the energy system's cost unit (e.g. ``"1e9 Euro"``).
    :type costUnit: string

    :return: ordered ``{property: unitString}`` mapping.
    :rtype: dict
    """
    perA = "[" + costUnit + "/a]"
    cost = "[" + costUnit + "]"
    return {
        "capexCap": perA,
        "capexIfBuilt": perA,
        "opexCap": perA,
        "opexIfBuilt": perA,
        "TAC": perA,
        "NPVcontribution": cost,
        "invest": cost,
        "investLifetimeExtension": cost,
        "revenueLifetimeShorteningResale": cost,
    }


def shapeOptimumResult(sub, name, timeDependent, dimension):
    """Shape a single component's optimum frame into a ``to_xarray``-ready ``Series``.

    Reproduces the per-case index handling the export applied to ``getOptimalValues`` output:
    time-dependent rows gain a ``time`` dimension; 2-dim rows are split into
    ``(locationIn, locationOut)`` (the time-independent 2-dim case keeps the historical
    transpose). The shaping uses the variable's own ``dimension``, which may differ from the
    component's (e.g. the LOPF phase angle is 1-dim on a 2-dim component).

    Variables that carry an extra index level beyond ``location`` (e.g. the part-load
    discretization point/segment variables, indexed by ``(discretizationIndex, location)``
    per component) keep that level so each variable exports under its own name instead of
    colliding on an anonymous stacked column. The extra level names are propagated from the
    frame (labelled in :func:`fine.utils.formatOptimizationOutput`) rather than re-derived
    here.

    :param sub: the component slice ``frame.loc[component]``.
    :param name: variable name (becomes the data variable name).
    :param timeDependent: whether the variable carries a ``time`` dimension.
    :param dimension: ``"1dim"`` or ``"2dim"`` shaping to apply to this variable.

    :rtype: pandas.Series
    """
    if timeDependent and dimension == Dimension.ONE:
        subT = sub.T
        if subT.columns.nlevels == 1:
            series = subT.stack()
            series.index = series.index.rename(["time", "location"])
        else:
            # extra index levels (e.g. discretizationIndex) sit before location; stack
            # every column level so nothing is lost or collides on export, keeping the
            # level names set by formatOptimizationOutput.
            series = subT.stack(list(range(subT.columns.nlevels)))
            extraNames = list(sub.index.names[:-1])
            series.index = series.index.rename(["time", *extraNames, "location"])
    elif timeDependent and dimension == Dimension.TWO:
        series = sub.stack()
        series.index = series.index.rename(["locationIn", "locationOut", "time"])
        series = series.reorder_levels(["time", "locationIn", "locationOut"])
    elif not timeDependent and dimension == Dimension.ONE:
        series = sub.rename_axis("location")
    else:  # time-independent 2-dim
        series = sub.T.stack()
        series.index = series.index.rename(["locationIn", "locationOut"])
    series = series.copy()
    series.name = name
    return series


def nameResultSeries(series, name, dimension):
    """Set the variable name and dimension-specific index names for the export.

    :param series: per-component result series (1dim: index = locations; 2dim: index =
        ``(locationIn, locationOut)`` tuples).
    :param name: variable name (becomes the data variable name after ``to_xarray``).
    :param dimension: the component's dimension.

    :return: the same series, ready for ``.to_xarray()``.
    :rtype: pandas.Series
    """
    series = series.copy()
    series.name = name
    if dimension == Dimension.ONE:
        series.index = series.index.rename("location")
    else:
        series.index = series.index.rename(["locationIn", "locationOut"])
    return series


def extractComponentResult(frame, compName, locations, dimension, mapC):
    """Shape a class-level result frame into the per-component export values.

    :param frame: frame indexed by component, columns are locations (1dim) or connections
        (2dim); may be ``None``.
    :param compName: component name to extract.
    :param locations: the energy system's locations (1-dim rows are reindexed onto them).
    :param dimension: the component's dimension.
    :param mapC: mapping ``"locIn_locOut" -> (locIn, locOut)`` for the 2-dim split.

    :return: per-location ``Series`` (1dim, NaN-filled), ``(locationIn, locationOut)``
        ``Series`` (2dim, NaN dropped) or ``None`` to skip the variable.
    :rtype: pandas.Series or None
    """
    if dimension == Dimension.ONE:
        locations = sorted(locations)
        if frame is None or compName not in frame.index:
            return pd.Series(np.nan, index=locations)
        return frame.loc[compName].reindex(locations)
    # 2dim: split connection columns into (locationIn, locationOut), dropping NaN
    if frame is None or compName not in frame.index:
        return None
    row = frame.loc[compName]
    index, values = [], []
    for connection, value in row.items():
        if pd.isna(value):
            continue
        index.append(mapC[connection])
        values.append(value)
    if not index:
        return None
    return pd.Series(values, index=pd.MultiIndex.from_tuples(index))


def writeOperationSummaryRows(optSummary, frames):
    """Write aggregated operation rows into a summary skeleton, in place.

    :param optSummary: summary skeleton with a ``(Component, Property, Unit) x columns``
        MultiIndex; filled in place.
    :param frames: ordered ``(property, frame, unit)`` triples, where ``unit`` is either a
        unit string or a callable mapping a component name to one. A ``None`` or empty
        frame contributes no rows.

    :return: the ``{property: frame}`` mapping (so callers can reuse the aggregated frames,
        e.g. for the storage charge/discharge warning).
    :rtype: dict
    """
    framesByProp = {}
    for prop, frame, unit in frames:
        framesByProp[prop] = frame
        if frame is None or frame.empty:
            continue
        optSummary.loc[
            [(ix, prop, unit(ix) if callable(unit) else unit) for ix in frame.index],
            frame.columns,
        ] = frame.values
    return framesByProp
