"""Tests for the pure frame helpers of the result post-processing.

``fine.results.frames`` and ``fine.results.summary`` touch nothing but pandas frames and
scalars, so unlike the accessor tests these run on hand-built input and never solve a
model. That is the point of them being free functions.
"""

import numpy as np
import pandas as pd
import pytest

from fine.enums import Dimension
from fine.results import frames


def test_connectionLocationMap_covers_every_ordered_pair():
    mapC = frames.connectionLocationMap(["a", "b"])

    assert mapC == {
        "a_a": ("a", "a"),
        "a_b": ("a", "b"),
        "b_a": ("b", "a"),
        "b_b": ("b", "b"),
    }


def test_connectionLocationMap_is_independent_of_the_input_order():
    """The pairs are cached per location set, so two orderings must not disagree."""
    assert frames.connectionLocationMap(["b", "a"]) == frames.connectionLocationMap(
        ["a", "b"]
    )


def test_connectionLocationMap_hands_out_independent_dicts():
    """The cache is global, so callers must not be able to affect one another.

    Two models with the same locations ask for the map independently; a mutation by one
    must not reach the other.
    """
    first = frames.connectionLocationMap(["a", "b"])
    first["a_b"] = ("mutated", "mutated")

    assert frames.connectionLocationMap(["a", "b"])["a_b"] == ("a", "b")


def test_economicSummaryUnits_separates_annual_from_absolute_costs():
    units = frames.economicSummaryUnits("1e9 Euro")

    assert units["TAC"] == "[1e9 Euro/a]"
    assert units["capexCap"] == "[1e9 Euro/a]"
    assert units["NPVcontribution"] == "[1e9 Euro]"
    assert units["invest"] == "[1e9 Euro]"


def test_shapeOptimumResult_adds_a_time_dimension_for_one_dim_variables():
    sub = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["loc1", "loc2"], columns=[0, 1])

    series = frames.shapeOptimumResult(sub, "operation", True, Dimension.ONE)

    assert series.name == "operation"
    assert series.index.names == ["time", "location"]
    assert series.loc[(1, "loc2")] == 4.0


def test_shapeOptimumResult_splits_two_dim_connections():
    index = pd.MultiIndex.from_tuples([("loc1", "loc2"), ("loc2", "loc1")])
    sub = pd.DataFrame([[1.0], [2.0]], index=index, columns=[0])

    series = frames.shapeOptimumResult(sub, "operation", True, Dimension.TWO)

    assert series.index.names == ["time", "locationIn", "locationOut"]
    assert series.loc[(0, "loc1", "loc2")] == 1.0


def test_shapeOptimumResult_keeps_an_extra_index_level():
    """Part-load variables carry a discretization level that must survive the shaping."""
    index = pd.MultiIndex.from_tuples(
        [(0, "loc1"), (1, "loc1")], names=["discretizationIndex", "location"]
    )
    sub = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=[0, 1])

    series = frames.shapeOptimumResult(sub, "discretizationPoint", True, Dimension.ONE)

    assert series.index.names == ["time", "discretizationIndex", "location"]
    assert series.loc[(1, 1, "loc1")] == 4.0


def test_shapeOptimumResult_does_not_alias_its_input():
    sub = pd.DataFrame([[1.0]], index=["loc1"], columns=["cap"])

    series = frames.shapeOptimumResult(sub, "capacity", False, Dimension.ONE)
    series.iloc[0, 0] = 99.0

    assert sub.loc["loc1", "cap"] == 1.0


def test_nameResultSeries_names_the_index_per_dimension():
    series = pd.Series([1.0], index=["loc1"])

    named = frames.nameResultSeries(series, "capacity", Dimension.ONE)

    assert named.name == "capacity"
    assert named.index.name == "location"
    assert series.name is None, "the input must not be renamed in place"


def test_extractComponentResult_fills_missing_one_dim_components_with_nan():
    frame = pd.DataFrame([[1.0, 2.0]], index=["comp"], columns=["loc1", "loc2"])

    values = frames.extractComponentResult(
        frame, "absent", ["loc1", "loc2"], Dimension.ONE, {}
    )

    assert list(values.index) == ["loc1", "loc2"]
    assert values.isna().all()


def test_extractComponentResult_reindexes_onto_the_sorted_locations():
    frame = pd.DataFrame([[2.0]], index=["comp"], columns=["loc2"])

    values = frames.extractComponentResult(
        frame, "comp", ["loc2", "loc1"], Dimension.ONE, {}
    )

    assert list(values.index) == ["loc1", "loc2"]
    assert np.isnan(values["loc1"])
    assert values["loc2"] == 2.0


def test_extractComponentResult_drops_empty_two_dim_rows():
    mapC = frames.connectionLocationMap(["loc1", "loc2"])
    frame = pd.DataFrame(
        [[np.nan, np.nan, np.nan, np.nan]],
        index=["comp"],
        columns=["loc1_loc1", "loc1_loc2", "loc2_loc1", "loc2_loc2"],
    )

    assert (
        frames.extractComponentResult(
            frame, "comp", ["loc1", "loc2"], Dimension.TWO, mapC
        )
        is None
    )


def test_extractComponentResult_splits_two_dim_connections():
    mapC = frames.connectionLocationMap(["loc1", "loc2"])
    frame = pd.DataFrame(
        [[np.nan, 5.0, np.nan, np.nan]],
        index=["comp"],
        columns=["loc1_loc1", "loc1_loc2", "loc2_loc1", "loc2_loc2"],
    )

    values = frames.extractComponentResult(
        frame, "comp", ["loc1", "loc2"], Dimension.TWO, mapC
    )

    assert list(values.index) == [("loc1", "loc2")]
    assert values.iloc[0] == 5.0


@pytest.fixture
def summarySkeleton():
    index = pd.MultiIndex.from_tuples(
        [("comp", "operation", "[kW*h]"), ("comp", "opexOp", "[Euro/a]")],
        names=["Component", "Property", "Unit"],
    )
    return pd.DataFrame(index=index, columns=["loc1", "loc2"])


def test_writeOperationSummaryRows_writes_rows_and_reports_the_frames(summarySkeleton):
    operation = pd.DataFrame([[1.0, 2.0]], index=["comp"], columns=["loc1", "loc2"])

    byProp = frames.writeOperationSummaryRows(
        summarySkeleton, [("operation", operation, "[kW*h]")]
    )

    assert byProp == {"operation": operation}
    assert summarySkeleton.loc[("comp", "operation", "[kW*h]"), "loc2"] == 2.0


def test_writeOperationSummaryRows_resolves_a_callable_unit(summarySkeleton):
    """A per-component unit is resolved by calling it with the component name."""
    operation = pd.DataFrame([[1.0, 2.0]], index=["comp"], columns=["loc1", "loc2"])
    seen = []

    def unit(compName):
        seen.append(compName)
        return "[kW*h]"

    frames.writeOperationSummaryRows(summarySkeleton, [("operation", operation, unit)])

    assert seen == ["comp"]
    assert summarySkeleton.loc[("comp", "operation", "[kW*h]"), "loc1"] == 1.0


def test_writeOperationSummaryRows_skips_absent_and_empty_frames(summarySkeleton):
    byProp = frames.writeOperationSummaryRows(
        summarySkeleton,
        [("operation", None, "[kW*h]"), ("opexOp", pd.DataFrame(), "[Euro/a]")],
    )

    assert set(byProp) == {"operation", "opexOp"}
    assert summarySkeleton.isna().all().all()
