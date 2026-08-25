"""Tests for the optimization summary assembly.

``fine.results.summary.buildOptimizationSummary`` renders the frames that the result
pipeline has already produced into the ``(Component, Property, Unit) x locations`` summary.
It needs no pyomo and no solved model, only a raw results dict, so these tests hand it one
directly instead of optimizing first. That covers cases which are awkward to provoke
through a real model - an absent capacity frame, a component sitting at its big-M bound -
and it runs in milliseconds.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from fine.results.summary import buildOptimizationSummary

LOCATIONS = ["loc1", "loc2"]


@contextmanager
def noWarning():
    """Turn UserWarnings into errors, so an unexpected warning fails the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        yield


class FakeComponent:
    """The component attributes the summary reads."""

    def __init__(
        self,
        commodityUnit="kW",
        physicalUnit="kW_el",
        hasIsBuiltBinaryVariable=False,
        processedCapacityMax=None,
        bigM=1000.0,
    ):
        self.commodityUnit = commodityUnit
        self.physicalUnit = physicalUnit
        self.hasIsBuiltBinaryVariable = hasIsBuiltBinaryVariable
        self.processedCapacityMax = processedCapacityMax
        self.bigM = bigM


class FakeEsM:
    """The energy system attributes the summary reads."""

    def __init__(self, investmentPeriodNames=("2020",), verboseLogLevel=0):
        self.investmentPeriodNames = list(investmentPeriodNames)
        self.investmentPeriods = list(range(len(self.investmentPeriodNames)))
        self.locations = LOCATIONS
        self.costUnit = "1e9 Euro"
        self.verboseLogLevel = verboseLogLevel


def frame(values, components=("comp",), columns=LOCATIONS):
    return pd.DataFrame(values, index=list(components), columns=list(columns))


def designFrames(capacity=None, commissioning=None, decommissioning=None, isBuilt=None):
    """Return the four 1-dim design frames extractRawResults produces."""
    return {
        "capacity": frame([[10.0, 20.0]]) if capacity is None else capacity,
        "commissioning": frame([[1.0, 2.0]])
        if commissioning is None
        else commissioning,
        "decommissioning": (
            frame([[0.5, 0.0]]) if decommissioning is None else decommissioning
        ),
        "isBuilt": frame([[1.0, 0.0]]) if isBuilt is None else isBuilt,
    }


def build(
    components=None,
    raw=None,
    raw1dim=None,
    esM=None,
    plantUnit="commodityUnit",
    unitApp="",
):
    components = components or {"comp": FakeComponent()}
    esM = esM or FakeEsM()
    ipNames = esM.investmentPeriodNames
    raw = raw if raw is not None else {ip: {} for ip in ipNames}
    raw1dim = raw1dim if raw1dim is not None else {ip: designFrames() for ip in ipNames}
    return buildOptimizationSummary(
        components, raw, raw1dim, esM, LOCATIONS, plantUnit, unitApp
    )


# --------------------------------------------------------------------------------------
# investment periods
# --------------------------------------------------------------------------------------


def test_a_single_investment_period_yields_one_summary():
    summary = build()

    assert list(summary) == ["2020"]
    assert summary["2020"].loc[("comp", "capacity", "[kW]"), "loc2"] == 20.0


def test_every_investment_period_gets_its_own_summary():
    """The frames are read per period; one period's values must not leak into another."""
    esM = FakeEsM(investmentPeriodNames=("2020", "2025"))
    raw1dim = {
        "2020": designFrames(capacity=frame([[10.0, 20.0]])),
        "2025": designFrames(capacity=frame([[30.0, 40.0]])),
    }

    summary = build(esM=esM, raw1dim=raw1dim)

    assert list(summary) == ["2020", "2025"]
    assert summary["2020"].loc[("comp", "capacity", "[kW]"), "loc1"] == 10.0
    assert summary["2025"].loc[("comp", "capacity", "[kW]"), "loc1"] == 30.0


def test_the_summary_columns_are_the_sorted_index_columns():
    summary = build()

    assert list(summary["2020"].columns) == ["loc1", "loc2"]


# --------------------------------------------------------------------------------------
# absent frames
# --------------------------------------------------------------------------------------


def test_an_absent_capacity_frame_leaves_the_row_empty():
    """A component without a capacity variable still gets its rows, filled with NaN."""
    raw1dim = {"2020": designFrames()}
    raw1dim["2020"]["capacity"] = None

    summary = build(raw1dim=raw1dim)

    assert summary["2020"].loc[("comp", "capacity", "[kW]")].isna().all()
    # the isBuilt row is independent of the capacity frame and is still written
    assert summary["2020"].loc[("comp", "isBuilt", "[-]"), "loc1"] == 1.0


def test_an_absent_binary_frame_leaves_the_isBuilt_row_empty():
    raw1dim = {"2020": designFrames()}
    raw1dim["2020"]["isBuilt"] = None

    summary = build(raw1dim=raw1dim)

    assert summary["2020"].loc[("comp", "isBuilt", "[-]")].isna().all()
    assert summary["2020"].loc[("comp", "capacity", "[kW]"), "loc1"] == 10.0


def test_commissioning_rows_are_dropped_without_capacity_and_decommissioning():
    """The single-year case: neither frame exists, so the rows stay empty."""
    raw1dim = {"2020": designFrames()}
    raw1dim["2020"]["capacity"] = None
    raw1dim["2020"]["decommissioning"] = None

    summary = build(raw1dim=raw1dim)

    assert summary["2020"].loc[("comp", "commissioning", "[kW]")].isna().all()
    assert summary["2020"].loc[("comp", "decommissioning", "[kW]")].isna().all()


# --------------------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------------------


def test_the_plant_unit_attribute_selects_the_design_row_unit():
    summary = build(plantUnit="physicalUnit")

    assert ("comp", "capacity", "[kW_el]") in summary["2020"].index
    assert ("comp", "capacity", "[kW]") not in summary["2020"].index


def test_the_storage_suffix_is_appended_to_every_design_row():
    """StorageModel reports capacities in commodityUnit*h, commissioning included."""
    summary = build(unitApp="*h")

    index = summary["2020"].index
    for prop in ("capacity", "commissioning", "decommissioning"):
        assert ("comp", prop, "[kW*h]") in index
    # the isBuilt row carries a fixed unit and must not gain the suffix
    assert ("comp", "isBuilt", "[-]") in index


def test_each_component_carries_its_own_plant_unit():
    components = {
        "electrolyzer": FakeComponent(commodityUnit="kW_el"),
        "pipeline": FakeComponent(commodityUnit="kW_H2"),
    }
    raw1dim = {
        "2020": designFrames(
            capacity=frame([[10.0, 20.0], [30.0, 40.0]], components=components),
            commissioning=frame([[1.0, 2.0], [3.0, 4.0]], components=components),
            decommissioning=frame([[0.0, 0.0], [0.0, 0.0]], components=components),
            isBuilt=frame([[1.0, 1.0], [1.0, 1.0]], components=components),
        )
    }

    summary = build(components=components, raw1dim=raw1dim)

    assert summary["2020"].loc[("electrolyzer", "capacity", "[kW_el]"), "loc1"] == 10.0
    assert summary["2020"].loc[("pipeline", "capacity", "[kW_H2]"), "loc1"] == 30.0


def test_the_economic_units_follow_the_cost_unit():
    index = build()["2020"].index

    assert ("comp", "TAC", "[1e9 Euro/a]") in index
    assert ("comp", "invest", "[1e9 Euro]") in index


# --------------------------------------------------------------------------------------
# big-M warning
# --------------------------------------------------------------------------------------


def test_a_capacity_at_the_big_M_bound_warns():
    components = {"comp": FakeComponent(hasIsBuiltBinaryVariable=True, bigM=100.0)}
    raw1dim = {"2020": designFrames(capacity=frame([[10.0, 95.0]]))}

    with pytest.warns(UserWarning, match="close or equal to the chosen Big M"):
        build(components=components, raw1dim=raw1dim)


def test_no_big_M_warning_below_the_threshold():
    components = {"comp": FakeComponent(hasIsBuiltBinaryVariable=True, bigM=100.0)}
    raw1dim = {"2020": designFrames(capacity=frame([[10.0, 89.0]]))}

    with noWarning():
        build(components=components, raw1dim=raw1dim)


def test_no_big_M_warning_when_a_capacity_max_replaced_it():
    """BigM is substituted by capacityMax in the constraint, so the check does not apply."""
    components = {
        "comp": FakeComponent(
            hasIsBuiltBinaryVariable=True, bigM=100.0, processedCapacityMax=95.0
        )
    }
    raw1dim = {"2020": designFrames(capacity=frame([[10.0, 95.0]]))}

    with noWarning():
        build(components=components, raw1dim=raw1dim)


def test_no_big_M_warning_when_the_log_level_silences_it():
    components = {"comp": FakeComponent(hasIsBuiltBinaryVariable=True, bigM=100.0)}
    raw1dim = {"2020": designFrames(capacity=frame([[10.0, 95.0]]))}

    with noWarning():
        build(
            components=components,
            raw1dim=raw1dim,
            esM=FakeEsM(verboseLogLevel=2),
        )


# --------------------------------------------------------------------------------------
# economic rows
# --------------------------------------------------------------------------------------


def test_economic_frames_are_written_under_their_unit():
    raw = {
        "2020": {
            "TAC": frame([[1.5, 2.5]]),
            "invest": frame([[100.0, 200.0]]),
            "NPVcontribution": frame([[7.0, 8.0]]),
        }
    }

    summary = build(raw=raw)["2020"]

    assert summary.loc[("comp", "TAC", "[1e9 Euro/a]"), "loc2"] == 2.5
    assert summary.loc[("comp", "invest", "[1e9 Euro]"), "loc1"] == 100.0
    assert summary.loc[("comp", "NPVcontribution", "[1e9 Euro]"), "loc2"] == 8.0


def test_an_economic_frame_that_is_absent_leaves_its_row_empty():
    summary = build(raw={"2020": {"TAC": frame([[1.5, 2.5]])}})["2020"]

    assert summary.loc[("comp", "opexCap", "[1e9 Euro/a]")].isna().all()


def test_an_empty_economic_frame_is_skipped():
    """The empty frame writes no value, so the fold normalizes the TAC row to 0."""
    summary = build(raw={"2020": {"TAC": pd.DataFrame()}})["2020"]

    assert (summary.loc[("comp", "TAC", "[1e9 Euro/a]")] == 0).all()


def test_the_folded_rows_are_normalized_to_zero():
    """The former inline implementation wrote the TAC and NPVcontribution rows as a
    groupby sum over the summary, which turned the cells the derived frames do not cover
    into 0. Uncovered ``loc2`` must therefore read 0, not NaN, while an unfolded row such
    as ``invest`` keeps its NaN.
    """
    raw = {
        "2020": {
            "TAC": frame([[1.5, np.nan]]),
            "NPVcontribution": frame([[7.0, np.nan]]),
            "invest": frame([[100.0, np.nan]]),
        }
    }

    summary = build(raw=raw)["2020"]

    assert summary.loc[("comp", "TAC", "[1e9 Euro/a]"), "loc2"] == 0
    assert summary.loc[("comp", "NPVcontribution", "[1e9 Euro]"), "loc2"] == 0
    assert pd.isna(summary.loc[("comp", "invest", "[1e9 Euro]"), "loc2"])


def test_lifetime_corrections_are_written_cell_wise():
    """These rows are written per cell to keep their numpy scalar dtype."""
    raw = {
        "2020": {
            "investLifetimeExtension": frame([[np.float64(3.0), np.float64(4.0)]]),
            "revenueLifetimeShorteningResale": frame([[1.0, 2.0]]),
        }
    }

    summary = build(raw=raw)["2020"]

    assert summary.loc[("comp", "investLifetimeExtension", "[1e9 Euro]"), "loc1"] == 3.0
    assert (
        summary.loc[("comp", "revenueLifetimeShorteningResale", "[1e9 Euro]"), "loc2"]
        == 2.0
    )


def test_design_props_are_not_overwritten_by_the_economic_loop():
    """A raw results entry that shares a design property name must not win over the frame."""
    raw = {"2020": {"capacity": frame([[999.0, 999.0]])}}

    summary = build(raw=raw)["2020"]

    assert summary.loc[("comp", "capacity", "[kW]"), "loc1"] == 10.0
