"""Windows of the rolling horizon and the sequential driver against lead times.

A decision a window implements may become available only after the end of that window.
The drivers warn about it before solving, and refuse a lower or fixed commissioning bound
(commissioningMin/Fix, or a componentLimit of type "commissioning") that no window can
enforce. The evaluation of an implemented pathway fixes the decisions themselves, since
the last ones become available after the evaluated years. These tests need the lead-time
feature together with the rolling horizon and the sequential driver.
"""

import pandas as pd
import pyomo.environ as pyomo
import pytest

import fine as fn
from fine.expansionModules.rollingHorizon import (
    _checkWindowsAgainstLeadTimes,
    rollingHorizonOptimization,
)
from fine.utils import ImplementedSolvers

_NODE = "node1"
_YEARS = [2020, 2025, 2030, 2035]


def _perYear(values, make):
    return {year: make(values.get(year, 0)) for year in _YEARS}


def _timeSeries(value):
    return pd.DataFrame([value], index=[0], columns=[_NODE])


def _locSeries(value):
    return pd.Series([value], index=[_NODE])


def _pathway(leadTime=10, componentLimit=None, **sourceKwargs):
    """One node, four periods of five years, a Source with a lead time of 2 periods."""
    esM = fn.EnergySystemModel(
        locations={_NODE},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW_el"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
        startYear=_YEARS[0],
        numberOfInvestmentPeriods=len(_YEARS),
        investmentPeriodInterval=5,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=componentLimit,
        componentLimitEligibility=(
            None
            if componentLimit is None
            else pd.DataFrame(index=[_NODE], columns=["lim"], data=1)
        ),
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="src",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=_perYear({year: 1 for year in _YEARS}, _timeSeries),
            investPerCapacity=1.0,
            economicLifetime=30,
            technicalLifetime=30,
            leadTime=leadTime,
            componentLimitID=None if componentLimit is None else "lim",
            **sourceKwargs,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="backup",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=1000.0,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=_perYear({2030: 5, 2035: 5}, _timeSeries),
        )
    )
    return esM


def _rollingHorizonWindows(length):
    """Lay out rolling horizon windows of ``length`` periods over the pathway."""
    starts = range(len(_YEARS) - length + 1)
    return [
        (
            list(range(start, start + length)),
            list(range(start, start + (length if start == starts[-1] else 1))),
        )
        for start in starts
    ]


def _commissioningBound(year, value):
    return {y: _locSeries(value if y == year else 0) for y in _YEARS}


def _commissioningLimit(bound):
    return pd.DataFrame(
        index=["lim"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        data=[[1.0, bound, "commissioning", None, 2030, None]],
    )


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------


def test_windows_long_enough_for_the_lead_time_pass_silently(recwarn):
    """With windows of 3 periods, every implemented decision arrives inside its window."""
    _checkWindowsAgainstLeadTimes(_pathway(), _rollingHorizonWindows(3))
    assert not [w for w in recwarn if "after the end of the window" in str(w.message)]


def test_a_lead_time_longer_than_the_window_warns():
    """With windows of 2 periods, the 2020 decision arrives in 2030, after its window."""
    with pytest.warns(UserWarning, match=r"after the end of the window.*'src': 2"):
        _checkWindowsAgainstLeadTimes(_pathway(), _rollingHorizonWindows(2))


def test_a_decision_arriving_after_the_implemented_years_does_not_warn(recwarn):
    """The last window's decisions arrive after the pathway, as in perfect foresight."""
    _checkWindowsAgainstLeadTimes(_pathway(), [([0, 1, 2, 3], [0, 1, 2, 3])])
    assert not [w for w in recwarn if "after the end of the window" in str(w.message)]


@pytest.mark.parametrize("bound", ["commissioningMin", "commissioningFix"])
def test_a_lower_bound_no_window_can_enforce_raises(bound):
    """The bound on 2030 binds the 2020 decision, which the window 2020-2025 takes."""
    esM = _pathway(**{bound: _commissioningBound(2030, 1)})
    with pytest.raises(
        ValueError, match=f"no window can enforce them: {bound} of 'src'"
    ):
        _checkWindowsAgainstLeadTimes(esM, _rollingHorizonWindows(2))


def test_a_minimum_the_arriving_stock_meets_is_accepted():
    """Stock decided in 2015 with a lead time of 15 years brings 2 in 2030."""
    esM = _pathway(
        leadTime={2015: 15, 2020: 10, 2025: 10, 2030: 10, 2035: 10},
        stockCommissioning={2015: _locSeries(2)},
        commissioningMin=_commissioningBound(2030, 1),
    )
    with pytest.warns(UserWarning, match="after the end of the window"):
        _checkWindowsAgainstLeadTimes(esM, _rollingHorizonWindows(2))


def test_an_upper_bound_no_window_can_enforce_warns():
    esM = _pathway(commissioningMax=_commissioningBound(2030, 1))
    with pytest.warns(
        UserWarning, match="commissioningMax of 'src' at 'node1' in 2030"
    ):
        _checkWindowsAgainstLeadTimes(esM, _rollingHorizonWindows(2))


@pytest.mark.parametrize("bound", ["lower", "fixed"])
def test_a_lower_componentLimit_no_window_can_enforce_raises(bound):
    esM = _pathway(componentLimit=_commissioningLimit(bound))
    with pytest.raises(ValueError, match="componentLimit 'lim' for 'src'"):
        _checkWindowsAgainstLeadTimes(esM, _rollingHorizonWindows(2))


def test_an_upper_componentLimit_no_window_can_enforce_warns():
    esM = _pathway(componentLimit=_commissioningLimit("upper"))
    with pytest.warns(UserWarning, match="componentLimit 'lim' for 'src'"):
        _checkWindowsAgainstLeadTimes(esM, _rollingHorizonWindows(2))


# ---------------------------------------------------------------------------
# The drivers check before solving any window
# ---------------------------------------------------------------------------


def test_the_rolling_horizon_refuses_an_unenforceable_bound_before_solving():
    esM = _pathway(commissioningMin=_commissioningBound(2030, 1))
    with pytest.raises(ValueError, match="no window can enforce them"):
        rollingHorizonOptimization(
            esM,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )


def test_the_sequential_driver_refuses_an_unenforceable_bound_before_solving():
    sso = pytest.importorskip("fine.expansionModules.sequentialStochasticOptimization")

    def buildTree(*args, **kwargs):
        raise AssertionError("no window may be built")

    esM = _pathway(commissioningFix=_commissioningBound(2030, 1))
    with pytest.raises(ValueError, match="no window can enforce them"):
        sso.sequentialStochasticOptimization(
            esM,
            windowLength=2,
            stageInvestmentPeriods=[[0], [1]],
            buildTree=buildTree,
            endOfHorizon="deterministic",
        )


# ---------------------------------------------------------------------------
# The evaluation fixes the decisions themselves
# ---------------------------------------------------------------------------


def test_the_evaluation_fixes_decisions_that_arrive_after_its_years():
    """The 2030 decision of 3 arrives in 2040, after the evaluated years. Fixed by
    commissioningFix of its year instead, it would bind the 2020 decision.
    """
    sso = pytest.importorskip("fine.expansionModules.sequentialStochasticOptimization")
    implemented = {
        ("SourceSinkModel", "src", _NODE, 2020): 5.0,
        ("SourceSinkModel", "src", _NODE, 2025): 0.0,
        ("SourceSinkModel", "src", _NODE, 2030): 3.0,
    }
    evaluated = sso.evaluateImplementedPathway(
        _pathway(),
        implemented,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    commis = evaluated.pyM.commis_srcSnk
    cap = evaluated.pyM.cap_srcSnk
    assert [pyomo.value(commis[_NODE, "src", ip]) for ip in range(3)] == pytest.approx(
        [5, 0, 3]
    )
    assert [pyomo.value(cap[_NODE, "src", ip]) for ip in range(3)] == pytest.approx(
        [0, 0, 5]
    )
