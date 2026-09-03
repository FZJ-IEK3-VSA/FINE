"""Tests for `fine.utils.runInfeasibilityPrechecks`.

The energy system model built in `_build_infeasible_esM` is taken from the
example notebook that was used to manually validate the infeasibility
pre-checks. It combines a capacity-limited Wind source, an Electrolyzer,
a battery storage, electricity/hydrogen transmission between three regions
and a fixed hydrogen demand that spikes in the last time step.

With the capacities as given in the notebook, the model is infeasible:
the Electrolyzer's electricity demand (needed to cover the hydrogen
demand) exceeds what the Wind source can supply, both over the whole
time horizon and in the last time step. `runInfeasibilityPrechecks`
is expected to catch this before the optimization problem is even
declared, via `checkJointInputDemandAggregated` and
`checkJointInputDemandPerTimeStep`.

`_build_feasible_esM` is the same model with the Wind, AC cable and
hydrogen pipeline capacities raised, which removes the shortage. It is
used as a control case to guard against false positives.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import fine as fn
from fine.utils import (
    checkCommodityReachability,
    checkJointInputDemandAggregated,
    checkJointInputDemandPerTimeStep,
    checkTimeStepBalance,
    runInfeasibilityPrechecks,
)


def _build_esm(windCapacityMax, transmissionCapacityMax):
    """Build the example ESM from the notebook with configurable capacities."""
    locations = {"Region1", "Region2", "Region3"}
    commodityUnitDict = {
        "electricity": "GW_el",
        "hydrogen": "GW_H2",
        "heat": "GW_th",
    }
    commodities = {"electricity", "hydrogen", "heat"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=4,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    windOperationRateMax = pd.DataFrame(
        {
            "Region1": [1, 1, 1, 1],
            "Region2": [1, 1, 1, 1],
            "Region3": [1, 1, 0, 0],
        }
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=windCapacityMax,
            operationRateMax=windOperationRateMax,
            investPerCapacity=2 * 2190,
            opexPerCapacity=0,
            interestRate=0,
            opexPerOperation=0,
            economicLifetime=1,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit="GW_el",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=1,
            chargeEfficiency=0.95,
            cyclicLifetime=10000,
            dischargeEfficiency=0.95,
            selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
        )
    )

    locs = sorted(locations)
    distances = pd.DataFrame(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), index=locs, columns=locs
    )
    eligibility = pd.DataFrame(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), index=locs, columns=locs
    )

    esM.add(
        fn.Transmission(
            esM=esM,
            name="AC cable",
            commodity="electricity",
            losses=0,
            distances=distances,
            hasCapacityVariable=True,
            capacityMax=transmissionCapacityMax,
            locationalEligibility=eligibility,
            investPerCapacity=0.1,
            interestRate=0.08,
            economicLifetime=50,
        )
    )

    esM.add(
        fn.Transmission(
            esM=esM,
            name="hydrogen pipeline",
            commodity="hydrogen",
            losses=0,
            distances=distances,
            capacityMax=transmissionCapacityMax,
            hasCapacityVariable=True,
            locationalEligibility=eligibility,
            investPerCapacity=0.1,
            interestRate=0.08,
            economicLifetime=50,
        )
    )

    demand = pd.DataFrame(
        {
            "Region1": [1, 1, 1, 1],
            "Region2": [0, 0, 0, 0],
            "Region3": [0, 0, 0, 4],
        }
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    return esM


@pytest.fixture
def infeasible_esM():
    """ESM from the notebook: Wind cannot cover the Electrolyzer's demand."""
    return _build_esm(windCapacityMax=1, transmissionCapacityMax=1)


@pytest.fixture
def feasible_esM():
    """Build ESM with enough Wind/transmission capacity to remove the shortage."""
    return _build_esm(windCapacityMax=5, transmissionCapacityMax=5)


def test_runInfeasibilityPrechecks_raises_for_infeasible_esM(infeasible_esM):
    """The notebook's example ESM must be caught before the solver runs."""
    with pytest.raises(ValueError) as excinfo:
        runInfeasibilityPrechecks(infeasible_esM)

    message = str(excinfo.value)
    assert "checkJointInputDemandAggregated" in message
    assert "checkJointInputDemandPerTimeStep" in message
    assert "electricity" in message


def test_runInfeasibilityPrechecks_raiseError_false_returns_problems(
    infeasible_esM,
):
    """With raiseError=False, problems are returned instead of raised."""
    problems = runInfeasibilityPrechecks(infeasible_esM, raiseError=False)

    assert problems, "expected at least one detected problem"
    assert any("checkJointInputDemandAggregated" in p for p in problems)
    assert any("checkJointInputDemandPerTimeStep" in p for p in problems)


def test_runInfeasibilityPrechecks_passes_for_feasible_esM(feasible_esM):
    """Control case: raising the capacities removes the infeasibility.

    Guards against false positives / an overly strict check.
    """
    problems = runInfeasibilityPrechecks(feasible_esM, raiseError=False)
    assert problems == []


def test_runInfeasibilityPrechecks_single_check_reports_the_shortage(
    infeasible_esM,
):
    """The aggregated joint-input-demand check alone must already flag the model."""
    problems = checkJointInputDemandAggregated(infeasible_esM)

    assert len(problems) == 1
    assert "electricity" in problems[0]
    assert (
        "Region1" in problems[0]
        and "Region2" in problems[0]
        and "Region3" in problems[0]
    )


# ---------------------------------------------------------------------------
# checkCommodityReachability: catches a structural gap the quantity checks
# cannot see, namely a flexible conversion component whose alternative
# inputs are none of them actually available anywhere in the system.
# ---------------------------------------------------------------------------


@pytest.fixture
def unreachable_flexible_input_esM():
    """Create a boiler that can burn gas OR oil to produce heat, but neither exists.

    The quantity-based checks (`checkJointInputDemand*`, `checkTimeStepBalance`)
    deliberately do not restrict a flexible conversion component by its
    inputs, because the input demand of a flexible conversion cannot be
    assigned to a single commodity (see the comment next to `isFlexible`
    in `utils.py`). They therefore assume the boiler's heat output is
    available up to its capacity, regardless of whether gas or oil can
    actually be supplied. Only `checkCommodityReachability`, which is a
    purely structural/qualitative check, requires at least one of the two
    fuels to be reachable and catches the gap.
    """
    esM = fn.EnergySystemModel(
        locations={"Region1"},
        commodities={"heat", "gas", "oil"},
        numberOfTimeSteps=2,
        commodityUnitsDict={
            "heat": "GW_th",
            "gas": "GW_ch4",
            "oil": "GW_oil",
        },
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    # Flexible conversion: {'gas': -1, 'oil': -1.2} is a commodity GROUP,
    # meaning the boiler needs *either* gas *or* oil, not both.
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Flexible boiler",
            physicalUnit="GW_th",
            commodityConversionFactors={"heat": 1, "fuel": {"gas": -1, "oil": -1.2}},
            hasCapacityVariable=True,
            capacityFix=5,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0,
            economicLifetime=1,
        )
    )
    # Note: no Source produces gas or oil anywhere in the system.

    demand = pd.DataFrame({"Region1": [1, 1]})
    esM.add(
        fn.Sink(
            esM=esM,
            name="Heat demand",
            commodity="heat",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    return esM


def test_checkCommodityReachability_flags_unavailable_flexible_input(
    unreachable_flexible_input_esM,
):
    """CheckCommodityReachability alone must catch the missing fuel."""
    problems = checkCommodityReachability(unreachable_flexible_input_esM)

    assert len(problems) == 1
    assert "Heat demand" in problems[0]
    assert "heat" in problems[0]
    assert "Region1" in problems[0]


def test_quantity_checks_miss_the_unavailable_flexible_input(
    unreachable_flexible_input_esM,
):
    """Documents the known blind spot: the quantity checks do not see it.

    This is not a bug to fix, it is *why* `checkCommodityReachability`
    exists as a separate, structural check (see its docstring and the
    comment on `isFlexible` in `checkJointInputDemand`/
    `checkTimeStepBalance`). If this test ever starts failing because the
    quantity checks *do* start flagging the problem, the reachability
    check may have become partially redundant - worth a second look, not
    a silent fix.
    """
    esM = unreachable_flexible_input_esM
    assert checkJointInputDemandAggregated(esM) == []
    assert checkJointInputDemandPerTimeStep(esM) == []
    assert checkTimeStepBalance(esM) == []


def test_runInfeasibilityPrechecks_raises_via_reachability_for_flexible_input(
    unreachable_flexible_input_esM,
):
    """End-to-end: the wrapper still raises, driven solely by reachability."""
    with pytest.raises(ValueError) as excinfo:
        runInfeasibilityPrechecks(unreachable_flexible_input_esM)

    assert "checkCommodityReachability" in str(excinfo.value)


# ---------------------------------------------------------------------------
# checkTimeStepBalance: catches a transport bottleneck on a path across an
# intermediate location, which neither the per-location nor the per-island
# balance condition (nor the joint-input-demand checks) can see.
# ---------------------------------------------------------------------------


@pytest.fixture
def chain_bottleneck_esM():
    """A-B-C chain where A supplies both B and C, bottlenecked on A-B.

    Source capacity (10) and the total demand (3 + 3 = 6) are both fine in
    isolation: every location has enough *direct* import capacity, and the
    connected island has enough pooled supply. But *all* electricity for
    both B and C must physically pass through the A-B link, whose capacity
    (3) is too small for the combined flow of 6. Only the network-wide
    maximum-transportable-flow condition in `checkTimeStepBalance` (its
    condition (c)) detects this; there is no Conversion component, so the
    joint-input-demand checks - which only look at conversion inputs - have
    nothing to check and stay silent.
    """
    locs = ["A", "B", "C"]
    esM = fn.EnergySystemModel(
        locations=set(locs),
        commodities={"electricity"},
        numberOfTimeSteps=1,
        commodityUnitsDict={"electricity": "GW_el"},
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Plant",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=10,
            investPerCapacity=0,
            opexPerCapacity=0,
            interestRate=0,
            economicLifetime=1,
            locationalEligibility=pd.Series({"A": 1, "B": 0, "C": 0}),
        )
    )

    # Chain topology: A-B and B-C only, no direct A-C link.
    eligibility = pd.DataFrame(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), index=locs, columns=locs
    )
    capacityMax = pd.DataFrame(
        np.array([[0, 3, 0], [3, 0, 10], [0, 10, 0]]), index=locs, columns=locs
    )
    distances = pd.DataFrame(
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), index=locs, columns=locs
    )
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Line",
            commodity="electricity",
            losses=0,
            distances=distances,
            hasCapacityVariable=True,
            capacityMax=capacityMax,
            locationalEligibility=eligibility,
            investPerCapacity=0.1,
            interestRate=0.08,
            economicLifetime=50,
        )
    )

    demand = pd.DataFrame({"A": [0], "B": [3], "C": [3]})
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    return esM


def test_checkTimeStepBalance_flags_bottleneck_on_intermediate_link(
    chain_bottleneck_esM,
):
    """CheckTimeStepBalance alone must catch the A-B bottleneck."""
    problems = checkTimeStepBalance(chain_bottleneck_esM)

    assert len(problems) == 1
    assert "maximum transportable flow of 3" in problems[0]
    assert "total demand of 6" in problems[0]


def test_other_checks_miss_the_chain_bottleneck(chain_bottleneck_esM):
    """Per-location and per-island balance alone would have missed this.

    Every location has enough *direct* import capacity and the pooled
    island supply covers the pooled demand - the shortage only shows up
    once the shared A-B link is treated as a real network constraint.
    """
    esM = chain_bottleneck_esM
    assert checkCommodityReachability(esM) == []
    assert checkJointInputDemandAggregated(esM) == []
    assert checkJointInputDemandPerTimeStep(esM) == []


def test_runInfeasibilityPrechecks_raises_via_timestep_balance_for_bottleneck(
    chain_bottleneck_esM,
):
    """End-to-end: the wrapper still raises, driven solely by checkTimeStepBalance."""
    with pytest.raises(ValueError) as excinfo:
        runInfeasibilityPrechecks(chain_bottleneck_esM)

    assert "checkTimeStepBalance" in str(excinfo.value)


# ---------------------------------------------------------------------------
# runInfeasibilityPrechecks: a check that fails to run (e.g. because a
# component holds unexpected data) must be skipped with a warning, not
# crash the whole pre-check run or silently swallow the other checks.
# ---------------------------------------------------------------------------


def _brokenCheck(esM):
    raise RuntimeError("unexpected component data")


def test_runInfeasibilityPrechecks_skips_a_broken_check_with_warning(
    unreachable_flexible_input_esM,
):
    """A check that raises is skipped, but other checks still run and count."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as excinfo:
            runInfeasibilityPrechecks(
                unreachable_flexible_input_esM,
                checks=(_brokenCheck, checkCommodityReachability),
            )

    # The broken check must not crash the run, only warn ...
    assert any(
        "_brokenCheck" in str(w.message) and "skipped" in str(w.message) for w in caught
    )
    # ... and the still-working check must still be reflected in the result.
    assert "checkCommodityReachability" in str(excinfo.value)


def test_runInfeasibilityPrechecks_does_not_raise_if_only_check_is_broken():
    """If the only requested check fails to run, no problems are found.

    A broken check must never be misinterpreted as a detected problem -
    `runInfeasibilityPrechecks` must not block an otherwise valid model
    just because one check could not be evaluated.
    """
    esM = _build_esm(windCapacityMax=5, transmissionCapacityMax=5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        problems = runInfeasibilityPrechecks(
            esM, checks=(_brokenCheck,), raiseError=False
        )

    assert problems == []
    assert any("_brokenCheck" in str(w.message) for w in caught)
