"""Cost-distribution tests for the lead-time feature (Step 7: CAPEX wiring;
Step 8: OPEX shift; Step 9.5: setOptimalValues/getOptimizationSummary() reporting
fix).

Mirrors the style of test_shadowCostOutput.py / test_netPresentValue.py: small,
deterministic one-node models (commissioning forced via commissioningFix so the
solver can't route around the exact scenario being tested) are solved end to end
and the actual solved objective/cost breakdown is inspected -- not just the Step 6
primitives in isolation.

Most tests below read cost results directly through
ComponentModel.getEconomicsDesign(getOptValue=True, ...) with the same
widened/shifted arguments getObjectiveFunctionContribution uses, so they inspect
exactly what the solved objective actually booked, independent of the separate
reporting path. The dedicated "optimization summary" section near the end of this
file cross-checks esM.getOptimizationSummary() (Component.setOptimalValues())
against that same ground truth -- before Step 9.5, that reporting path used the
old, un-widened/un-shifted attribute names and would have disagreed; Step 9.5
wires it to the same widened/shifted arguments, so it's now consistent.
"""

import pandas as pd
import pytest

import fine as fn
from fine.utils import ImplementedSolvers

_LOC = "node1"


def _ip_year(ip, interval, start_year=2020):
    return start_year + ip * interval


def _commissioning_series(commissioning_by_ip, n_ips, interval, loc=_LOC):
    return {
        _ip_year(ip, interval): pd.Series([commissioning_by_ip.get(ip, 0)], index=[loc])
        for ip in range(n_ips)
    }


def _zero_demand_series(n_ips, interval, loc=_LOC):
    return {
        _ip_year(ip, interval): pd.DataFrame([0], index=[0], columns=[loc])
        for ip in range(n_ips)
    }


def _op_max_series(n_ips, interval, loc=_LOC):
    return {
        _ip_year(ip, interval): pd.DataFrame([1], index=[0], columns=[loc])
        for ip in range(n_ips)
    }


def _build_leadtime_cost_model(
    *,
    n_ips,
    interval,
    economic_lifetime,
    technical_lifetime,
    invest_per_capacity,
    interest_rate,
    commissioning_by_ip,
    lead_time=None,
    opex_per_capacity=0,
    annuity_perpetuity=False,
):
    esM = fn.EnergySystemModel(
        locations={_LOC},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "MW_el"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
        startYear=2020,
        numberOfInvestmentPeriods=n_ips,
        investmentPeriodInterval=interval,
        costUnit="Euro",
        lengthUnit="km",
        annuityPerpetuity=annuity_perpetuity,
        verboseLogLevel=0,
    )
    sourceKwargs = dict(
        esM=esM,
        name="src",
        commodity="electricity",
        hasCapacityVariable=True,
        operationRateMax=_op_max_series(n_ips, interval),
        commissioningFix=_commissioning_series(commissioning_by_ip, n_ips, interval),
        investPerCapacity=invest_per_capacity,
        opexPerCapacity=opex_per_capacity,
        economicLifetime=economic_lifetime,
        technicalLifetime=technical_lifetime,
        interestRate=interest_rate,
    )
    if lead_time is not None:
        sourceKwargs["leadTime"] = lead_time
    esM.add(fn.Source(**sourceKwargs))
    esM.add(
        fn.Sink(
            esM=esM,
            name="sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=_zero_demand_series(n_ips, interval),
        )
    )
    esM.optimize(
        timeSeriesAggregation=False, solver=ImplementedSolvers.STANDARD_SOLVER.value
    )
    return esM


def _capex_by_ip(esM, costType, lifetimeAttr, divisorName, compName="src"):
    mdl = esM.componentModelingDict["SourceSinkModel"]
    results = mdl.getEconomicsDesign(
        esM.pyM,
        esM,
        factorNames=["processedInvestPerCapacity", "QPcostDev"],
        QPfactorNames=["processedQPcostScale", "processedInvestPerCapacity"],
        lifetimeAttr=lifetimeAttr,
        varName="commis",
        divisorName=divisorName,
        QPdivisorNames=["QPbound", divisorName],
        getOptValue=True,
        getOptValueCostType=costType,
    )
    return {
        ip: (df.loc[compName, _LOC] if compName in df.index else 0.0)
        for ip, df in results.items()
    }


def _widened_capex_tac_by_ip(esM, compName="src"):
    return _capex_by_ip(
        esM, "TAC", "ipLeadTimeEconomicLifetime", "CCFLeadTime", compName
    )


def _widened_capex_npv_sum(esM, compName="src"):
    return sum(
        _capex_by_ip(esM, "NPV", "ipLeadTimeEconomicLifetime", "CCFLeadTime", compName)
        .values()
    )


def _baseline_capex_npv_sum(esM, compName="src"):
    """The pre-Step-7 (unwidened) formula, called directly with the old attribute
    names -- used only to prove leadTime=0 reproduces the exact old numbers."""
    return sum(
        _capex_by_ip(esM, "NPV", "ipEconomicLifetime", "CCF", compName).values()
    )


def _opex_tac_by_ip(esM, compName="src", shiftByLeadTime=True):
    """shiftByLeadTime=True (the default) matches what
    getObjectiveFunctionContribution actually passes for opexCap/opexDec since
    Step 8; pass False only to compute the pre-Step-8 (unshifted) reference."""
    mdl = esM.componentModelingDict["SourceSinkModel"]
    results = mdl.getEconomicsDesign(
        esM.pyM,
        esM,
        factorNames=["processedOpexPerCapacity", "QPcostDev"],
        QPfactorNames=["processedQPcostScale", "processedOpexPerCapacity"],
        lifetimeAttr="ipTechnicalLifetime",
        varName="commis",
        QPdivisorNames=["QPbound"],
        getOptValue=True,
        getOptValueCostType="TAC",
        shiftByLeadTime=shiftByLeadTime,
    )
    return {
        ip: (df.loc[compName, _LOC] if compName in df.index else 0.0)
        for ip, df in results.items()
    }


# interval == economicLifetime == leadTime == 5 years -> ipEconomicLifetime == 1.0
# and ipLeadTimeEconomicLifetime == 2.0 exactly (no partial-interval remainder),
# keeping expectations simple: baseline books 1 full interval, widened books 2.
_INTERVAL = 5
_ECON_LIFETIME = 5
_TECH_LIFETIME = 5
_LEAD_TIME = 5
_INVEST_PER_CAPACITY = 100
_INTEREST_RATE = 0.05
_CAPACITY = 10


def test_leadtime_zero_reproduces_baseline_capex_exactly():
    """leadTime=0 (explicit or omitted) must reproduce the pre-Step-7 (unwidened)
    capex NPV exactly -- the top-priority regression guard."""
    esM_omitted = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
    )
    esM_zero = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        lead_time=0,
    )

    assert esM_omitted.pyM.Obj() == pytest.approx(esM_zero.pyM.Obj())

    # the actual solved widened-formula NPV must match the old formula's NPV,
    # called directly with the pre-Step-7 attribute names, byte for byte.
    for esM in (esM_omitted, esM_zero):
        assert _widened_capex_npv_sum(esM) == pytest.approx(
            _baseline_capex_npv_sum(esM)
        )


def test_leadtime_widens_capex_into_smaller_equal_shares():
    """A leadTime spanning exactly one extra interval must spread the same total
    capex over one more, correspondingly smaller, equal per-period share."""
    baseline = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        lead_time=0,
    )
    widened = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        lead_time=_LEAD_TIME,
    )

    baseline_tac = _widened_capex_tac_by_ip(baseline)
    widened_tac = _widened_capex_tac_by_ip(widened)

    # baseline: entire cost booked in the single interval covering the economic
    # lifetime (ip=0 only)
    assert baseline_tac[0] > 0
    assert baseline_tac[1] == pytest.approx(0)
    assert baseline_tac[2] == pytest.approx(0)

    # widened: same total spread equally over 2 intervals (ip=0, ip=1), each
    # strictly smaller than the baseline's single-interval share
    assert widened_tac[0] == pytest.approx(widened_tac[1])
    assert widened_tac[0] < baseline_tac[0]
    assert widened_tac[2] == pytest.approx(0)


def test_leadtime_capex_npv_conserves_total_investment():
    """Total discounted capex for one commissioning decision must be unaffected by
    how many periods it's chopped into, as long as the widened window still fits
    inside the model horizon (no truncation)."""
    baseline = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        lead_time=0,
    )
    widened = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        lead_time=_LEAD_TIME,
    )

    assert _widened_capex_npv_sum(widened) == pytest.approx(
        _widened_capex_npv_sum(baseline)
    )


def test_leadtime_greater_than_economic_lifetime_solves_without_error():
    esM = _build_leadtime_cost_model(
        n_ips=6,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        lead_time=20,  # > economicLifetime
    )
    assert esM.pyM.Obj() is not None
    assert esM.pyM.Obj() > 0


def test_leadtime_horizon_boundary_truncates_without_double_counting_or_erroring():
    """When the widened window extends past the model horizon, the tail must be
    dropped (matching the existing implicit truncation behavior), not
    double-counted and not erroring."""
    truncated = _build_leadtime_cost_model(
        n_ips=2,  # only ip=0, ip=1 exist
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={1: _CAPACITY},  # widened window needs ip=1 AND ip=2
        lead_time=_LEAD_TIME,
    )
    untruncated_reference = _build_leadtime_cost_model(
        n_ips=3,  # ip=0, 1, 2 all exist -- same commissioning decision, no truncation
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={1: _CAPACITY},
        lead_time=_LEAD_TIME,
    )

    truncated_npv = _widened_capex_npv_sum(truncated)
    untruncated_npv = _widened_capex_npv_sum(untruncated_reference)

    assert truncated_npv >= 0
    assert truncated_npv < untruncated_npv


def test_leadtime_with_annuity_perpetuity_solves_without_error():
    esM = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={2: _CAPACITY},  # last investment period
        lead_time=_LEAD_TIME,
        annuity_perpetuity=True,
    )
    assert esM.pyM.Obj() is not None
    assert esM.pyM.Obj() > 0


def test_leadtime_opex_shifts_to_availability_not_decision_year():
    """Superseded by Step 8: this test was originally
    test_leadtime_does_not_affect_opex (Step 7), asserting OPEX was
    byte-identical regardless of leadTime -- true at the time, since Step 7 only
    wired CAPEX. Step 8 deliberately makes OPEX shift (decision #3: fixed O&M
    only starts once the asset is physically available), so that assertion is no
    longer correct by design; renamed and rewritten to assert the new, correct
    behavior, kept as a regression marker for the shift itself."""
    zero = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        opex_per_capacity=5,
        lead_time=0,
    )
    widened = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        opex_per_capacity=5,
        lead_time=_LEAD_TIME,
    )

    opex_zero = _opex_tac_by_ip(zero)
    opex_widened = _opex_tac_by_ip(widened)

    # leadTime=0: today's opex numbers exactly (top-priority regression guard) --
    # booked starting at the decision year (ip=0), matches the unshifted formula.
    assert opex_zero == _opex_tac_by_ip(zero, shiftByLeadTime=False)
    assert opex_zero[0] > 0
    assert opex_zero[1] == pytest.approx(0)
    assert opex_zero[2] == pytest.approx(0)

    # leadTime=5 (1 interval): zero opex before availability (ip=0), the full
    # (unwidened, technicalLifetime-length) per-period share starting exactly at
    # the availability ip (ip=1, cross-checked against Step 4/5's capacity-shift
    # convention: commisYear + roundedIpLeadTime).
    assert opex_widened[0] == pytest.approx(0)
    assert opex_widened[1] == pytest.approx(opex_zero[0])
    assert opex_widened[2] == pytest.approx(0)


def test_leadtime_opex_horizon_boundary_shifted_past_horizon_is_zero_no_crash():
    """When the shifted opex window starts beyond the model horizon, opex must be
    zero everywhere within the horizon -- not erroring, not booked anywhere."""
    esM = _build_leadtime_cost_model(
        n_ips=2,  # only ip=0, ip=1 exist
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={1: _CAPACITY},  # availability ip = 1 + 2 = 3, out of range
        opex_per_capacity=5,
        lead_time=10,  # 2 intervals
    )
    opex = _opex_tac_by_ip(esM)
    assert opex[0] == pytest.approx(0)
    assert opex[1] == pytest.approx(0)


def test_leadtime_opex_shift_varies_by_commissioning_ip():
    """Per-investment-period-varying leadTime: the opex shift amount must track
    each commissioning decision's own lead time, not a single global shift."""
    # decisionIp=1 -> availabilityIp=2; decisionIp=2 -> availabilityIp=3 (kept
    # nonzero too, otherwise it would collide with decisionIp=1's availabilityIp=2)
    varying_lead_time = {
        _ip_year(0, _INTERVAL): 0,
        _ip_year(1, _INTERVAL): _LEAD_TIME,
        _ip_year(2, _INTERVAL): _LEAD_TIME,
    }

    commission_at_ip0 = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},  # leadTime=0 at this decision ip
        opex_per_capacity=5,
        lead_time=varying_lead_time,
    )
    commission_at_ip1 = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={1: _CAPACITY},  # leadTime=5 (1 interval) at this ip
        opex_per_capacity=5,
        lead_time=varying_lead_time,
    )

    opex_ip0 = _opex_tac_by_ip(commission_at_ip0)
    opex_ip1 = _opex_tac_by_ip(commission_at_ip1)

    # commissioning at ip=0 with leadTime=0 there -> opex starts immediately at ip=0
    assert opex_ip0[0] > 0
    assert opex_ip0[1] == pytest.approx(0)
    assert opex_ip0[2] == pytest.approx(0)

    # commissioning at ip=1 with leadTime=5 (1 interval) there -> opex shifts to ip=2
    assert opex_ip1[0] == pytest.approx(0)
    assert opex_ip1[1] == pytest.approx(0)
    assert opex_ip1[2] > 0


def test_leadtime_reproduces_notebook_example_costs_scenario_fix():
    """Direct reproduction of examples/12_LeadTimes/12_leadTimes_example_costs.ipynb's
    exact scenario (leadTime=1, economicLifetime=1, technicalLifetime=1,
    investPerCapacity=10, opexPerCapacity=20, interestRate=0.10, 2 investment
    periods of 1 year each, commissioning forced in the first IP), which
    originally demonstrated (in the notebook's own words) a "Failure Expectation":
    "still: operational fixed costs already in first IP" -- fixed O&M was booked
    at the decision year even though the asset wasn't physically available until
    the second IP. Confirms Step 8 resolves this at the objective level (reading
    directly via getEconomicsDesign, independent of the separate
    esM.getOptimizationSummary() reporting path fixed in Step 9.5 -- see
    test_optimization_summary_reproduces_notebook_example_costs_scenario_fix
    below for the same scenario read through that path)."""
    esM = _build_leadtime_cost_model(
        n_ips=2,
        interval=1,
        economic_lifetime=1,
        technical_lifetime=1,
        invest_per_capacity=10,
        interest_rate=0.10,
        commissioning_by_ip={0: 1},
        opex_per_capacity=20,
        lead_time=1,
    )

    opex = _opex_tac_by_ip(esM)
    # the originally-documented bug: fixed opex must NOT be booked in the first
    # (construction/decision) IP...
    assert opex[0] == pytest.approx(0)
    # ...it must be booked once the asset is physically available, in the second IP.
    assert opex[1] > 0

    # capex, by contrast, is correctly booked starting at the decision IP
    # (decision #2), now widened (leadTime + economicLifetime = 2 intervals)
    # rather than dumped entirely into the first IP as the pre-fix WIP did.
    capex = _widened_capex_tac_by_ip(esM)
    assert capex[0] > 0
    assert capex[1] == pytest.approx(capex[0])


# ---------------------------------------------------------------------------
# Step 9.5: esM.getOptimizationSummary() (Component.setOptimalValues()) reporting
# fix. Prior to this step, setOptimalValues called getEconomicsDesign with the
# old, un-widened/un-shifted attribute names regardless of leadTime, so the
# reported summary could disagree with what the solved objective actually
# booked. These tests read the summary through the actual public API
# (esM.getOptimizationSummary()), not the getEconomicsDesign bypass the tests
# above use, and cross-check it against that same ground truth.
# ---------------------------------------------------------------------------


def _optimization_summary_value(esM, ip, prop, compName="src", loc=_LOC):
    """ip here is the investment-period *index* (0, 1, 2, ...), matching every
    other helper in this file; converted to the calendar year
    esM.getOptimizationSummary() itself requires."""
    df = esM.getOptimizationSummary(
        "SourceSinkModel", ip=esM.investmentPeriodNames[ip], outputLevel=0
    )
    row = df.xs((compName, prop), level=("Component", "Property"))
    val = row[loc].iloc[0]
    return 0.0 if pd.isna(val) else float(val)


def _summary_capex_by_ip(esM, compName="src"):
    return {
        ip: _optimization_summary_value(esM, ip, "capexCap", compName)
        for ip in esM.investmentPeriods
    }


def _summary_opex_by_ip(esM, compName="src"):
    return {
        ip: _optimization_summary_value(esM, ip, "opexCap", compName)
        for ip in esM.investmentPeriods
    }


def _summary_npv_contribution_sum(esM, compName="src"):
    return sum(
        _optimization_summary_value(esM, ip, "NPVcontribution", compName)
        for ip in esM.investmentPeriods
    )


def test_optimization_summary_leadtime_zero_matches_baseline():
    """leadTime=0: esM.getOptimizationSummary()'s capexCap/opexCap must match the
    ground truth (and therefore today's pre-fix numbers too, since both are
    identical at leadTime=0) -- the top-priority regression guard for this step."""
    esM = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        opex_per_capacity=5,
        lead_time=0,
    )

    summary_capex = _summary_capex_by_ip(esM)
    summary_opex = _summary_opex_by_ip(esM)
    ground_truth_capex = _widened_capex_tac_by_ip(esM)
    ground_truth_opex = _opex_tac_by_ip(esM)

    for ip in esM.investmentPeriods:
        assert summary_capex[ip] == pytest.approx(ground_truth_capex[ip])
        assert summary_opex[ip] == pytest.approx(ground_truth_opex[ip])


def test_optimization_summary_reflects_widened_capex_and_shifted_opex():
    """leadTime>0: esM.getOptimizationSummary()'s capexCap/opexCap must match the
    same widened/shifted ground truth the solved objective actually used --
    the core proof the reporting gap is closed."""
    esM = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        opex_per_capacity=5,
        lead_time=_LEAD_TIME,
    )

    summary_capex = _summary_capex_by_ip(esM)
    summary_opex = _summary_opex_by_ip(esM)
    ground_truth_capex = _widened_capex_tac_by_ip(esM)
    ground_truth_opex = _opex_tac_by_ip(esM)

    for ip in esM.investmentPeriods:
        assert summary_capex[ip] == pytest.approx(ground_truth_capex[ip])
        assert summary_opex[ip] == pytest.approx(ground_truth_opex[ip])

    # and, concretely, capex is split into two equal nonzero shares while opex
    # is shifted entirely to ip=1 -- not silently identical to the leadTime=0 case
    assert summary_capex[0] == pytest.approx(summary_capex[1])
    assert summary_capex[0] > 0
    assert summary_opex[0] == pytest.approx(0)
    assert summary_opex[1] > 0


def test_optimization_summary_npv_sum_equals_objective_with_leadtime():
    """Mirrors test_netPresentValue.py's own invariant (NPVcontribution sum ==
    esM.pyM.Obj()), now checked for a leadTime>0 model whose widened/shifted
    windows both fit inside the model horizon (no truncation) -- this invariant
    was flagged (Steps 7-8's completion records) as capable of breaking for
    leadTime>0 components before this reporting path was fixed."""
    esM = _build_leadtime_cost_model(
        n_ips=3,
        interval=_INTERVAL,
        economic_lifetime=_ECON_LIFETIME,
        technical_lifetime=_TECH_LIFETIME,
        invest_per_capacity=_INVEST_PER_CAPACITY,
        interest_rate=_INTEREST_RATE,
        commissioning_by_ip={0: _CAPACITY},
        opex_per_capacity=5,
        lead_time=_LEAD_TIME,
    )
    assert _summary_npv_contribution_sum(esM) == pytest.approx(esM.pyM.Obj())


def test_optimization_summary_reproduces_notebook_example_costs_scenario_fix():
    """Same exact examples/12_LeadTimes/12_leadTimes_example_costs.ipynb scenario
    as test_leadtime_reproduces_notebook_example_costs_scenario_fix, this time
    read through the actual public esM.getOptimizationSummary() API rather than
    the getEconomicsDesign bypass -- confirming a user calling
    esM.getOptimizationSummary() today would see the documented bug ("still:
    operational fixed costs already in first IP") as fixed, not just the
    internal objective."""
    esM = _build_leadtime_cost_model(
        n_ips=2,
        interval=1,
        economic_lifetime=1,
        technical_lifetime=1,
        invest_per_capacity=10,
        interest_rate=0.10,
        commissioning_by_ip={0: 1},
        opex_per_capacity=20,
        lead_time=1,
    )

    opex = _summary_opex_by_ip(esM)
    assert opex[0] == pytest.approx(0)
    assert opex[1] > 0

    capex = _summary_capex_by_ip(esM)
    assert capex[0] > 0
    assert capex[1] == pytest.approx(capex[0])
