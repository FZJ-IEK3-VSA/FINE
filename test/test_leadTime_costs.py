"""Cost-distribution tests for the lead-time feature (Step 7: CAPEX wiring).

Mirrors the style of test_shadowCostOutput.py / test_netPresentValue.py: small,
deterministic one-node models (commissioning forced via commissioningFix so the
solver can't route around the exact scenario being tested) are solved end to end
and the actual solved objective/cost breakdown is inspected -- not just the Step 6
primitives in isolation.

Note: esM.getOptimizationSummary()'s capexCap/TAC/NPVcontribution columns are
produced by Component.setOptimalValues(), which Step 7 deliberately does not
touch (out of the plan's stated file scope) and therefore still uses the
un-widened ipEconomicLifetime/CCF attributes. Reading capex results here goes
directly through ComponentModel.getEconomicsDesign(getOptValue=True, ...) with
the same widened attributes getObjectiveFunctionContribution now uses, so these
tests inspect exactly what the solved objective actually booked. See CLAUDE.md's
"Deferred / follow-up items" for this reporting-path gap.
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


def _opex_tac_by_ip(esM, compName="src"):
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


def test_leadtime_does_not_affect_opex():
    """Step 7 only widens CAPEX; OPEX (Step 8's job) must be byte-identical
    regardless of leadTime."""
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
    for ip in zero.investmentPeriods:
        assert opex_widened[ip] == pytest.approx(opex_zero[ip])
