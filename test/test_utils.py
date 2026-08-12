import pandas as pd
from fine import utils
import fine as fn
import numpy as np
import pytest

from fine.utils import ImplementedSolvers


def test_checkSimultaneousChargeDischarge():
    """Test a minimal example, with two regions and 10 days, where simultaneous charge and discharge occurs."""
    locations = {"Region1", "Region2"}
    commodityUnitDict = {"electricity": r"MW$_{el}$"}
    commodities = {"electricity"}
    ndays = 10
    nhours = 24 * ndays
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
    )
    # Create synthetic daily demand profile
    dailyProfileSimple = [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.7,
        0.9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.9,
        0.8,
    ]
    demand = pd.DataFrame(
        [[u * 40, u * 60] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    # Add storage 'Batteries'
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 1000, 0
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000
    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=chargeEfficiency,
            cyclicLifetime=cyclicLifetime,
            dischargeEfficiency=dischargeEfficiency,
            selfDischarge=selfDischarge,
            chargeRate=chargeRate,
            dischargeRate=dischargeRate,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )
    # Create synthetic profile for PV and add PV with fixed operationRate. Therefore, it cannot be curtailed.
    # To achieve a curtailment, the system 'burns' energy by charging and discharging the storage simultaneously.
    dailyProfileSimple = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.05,
        0.15,
        0.2,
        0.4,
        0.8,
        0.7,
        0.4,
        0.2,
        0.15,
        0.05,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    operationRateFix = pd.DataFrame(
        [[u, u] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    )
    capacityMax = pd.Series([10000, 10000], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 100, 10
    interestRate, economicLifetime = 0.08, 25
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=operationRateFix,
            capacityFix=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    with pytest.warns(UserWarning, match="Charge and discharge at the same time"):
        esM.optimize(
            timeSeriesAggregation=False,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )
    # Get the charge and discharge time series of the Batteries and use the check in the utils.
    tsCharge = esM.componentModelingDict[
        "StorageModel"
    ].chargeOperationVariablesOptimum.loc["Batteries"]
    tsDischarge = esM.componentModelingDict[
        "StorageModel"
    ].dischargeOperationVariablesOptimum.loc["Batteries"]
    simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
        tsCharge, tsDischarge
    )

    assert simultaneousChargeDischarge, (
        "Check for simultaneous charge & discharge should have returned True"
    )


def test_functionality_checkSimultaneousChargeDischarge():
    """Simple functionality test for utils.checkSimultaneousChargeDischarge."""
    # Define charge and discharge time series for one region
    tsCharge = pd.DataFrame(columns=["Region1"])
    tsCharge["Region1"] = 3 * [1] + 1 * [0]
    tsDischarge = pd.DataFrame(columns=["Region1"])
    tsDischarge["Region1"] = 2 * [0] + 2 * [1]
    simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
        tsCharge, tsDischarge
    )

    assert simultaneousChargeDischarge, (
        "Check for simultaneous charge & discharge should have returned True"
    )


def test_check_and_set_cost_parameter():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # Test with valid integer data (1dim)
    assert utils.checkAndSetCostParameter(esM, "testParam", 10, "1dim", None).equals(
        pd.Series([10.0], index=esM.locations)
    )

    # Test with valid series data (1dim)
    valid_series_1dim = pd.Series([10], index=esM.locations)
    assert utils.checkAndSetCostParameter(
        esM, "testParam", valid_series_1dim, "1dim", None
    ).equals(valid_series_1dim.astype(float))

    # Test with NaN in integer data (1dim)
    with pytest.raises(ValueError):
        assert utils.checkAndSetCostParameter(
            esM, "testParam", np.nan, "1dim", None
        ).equals(pd.Series([np.nan], index=esM.locations))

    # Test with NaN in series data (2dim)
    with pytest.raises(ValueError):
        invalid_series_with_nan = pd.Series([10, np.nan], index=["loc1", "loc2"])
        assert utils.checkAndSetCostParameter(
            esM, "testParam", invalid_series_with_nan, "2dim", None
        ).equals(invalid_series_with_nan, index=esM.locations)


# --- Lead-time-widened CCF / interval-apportionment primitives (Step 6) ---
# These primitives (Component.ipLeadTimeEconomicLifetime, Component.CCFLeadTime,
# utils.getParametersForUnevenLifetimes(..., "ipLeadTimeEconomicLifetime", esM, ip))
# are purely additive building blocks for decision #4 (CAPEX total-cost conservation
# via a widened Capital Charge Factor) -- not yet wired into the objective function
# (that is Step 7). Tested standalone here, independent of Pyomo/optimize().

_LEADTIME_CCF_LOC = "loc1"
_LEADTIME_CCF_INTERVAL = 5


def _build_leadtime_ccf_esM(leadTime, economicLifetime, interestRate):
    esM = fn.EnergySystemModel(
        locations={_LEADTIME_CCF_LOC},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"GW$_{el}$"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        lengthUnit="km",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=_LEADTIME_CCF_INTERVAL,
        startYear=2020,
        verboseLogLevel=2,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="src",
            commodity="electricity",
            hasCapacityVariable=True,
            leadTime=leadTime,
            economicLifetime=economicLifetime,
            # kept far above any (leadTime, economicLifetime) combination used in
            # these tests so the technical-lifetime/economic-lifetime "same
            # interval" interaction in getParametersForUnevenLifetimes never
            # triggers by coincidence -- these tests target the widened
            # economic-lifetime primitive in isolation.
            technicalLifetime=100,
            interestRate=interestRate,
            investPerCapacity=1000,
        )
    )
    return esM, esM.getComponent("src")


def test_ccf_leadtime_zero_matches_ccf_exactly():
    esM, comp = _build_leadtime_ccf_esM(leadTime=0, economicLifetime=8, interestRate=0.05)
    for ip in esM.investmentPeriods:
        assert comp.CCFLeadTime[ip][_LEADTIME_CCF_LOC] == pytest.approx(
            comp.CCF[ip][_LEADTIME_CCF_LOC]
        )


def test_ip_leadtime_economic_lifetime_zero_leadtime_matches_economic_lifetime():
    esM, comp = _build_leadtime_ccf_esM(leadTime=0, economicLifetime=8, interestRate=0.05)
    for ip in esM.investmentPeriods:
        assert comp.ipLeadTimeEconomicLifetime[ip][_LEADTIME_CCF_LOC] == pytest.approx(
            comp.ipEconomicLifetime[_LEADTIME_CCF_LOC]
        )


@pytest.mark.parametrize(
    "leadTime, economicLifetime, interestRate",
    [
        (0, 8, 0.05),
        (2, 8, 0.05),
        (3, 8, 0.05),
        (5, 8, 0.05),
        (10, 8, 0.05),
        (12, 8, 0.02),
        (20, 8, 0.05),  # leadTime > economicLifetime
        (5, 8, 0.0),  # zero interest rate
    ],
)
def test_ccf_leadtime_conserves_total_cost(leadTime, economicLifetime, interestRate):
    """The widened CCF and the widened interval-apportionment primitive must be
    mutually consistent: an annuity sized off CCFLeadTime, apportioned into full
    and partial investment-period chunks per getParametersForUnevenLifetimes(...,
    "ipLeadTimeEconomicLifetime", ...) and discounted chunk-by-chunk back to the
    commissioning date, must reconstruct investPerCapacity exactly (to floating
    point tolerance) -- this is the total-cost-conservation property decision #4
    relies on.
    """
    investPerCapacity = 1000.0
    esM, comp = _build_leadtime_ccf_esM(leadTime, economicLifetime, interestRate)
    loc = _LEADTIME_CCF_LOC
    interval = _LEADTIME_CCF_INTERVAL
    r = comp.interestRate[loc]

    for ip in esM.investmentPeriods:
        ccfLeadTime = comp.CCFLeadTime[ip][loc]
        annuity = investPerCapacity / ccfLeadTime

        fullCostIntervals, hasPartial, _ = utils.getParametersForUnevenLifetimes(
            comp.name, loc, "ipLeadTimeEconomicLifetime", esM, ip
        )

        total = 0.0
        for k in range(fullCostIntervals):
            chunk = annuity * utils.annuityPresentValueFactor(
                esM, comp.name, loc, interval
            )
            total += chunk / (1 + r) ** (k * interval)

        if hasPartial:
            widened = comp.ipLeadTimeEconomicLifetime[ip][loc]
            partialYears = (widened % 1) * interval
            chunk = annuity * utils.annuityPresentValueFactor(
                esM, comp.name, loc, partialYears
            )
            total += chunk / (1 + r) ** (fullCostIntervals * interval)

        assert total == pytest.approx(investPerCapacity, rel=1e-9)


def test_get_parameters_for_uneven_lifetimes_leadtime_zero_matches_baseline():
    esM, comp = _build_leadtime_ccf_esM(leadTime=0, economicLifetime=8, interestRate=0.05)
    loc = _LEADTIME_CCF_LOC
    baseline = utils.getParametersForUnevenLifetimes(
        comp.name, loc, "ipEconomicLifetime", esM
    )
    for ip in esM.investmentPeriods:
        widened = utils.getParametersForUnevenLifetimes(
            comp.name, loc, "ipLeadTimeEconomicLifetime", esM, ip
        )
        assert widened == baseline


def test_get_parameters_for_uneven_lifetimes_leadtime_spans_one_extra_interval():
    # interval=5, economicLifetime=8 -> ipEconomicLifetime=1.6 -> 1 full interval, partial 0.6
    # leadTime=5 -> ipLeadTime=1.0 -> widened=2.6 -> 2 full intervals, partial 0.6 (unchanged)
    esM, comp = _build_leadtime_ccf_esM(leadTime=5, economicLifetime=8, interestRate=0.05)
    loc = _LEADTIME_CCF_LOC
    for ip in esM.investmentPeriods:
        fullCostIntervals, hasPartial, _ = utils.getParametersForUnevenLifetimes(
            comp.name, loc, "ipLeadTimeEconomicLifetime", esM, ip
        )
        assert fullCostIntervals == 2
        assert hasPartial is True
        assert comp.ipLeadTimeEconomicLifetime[ip][loc] == pytest.approx(2.6)


def test_get_parameters_for_uneven_lifetimes_leadtime_spans_multiple_extra_intervals():
    # interval=5, economicLifetime=8 -> ipEconomicLifetime=1.6
    # leadTime=10 -> ipLeadTime=2.0 -> widened=3.6 -> 3 full intervals, partial 0.6
    esM, comp = _build_leadtime_ccf_esM(leadTime=10, economicLifetime=8, interestRate=0.05)
    loc = _LEADTIME_CCF_LOC
    for ip in esM.investmentPeriods:
        fullCostIntervals, hasPartial, _ = utils.getParametersForUnevenLifetimes(
            comp.name, loc, "ipLeadTimeEconomicLifetime", esM, ip
        )
        assert fullCostIntervals == 3
        assert hasPartial is True
        assert comp.ipLeadTimeEconomicLifetime[ip][loc] == pytest.approx(3.6)


def test_get_parameters_for_uneven_lifetimes_leadtime_fractional_remainder():
    # interval=5, economicLifetime=8 -> ipEconomicLifetime=1.6
    # leadTime=3 -> ipLeadTime=0.6 -> widened=2.2 -> 2 full intervals, partial 0.2
    # (a different fractional remainder than the baseline's 0.6, confirming the
    # partial-interval credit tracks the widened value, not the un-widened one)
    esM, comp = _build_leadtime_ccf_esM(leadTime=3, economicLifetime=8, interestRate=0.05)
    loc = _LEADTIME_CCF_LOC
    for ip in esM.investmentPeriods:
        fullCostIntervals, hasPartial, _ = utils.getParametersForUnevenLifetimes(
            comp.name, loc, "ipLeadTimeEconomicLifetime", esM, ip
        )
        assert fullCostIntervals == 2
        assert hasPartial is True
        widened = comp.ipLeadTimeEconomicLifetime[ip][loc]
        assert widened == pytest.approx(2.2)
        assert (widened % 1) == pytest.approx(0.2)


def test_get_parameters_for_uneven_lifetimes_leadtime_greater_than_economic_lifetime():
    # leadTime=20 > economicLifetime=8 -- no special-casing needed, the widened-window
    # math handles it the same as any other duration.
    esM, comp = _build_leadtime_ccf_esM(leadTime=20, economicLifetime=8, interestRate=0.05)
    loc = _LEADTIME_CCF_LOC
    for ip in esM.investmentPeriods:
        fullCostIntervals, hasPartial, _ = utils.getParametersForUnevenLifetimes(
            comp.name, loc, "ipLeadTimeEconomicLifetime", esM, ip
        )
        widened = comp.ipLeadTimeEconomicLifetime[ip][loc]
        assert widened == pytest.approx(5.6)
        assert fullCostIntervals == 5
        assert hasPartial is True


def test_get_parameters_for_uneven_lifetimes_requires_ip_for_widened_lifetime():
    esM, comp = _build_leadtime_ccf_esM(leadTime=5, economicLifetime=8, interestRate=0.05)
    with pytest.raises(ValueError):
        utils.getParametersForUnevenLifetimes(
            comp.name, _LEADTIME_CCF_LOC, "ipLeadTimeEconomicLifetime", esM
        )
