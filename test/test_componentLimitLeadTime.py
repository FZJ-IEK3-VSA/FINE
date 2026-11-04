"""A componentLimit with type="commissioning" bounds the capacity that becomes available.

With a lead time, a decision counts in the investment period it becomes available in,
not in the one it is taken in, and arriving stock counts as well. These tests need both
the componentLimit and the leadTime feature.
"""

import pandas as pd
import pytest
import pyomo.environ as pyomo
from pyomo.core.expr.visitor import identify_variables

import fine as fn
from fine.utils import ImplementedSolvers

_LIMIT_COLUMNS = ["value", "bound", "type", "commodity", "ip", "ipEnd"]


def _commissioningLimit(value, ip, ipEnd=None, bound="upper"):
    return pd.DataFrame(
        index=["lim"],
        columns=_LIMIT_COLUMNS,
        data=[[value, bound, "commissioning", None, ip, ipEnd]],
    )


# ---------------------------------------------------------------------------
# Which commissioning variable enters the limit, for every component class
# ---------------------------------------------------------------------------

_REGIONS = ["R1", "R2"]


def _twoRegionEsM(componentLimitEligibility2dim=None):
    regionEligibility = pd.DataFrame(index=_REGIONS, columns=["lim"], data=1)
    if componentLimitEligibility2dim is not None:
        regionEligibility = regionEligibility.drop(columns="lim")
    return fn.EnergySystemModel(
        locations=set(_REGIONS),
        commodities={"electricity", "gas"},
        commodityUnitsDict={"electricity": "GW_el", "gas": "GW_gas"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
        startYear=2020,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=_commissioningLimit(100.0, 2025),
        componentLimitEligibility=regionEligibility,
        componentLimitEligibility2dim=componentLimitEligibility2dim,
    )


def _addLimitedComponent(esM, componentClass, leadTime):
    common = dict(
        esM=esM,
        name="comp",
        hasCapacityVariable=True,
        investPerCapacity=1.0,
        leadTime=leadTime,
        componentLimitID="lim",
    )
    if componentClass == "Source":
        esM.add(fn.Source(commodity="electricity", **common))
    elif componentClass == "Conversion":
        esM.add(
            fn.Conversion(
                physicalUnit="GW_el",
                commodityConversionFactors={"electricity": 1, "gas": -1},
                **common,
            )
        )
    elif componentClass == "Storage":
        esM.add(fn.Storage(commodity="electricity", **common))
    elif componentClass == "Transmission":
        esM.add(
            fn.Transmission(
                commodity="electricity",
                locationalEligibility=pd.DataFrame(
                    [[0, 1], [1, 0]], index=_REGIONS, columns=_REGIONS
                ),
                **common,
            )
        )


def _limitVariables(esM):
    """Name and index of every variable in the componentLimit constraints."""
    constraint = esM.pyM.yearlyComponentLimitConstraint
    return {
        (var.parent_component().local_name, var.index())
        for key in constraint
        for var in identify_variables(constraint[key].body)
    }


@pytest.mark.parametrize(
    "componentClass", ["Source", "Conversion", "Storage", "Transmission"]
)
@pytest.mark.parametrize("leadTime, decisionIp", [(0, 1), (5, 0)])
def test_commissioning_limit_takes_the_decision_that_arrives(
    componentClass, leadTime, decisionIp
):
    """A limit on 2025 takes the 2025 decision without a lead time and the 2020
    decision with a lead time of one period, in every component class.
    """
    esM = _twoRegionEsM()
    _addLimitedComponent(esM, componentClass, leadTime)
    esM.declareOptimizationProblem(timeSeriesAggregation=False)

    variables = _limitVariables(esM)
    assert variables, "the limit holds no variable"
    assert {name.split("_")[0] for name, _ in variables} == {"commis"}
    assert {index[-1] for _, index in variables} == {decisionIp}


def test_commissioning_limit_over_connections_takes_the_decision_that_arrives():
    """The same holds for a limit over transmission connections."""
    esM = _twoRegionEsM(
        componentLimitEligibility2dim=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("R1", "R2")]), columns=["lim"], data=1
        )
    )
    _addLimitedComponent(esM, "Transmission", leadTime=5)
    esM.declareOptimizationProblem(timeSeriesAggregation=False)

    variables = _limitVariables(esM)
    assert variables, "the limit holds no variable"
    assert {index[-1] for _, index in variables} == {0}


# ---------------------------------------------------------------------------
# The limit binds the arrival in the optimum
# ---------------------------------------------------------------------------

_NODE = "node1"
_YEARS = [2020, 2025, 2030]


def _perYear(values, make):
    return {year: make(values.get(year, 0)) for year in _YEARS}


def _timeSeries(value):
    return pd.DataFrame([value], index=[0], columns=[_NODE])


def _locSeries(value):
    return pd.Series([value], index=[_NODE])


def _optimizePathway(
    componentLimit,
    leadTime,
    demand,
    stockCommissioning=None,
    commissioningMax=None,
):
    """One node, three periods of five years. "cheap" is limited, "backup" is
    expensive and unlimited, so the optimum builds "cheap" up to the limit.
    """
    esM = fn.EnergySystemModel(
        locations={_NODE},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW_el"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
        startYear=2020,
        numberOfInvestmentPeriods=len(_YEARS),
        investmentPeriodInterval=5,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=componentLimit,
        componentLimitEligibility=pd.DataFrame(index=[_NODE], columns=["lim"], data=1),
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="cheap",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1.0,
            economicLifetime=30,
            technicalLifetime=30,
            leadTime=leadTime,
            stockCommissioning=stockCommissioning,
            commissioningMax=commissioningMax,
            componentLimitID="lim",
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
            operationRateFix=_perYear(demand, _timeSeries),
        )
    )
    esM.optimize(
        timeSeriesAggregation=False, solver=ImplementedSolvers.STANDARD_SOLVER.value
    )
    return esM


def _designValues(esM, varName):
    var = getattr(esM.pyM, f"{varName}_srcSnk")
    return {ip: pyomo.value(var[_NODE, "cheap", ip]) for ip in esM.investmentPeriods}


def test_commissioning_limit_binds_the_decision_that_arrives_in_its_year():
    """A limit of 3 on 2025 caps the 2020 decision, which arrives in 2025. Bound
    to the decision period instead, it would cap the 2025 decision.
    """
    esM = _optimizePathway(
        componentLimit=_commissioningLimit(3.0, 2025),
        leadTime=5,
        demand={2025: 10, 2030: 10},
    )
    commis = _designValues(esM, "commis")
    cap = _designValues(esM, "cap")
    assert commis[0] == pytest.approx(3)
    assert commis[1] == pytest.approx(7)
    assert cap[1] == pytest.approx(3)
    assert cap[2] == pytest.approx(10)


def test_commissioning_limit_with_a_lead_time_that_varies_by_period():
    """With a lead time of 0 in 2020 and 5 years later, the 2025 decision arrives
    in 2030, so a limit on 2030 caps it.
    """
    esM = _optimizePathway(
        componentLimit=_commissioningLimit(3.0, 2030),
        leadTime={2020: 0, 2025: 5, 2030: 5},
        demand={2020: 5, 2025: 5, 2030: 10},
        # keeps the optimum from building the 2030 demand in 2020 already
        commissioningMax={
            2020: _locSeries(5),
            2025: _locSeries(100),
            2030: _locSeries(100),
        },
    )
    assert esM.getComponent("cheap").leadTimeVariesByInvestmentPeriod
    commis = _designValues(esM, "commis")
    assert commis[0] == pytest.approx(5)
    assert commis[1] == pytest.approx(3)
    assert _designValues(esM, "cap")[2] == pytest.approx(8)


def test_stock_arriving_in_the_year_counts_against_the_limit():
    """Stock decided in 2015 with a lead time of 10 years arrives in 2025 and
    takes 4 of the limit of 6, so the 2020 decision may only add 2.
    """
    esM = _optimizePathway(
        componentLimit=_commissioningLimit(6.0, 2025),
        leadTime={2015: 10, 2020: 5, 2025: 5, 2030: 5},
        stockCommissioning={2015: _locSeries(4)},
        demand={2025: 10, 2030: 10},
    )
    commis = _designValues(esM, "commis")
    cap = _designValues(esM, "cap")
    assert commis[0] == pytest.approx(2)
    assert cap[1] == pytest.approx(6)
    assert cap[2] == pytest.approx(10)


def test_commissioning_limit_over_a_range_sums_the_arrivals():
    """A range from 2025 to 2030 caps the decisions of 2020 and 2025 together,
    which arrive in that range.
    """
    esM = _optimizePathway(
        componentLimit=_commissioningLimit(6.0, 2025, ipEnd=2030),
        leadTime=5,
        demand={2025: 10, 2030: 10},
    )
    commis = _designValues(esM, "commis")
    assert commis[0] + commis[1] == pytest.approx(6)
    assert _designValues(esM, "cap")[2] == pytest.approx(6)


@pytest.mark.parametrize("bound", ["lower", "fixed"])
def test_a_lower_limit_that_only_stock_can_reach_raises(bound):
    """With a lead time of one period no decision arrives in 2020, so a limit of 3
    there would be skipped without a word. It raises instead.
    """
    with pytest.raises(ValueError, match="'lim'.*cannot be met in 2020: no decision"):
        _optimizePathway(
            componentLimit=_commissioningLimit(3.0, 2020, bound=bound),
            leadTime=5,
            demand={2025: 10, 2030: 10},
        )


def test_a_lower_limit_that_the_arriving_stock_meets_is_accepted():
    """Stock decided in 2015 with a lead time of 5 years brings 4 in 2020."""
    esM = _optimizePathway(
        componentLimit=_commissioningLimit(3.0, 2020, bound="lower"),
        leadTime={2015: 5, 2020: 5, 2025: 5, 2030: 5},
        stockCommissioning={2015: _locSeries(4)},
        demand={2020: 4, 2025: 10, 2030: 10},
    )
    assert _designValues(esM, "cap")[0] == pytest.approx(4)


def test_a_limit_without_a_contributing_component_stays_inactive():
    """A lower limit over a region where its component is not eligible builds no
    constraint and raises nothing, with or without a lead time.
    """
    esM = fn.EnergySystemModel(
        locations=set(_REGIONS),
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW_el"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
        startYear=2020,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=_commissioningLimit(3.0, 2020, bound="lower"),
        componentLimitEligibility=pd.DataFrame(
            index=_REGIONS, columns=["lim"], data=[[0], [1]]
        ),
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="comp",
            commodity="electricity",
            hasCapacityVariable=True,
            locationalEligibility=pd.Series({"R1": 1, "R2": 0}),
            investPerCapacity=1.0,
            leadTime=5,
            componentLimitID="lim",
        )
    )
    esM.declareOptimizationProblem(timeSeriesAggregation=False)
    assert len(esM.pyM.yearlyComponentLimitConstraint) == 0


@pytest.mark.parametrize(
    "decisionLeadTime",
    [
        pytest.param(5, id="a decision arrives as well"),
        pytest.param(10, id="only stock arrives"),
    ],
)
def test_arriving_stock_above_an_upper_limit_raises(decisionLeadTime):
    """Stock of 4 arriving in 2025 cannot fit under a limit of 3. The error names
    the limit instead of leaving an infeasible model to the solver.
    """
    with pytest.raises(ValueError, match="'lim'.*cannot be met in 2025"):
        _optimizePathway(
            componentLimit=_commissioningLimit(3.0, 2025),
            leadTime={
                2015: 10,
                2020: decisionLeadTime,
                2025: decisionLeadTime,
                2030: decisionLeadTime,
            },
            stockCommissioning={2015: _locSeries(4)},
            demand={2025: 10, 2030: 10},
        )
