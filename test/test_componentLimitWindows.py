"""componentLimit in the expansion modules: rolling horizon, sequential driver, MSSP.

Every window keeps the componentLimit rows of its own years. A commissioning row counts
the capacity that becomes available in its year, so a decision of an earlier window that
arrives as stock counts against the limit of a later window. Every scenario model of a
multi-stage stochastic program carries the componentLimit of its base model. These tests
need the componentLimit feature together with leadTime and the expansion modules.
"""

from types import SimpleNamespace

import pandas as pd
import pyomo.environ as pyomo
import pytest

import fine as fn
from fine.expansionModules.rollingHorizon import (
    _filterComponentLimitForInterval,
    _previousCommissioning,
    rollingHorizonOptimization,
)
from fine.utils import ImplementedSolvers

SOLVER = ImplementedSolvers.STANDARD_SOLVER.value
_COLUMNS = ["value", "bound", "type", "commodity", "ip", "ipEnd"]
_NODE = "node1"


def _yearlyCommissioningLimits(years, value):
    """One upper commissioning limit per year, named after the year."""
    return pd.DataFrame(
        index=[f"lim_{year}" for year in years],
        columns=_COLUMNS,
        data=[[value, "upper", "commissioning", None, year, None] for year in years],
    )


# ---------------------------------------------------------------------------
# Rows and tables of one window
# ---------------------------------------------------------------------------

_PATHWAY = SimpleNamespace(investmentPeriodNames=[2020, 2025, 2030, 2035])


def _esmDict(componentLimit):
    ids = list(dict.fromkeys(componentLimit.index))
    return {
        "componentLimit": componentLimit,
        "componentLimitEligibility": pd.DataFrame(
            index=["R1", "R2"], columns=ids, data=1
        ),
        "componentLimitEligibility2dim": None,
        "componentLimitGrouping": pd.DataFrame(
            index=["R1", "R2"], columns=ids, data="DE"
        ),
    }


def test_a_window_keeps_the_rows_of_its_years():
    componentLimit = pd.DataFrame(
        index=["single_2020", "single_2025", "same_2030", "range_2025_2030", "late"],
        columns=_COLUMNS,
        data=[
            [1.0, "upper", "commissioning", None, 2020, None],
            [1.0, "upper", "capacity", None, 2025, None],
            [1.0, "upper", "operation", None, 2030, 2030],
            [1.0, "upper", "commissioning", None, 2025, 2030],
            [1.0, "upper", "commissioning", None, 2035, None],
        ],
    )
    arguments, keptIDs = _filterComponentLimitForInterval(
        _esmDict(componentLimit), [2025, 2030], _PATHWAY
    )
    expected = ["single_2025", "same_2030", "range_2025_2030"]
    assert keptIDs == set(expected)
    assert list(arguments["componentLimit"].index) == expected
    assert list(arguments["componentLimitEligibility"].columns) == expected
    assert list(arguments["componentLimitGrouping"].columns) == expected
    assert arguments["componentLimitEligibility2dim"] is None


def test_a_window_without_rows_has_no_componentLimit():
    componentLimit = _yearlyCommissioningLimits([2020], 1.0)
    arguments, keptIDs = _filterComponentLimitForInterval(
        _esmDict(componentLimit), [2025, 2030], _PATHWAY
    )
    assert keptIDs == set()
    assert all(value is None for value in arguments.values())


def test_a_row_without_ip_belongs_to_the_first_year_of_the_pathway():
    """Without an ip a row applies to the first investment period of the pathway, not
    to the first one of a later window.
    """
    componentLimit = pd.DataFrame(
        index=["noIp"],
        columns=["value", "bound", "type", "commodity"],
        data=[[1.0, "upper", "commissioning", None]],
    )
    first, _ = _filterComponentLimitForInterval(
        _esmDict(componentLimit), [2020, 2025], _PATHWAY
    )
    later, keptIDs = _filterComponentLimitForInterval(
        _esmDict(componentLimit), [2025, 2030], _PATHWAY
    )
    assert first["componentLimit"].loc["noIp", "ip"] == 2020
    assert keptIDs == set()


def test_a_range_that_crosses_the_window_is_refused():
    componentLimit = pd.DataFrame(
        index=["budget"],
        columns=_COLUMNS,
        data=[[6.0, "upper", "commissioning", None, 2025, 2035]],
    )
    with pytest.raises(NotImplementedError, match="budget"):
        _filterComponentLimitForInterval(
            _esmDict(componentLimit), [2025, 2030], _PATHWAY
        )


def test_without_componentLimit_nothing_changes():
    assert _filterComponentLimitForInterval(
        {"componentLimit": None}, [2025, 2030], _PATHWAY
    ) == ({}, None)


# ---------------------------------------------------------------------------
# Rolling horizon
# ---------------------------------------------------------------------------

_YEARS = [2020, 2025, 2030, 2035]


def _pathway(componentLimit):
    """One node, four periods of five years. "cheap" arrives five years after the
    decision and at most 3 per year may arrive; "backup" covers the rest.
    """
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
        componentLimitEligibility=pd.DataFrame(
            index=[_NODE], columns=list(componentLimit.index), data=1
        ),
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
            leadTime=5,
            componentLimitID=list(componentLimit.index),
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
            operationRateFix={
                year: pd.DataFrame([0.0 if year == 2020 else 10.0], columns=[_NODE])
                for year in _YEARS
            },
        )
    )
    return esM


def test_a_rolling_horizon_with_yearly_limits_matches_perfect_foresight():
    """At most 3 may arrive per year. Every window sees the limits of its own years,
    and the decision of an earlier window arrives in the next one as stock, which
    takes the limit of its arrival year there. Since arriving later never pays off,
    every window builds the 3 that may arrive next, as perfect foresight does.
    """
    componentLimit = _yearlyCommissioningLimits(_YEARS, 3.0)
    perfectForesight = _pathway(componentLimit)
    perfectForesight.optimize(timeSeriesAggregation=False, solver=SOLVER)
    expected = {
        year: pyomo.value(perfectForesight.pyM.commis_srcSnk[_NODE, "cheap", ip])
        for ip, year in enumerate(_YEARS)
    }
    assert expected == pytest.approx({2020: 3, 2025: 3, 2030: 3, 2035: 0}, abs=1e-6)

    windows = rollingHorizonOptimization(
        _pathway(componentLimit),
        numberOfInvestmentPeriodsForRollingHorizon=2,
        timeSeriesAggregation=False,
        solver=SOLVER,
    )
    assert list(windows) == [2020, 2025, 2030]
    # every window builds only the rows and IDs of its own years
    assert list(windows[2025].componentLimit.index) == ["lim_2025", "lim_2030"]
    assert windows[2025].getComponent("cheap").componentLimitID == [
        "lim_2025",
        "lim_2030",
    ]
    implemented = {
        2020: _previousCommissioning(windows[2020], "cheap", 2020)[_NODE],
        2025: _previousCommissioning(windows[2025], "cheap", 2025)[_NODE],
        2030: _previousCommissioning(windows[2030], "cheap", 2030)[_NODE],
        2035: _previousCommissioning(windows[2030], "cheap", 2035)[_NODE],
    }
    assert implemented == pytest.approx(expected, abs=1e-6)


def test_a_rolling_horizon_refuses_a_range_across_windows_before_solving():
    componentLimit = pd.concat(
        [
            _yearlyCommissioningLimits(_YEARS, 3.0),
            pd.DataFrame(
                index=["budget"],
                columns=_COLUMNS,
                data=[[6.0, "upper", "commissioning", None, 2025, 2030]],
            ),
        ]
    )
    esM = _pathway(componentLimit)
    with pytest.raises(NotImplementedError, match="budget"):
        rollingHorizonOptimization(
            esM,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
            solver=SOLVER,
        )


# ---------------------------------------------------------------------------
# Sequential stochastic optimization
# ---------------------------------------------------------------------------

_SEQUENTIAL_YEARS = list(range(2020, 2026))


def _yearlyPathway():
    """One node, yearly periods, a growing demand; "cheap" arrives a year after the
    decision and at most 3 per year may arrive.
    """
    componentLimit = _yearlyCommissioningLimits(_SEQUENTIAL_YEARS, 3.0)
    esM = fn.EnergySystemModel(
        locations={_NODE},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW_el"},
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
        startYear=_SEQUENTIAL_YEARS[0],
        numberOfInvestmentPeriods=len(_SEQUENTIAL_YEARS),
        investmentPeriodInterval=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=componentLimit,
        componentLimitEligibility=pd.DataFrame(
            index=[_NODE], columns=list(componentLimit.index), data=1
        ),
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
            leadTime=1,
            componentLimitID=list(componentLimit.index),
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
            operationRateFix={
                year: pd.DataFrame([4.0 * (year - 2020)], columns=[_NODE])
                for year in _SEQUENTIAL_YEARS
            },
        )
    )
    return esM


def _demandTree(windowEsM, windowYears, stageYears, realizedYears, deterministic):
    """Root = realized demand, stage 2 = a low and a high forecast of it."""
    sso = pytest.importorskip("fine.expansionModules.sequentialStochasticOptimization")

    def demand(years, factor):
        realized = sso.realizedValues(
            windowEsM, {"demand": ["operationRateFix"]}, years
        )
        return {
            "demand": {
                "operationRateFix": {
                    year: value * factor
                    for year, value in realized["demand"]["operationRateFix"].items()
                }
            }
        }

    tree = {"root": {"values": demand(stageYears[0], 1.0)}}
    for name, factor in (("low", 0.9), ("high", 1.1)):
        tree[name] = {
            "parent": "root",
            "probability": 0.5,
            "values": demand(stageYears[1], factor),
        }
    return tree


def test_the_sequential_driver_keeps_the_yearly_limits_of_every_window():
    """Every window is an MSSP over its own years with the limits of these years. The
    implemented decisions arrive in the next year, at most 3 of them, and the
    evaluation of the implemented pathway keeps the limits too.
    """
    pytest.importorskip("mpisppy", reason="mpi-sppy is not installed")
    sso = pytest.importorskip("fine.expansionModules.sequentialStochasticOptimization")
    esM = _yearlyPathway()
    results = sso.sequentialStochasticOptimization(
        esM,
        windowLength=4,
        stageInvestmentPeriods=[[0, 1], [2, 3]],
        buildTree=_demandTree,
        lastYear=2023,
        msspSettings={"solver": SOLVER},
    )
    assert [window.startYear for window in results.windows] == [2020, 2022]
    implemented = {
        year: value
        for (_, compName, _, year), value in results.implementedCommissioning.items()
        if compName == "cheap"
    }
    # demand grows by 4 a year, so the 3 that may arrive are always built
    assert implemented == pytest.approx({2020: 3, 2021: 3, 2022: 3, 2023: 3}, abs=1e-6)

    evaluation = sso.evaluateImplementedPathway(
        esM, results, solver=SOLVER, timeSeriesAggregation=False
    )
    assert list(evaluation.componentLimit.index) == [
        f"lim_{year}" for year in range(2020, 2024)
    ]


# ---------------------------------------------------------------------------
# Multi-stage stochastic program
# ---------------------------------------------------------------------------

_REGIONS = ["north", "south"]
_HOURS = 2190


def _stochasticBaseModel():
    """Two regions, four periods of five years, every kind of componentLimit row:
    the wind arriving in 2025 (lead time of one period), the share of the wind
    capacity per region in 2030, the gas turbine operation in 2030 and the grid
    commissioning of 2020 over the north-south connection.
    """
    componentLimit = pd.DataFrame(
        index=["windArrivals", "windShare", "gasOperation", "gridCommissioning"],
        columns=_COLUMNS,
        data=[
            [5.0, "upper", "commissioning", None, 2025, None],
            [0.7, "shareMax", "capacity", None, 2030, None],
            [250000.0, "upper", "operation", None, 2030, None],
            [5.0, "upper", "commissioning", None, 2020, None],
        ],
    )
    esM = fn.EnergySystemModel(
        locations=set(_REGIONS),
        commodities={"electricity", "naturalGas"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"electricity": "GW_el", "naturalGas": "GW_CH4"},
        hoursPerTimeStep=_HOURS,
        costUnit="1e9 Euro",
        startYear=2020,
        numberOfInvestmentPeriods=4,
        investmentPeriodInterval=5,
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=componentLimit,
        componentLimitEligibility=pd.DataFrame(
            index=_REGIONS,
            columns=["windArrivals", "windShare", "gasOperation"],
            data=1,
        ),
        componentLimitEligibility2dim=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("north", "south")]),
            columns=["gridCommissioning"],
            data=1,
        ),
    )
    esM.add(
        fn.Source(
            esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame(
                {"north": [0.45, 0.30, 0.25, 0.40], "south": [0.32, 0.24, 0.20, 0.28]}
            ),
            capacityMax=pd.Series({"north": 60.0, "south": 40.0}),
            investPerCapacity=1.2,
            interestRate=0.05,
            economicLifetime=25,
            technicalLifetime=25,
            leadTime=5,
            componentLimitID=["windArrivals", "windShare"],
        )
    )
    esM.add(
        fn.Source(
            esM,
            name="GasPurchase",
            commodity="naturalGas",
            hasCapacityVariable=False,
            commodityCost=30e-6,
        )
    )
    esM.add(
        fn.Conversion(
            esM,
            name="GasTurbine",
            physicalUnit="GW_el",
            commodityConversionFactors={"naturalGas": -1 / 0.6, "electricity": 1.0},
            hasCapacityVariable=True,
            investPerCapacity=0.6,
            interestRate=0.05,
            economicLifetime=25,
            technicalLifetime=25,
            componentLimitID="gasOperation",
        )
    )
    esM.add(
        fn.Transmission(
            esM,
            name="Grid",
            commodity="electricity",
            hasCapacityVariable=True,
            distances=pd.DataFrame(
                [[0, 400], [400, 0]], index=_REGIONS, columns=_REGIONS
            ),
            investPerCapacity=0.001,
            interestRate=0.05,
            economicLifetime=40,
            technicalLifetime=40,
            componentLimitID="gridCommissioning",
        )
    )
    esM.add(
        fn.Sink(
            esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                {
                    "north": [30.0 * _HOURS] * 4,
                    "south": [20.0 * _HOURS] * 4,
                }
            ),
        )
    )
    return esM


def test_every_scenario_model_of_an_mssp_keeps_the_componentLimit():
    """The wind investment cost is uncertain, so every scenario rebuilds the wind
    component through updateComponent. Its componentLimitIDs survive, the scenarios
    pass the structural check, and every limit holds in every scenario.
    """
    pytest.importorskip("mpisppy", reason="mpi-sppy is not installed")
    mssp = pytest.importorskip("fine.expansionModules.multiStageStochasticProgram")
    tree = {
        "today": {"values": {"Wind": {"investPerCapacity": 1.2}}},
        "cheapWind": {
            "parent": "today",
            "probability": 0.5,
            "values": {"Wind": {"investPerCapacity": 0.6}},
        },
        "dearWind": {
            "parent": "today",
            "probability": 0.5,
            "values": {"Wind": {"investPerCapacity": 2.4}},
        },
    }
    results = mssp.optimizeMultiStageStochastic(
        _stochasticBaseModel(),
        tree,
        stageInvestmentPeriods=[[0], [1, 2, 3]],
        solver=SOLVER,
    )

    assert set(results.scenarioModels) == {"cheapWind", "dearWind"}
    for scenarioEsM in results.scenarioModels.values():
        assert scenarioEsM.getComponent("Wind").componentLimitID == [
            "windArrivals",
            "windShare",
        ]
        pyM = scenarioEsM.pyM
        assert {key[0] for key in pyM.yearlyComponentLimitDict} == {
            "windArrivals",
            "gasOperation",
            "gridCommissioning",
        }
        # with a lead time of one period, the 2020 decision arrives in 2025
        windArrivals = sum(
            pyomo.value(pyM.commis_srcSnk[region, "Wind", 0]) for region in _REGIONS
        )
        assert windArrivals == pytest.approx(5.0, abs=1e-6)
        windCapacity2030 = {
            region: pyomo.value(pyM.cap_srcSnk[region, "Wind", 2])
            for region in _REGIONS
        }
        assert windCapacity2030["north"] <= 0.7 * sum(windCapacity2030.values()) + 1e-6
        assert windCapacity2030["south"] <= 0.7 * sum(windCapacity2030.values()) + 1e-6
        gridCommissioning2020 = (
            pyomo.value(pyM.commis_trans["north_south", "Grid", 0])
            + pyomo.value(pyM.commis_trans["south_north", "Grid", 0])
        ) / 2
        assert gridCommissioning2020 <= 5.0 + 1e-6
