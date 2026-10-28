"""Tests for fine.expansionModules.sequentialStochasticOptimization.

The model is a small yearly pathway: two regions, four seasonal time steps, wind, gas
turbines and a growing demand. Its wind capacity factors are the "realized" weather of
every year; the trees of the windows hold them in the root and a calm and a windy forecast
in the second stage.
"""

import copy
import warnings

import pandas as pd
import pyomo.environ as pyomo
import pytest

import fine as fn
from fine.utils import ImplementedSolvers

pytest.importorskip("mpisppy", reason="mpi-sppy is not installed")
sso = pytest.importorskip(
    "fine.expansionModules.sequentialStochasticOptimization",
    reason="the sequential stochastic optimization needs mpi-sppy",
)
from fine.expansionModules.rollingHorizon import (  # noqa: E402
    _previousCommissioning,
    rollingHorizonOptimization,
)

NUMBER_OF_TIME_STEPS = 4
HOURS_PER_TIME_STEP = 2190
START_YEAR = 2020
SOLVER = ImplementedSolvers.STANDARD_SOLVER.value
MSSP_SETTINGS = {"solver": SOLVER}


def _timeSeries(values):
    return pd.DataFrame(values, index=range(NUMBER_OF_TIME_STEPS))


def _realizedWind(year):
    """Wind capacity factors of a year; every year's weather is a little different."""
    factor = 1.0 + 0.05 * ((year * 7) % 5 - 2)
    return _timeSeries(
        {
            "north": [0.45 * factor, 0.30 * factor, 0.25 * factor, 0.40 * factor],
            "south": [0.32 * factor, 0.24 * factor, 0.20 * factor, 0.28 * factor],
        }
    )


def _demand(year):
    """Electricity demand per time step, growing by 2 % a year."""
    growth = 1.02 ** (year - START_YEAR)
    shape = {"north": [1.00, 0.90, 0.85, 1.05], "south": [1.00, 0.92, 0.88, 1.02]}
    level = {"north": 30.0, "south": 20.0}
    return _timeSeries(
        {
            region: [level[region] * growth * f * HOURS_PER_TIME_STEP for f in factors]
            for region, factors in shape.items()
        }
    )


def pathwayEsM(numberOfInvestmentPeriods=10, windKwargs=None, **esMKwargs):
    years = [START_YEAR + i for i in range(numberOfInvestmentPeriods)]
    esM = fn.EnergySystemModel(
        locations={"north", "south"},
        commodities={"electricity", "naturalGas"},
        numberOfTimeSteps=NUMBER_OF_TIME_STEPS,
        commodityUnitsDict={"electricity": "GW_el", "naturalGas": "GW_CH4"},
        hoursPerTimeStep=HOURS_PER_TIME_STEP,
        costUnit="1e9 Euro",
        startYear=START_YEAR,
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        investmentPeriodInterval=1,
        lengthUnit="km",
        verboseLogLevel=2,
        **esMKwargs,
    )
    wind = {
        "name": "Wind",
        "commodity": "electricity",
        "hasCapacityVariable": True,
        "operationRateMax": {year: _realizedWind(year) for year in years},
        "capacityMax": pd.Series({"north": 80.0, "south": 50.0}),
        "investPerCapacity": 1.20,
        "opexPerCapacity": 1.20 * 0.02,
        "interestRate": 0.05,
        "economicLifetime": 15,
        "technicalLifetime": 15,
    }
    wind.update(windKwargs or {})
    esM.add(fn.Source(esM, **wind))
    esM.add(
        fn.Source(
            esM,
            name="GasPurchase",
            commodity="naturalGas",
            hasCapacityVariable=False,
            commodityCost=60e-6,
        )
    )
    esM.add(
        fn.Conversion(
            esM,
            name="GasTurbine",
            physicalUnit="GW_el",
            commodityConversionFactors={"naturalGas": -1 / 0.6, "electricity": 1.0},
            hasCapacityVariable=True,
            investPerCapacity=0.60,
            opexPerCapacity=0.60 * 0.03,
            interestRate=0.05,
            economicLifetime=15,
            technicalLifetime=15,
        )
    )
    esM.add(
        fn.Transmission(
            esM,
            name="Grid",
            commodity="electricity",
            hasCapacityVariable=True,
            distances=pd.DataFrame(
                [[0, 400], [400, 0]],
                index=["north", "south"],
                columns=["north", "south"],
            ),
            losses=0.0001,
            investPerCapacity=0.001,
            interestRate=0.05,
            economicLifetime=30,
            technicalLifetime=30,
        )
    )
    esM.add(
        fn.Sink(
            esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix={year: _demand(year) for year in years},
        )
    )
    return esM


class RecordingTreeBuilder:
    """Root = realized wind, stage 2 = a calm and a windy forecast of it.

    Records the arguments of every call. With ``branches=False`` the tree has a single
    path through the realized wind, i.e. it is deterministic with perfect foresight over
    the window, which is what a rolling horizon solves.
    """

    def __init__(self, branches=True):
        self.branches = branches
        self.calls = []

    def __call__(
        self, windowEsM, windowYears, stageYears, realizedYears, deterministic
    ):
        """Record the call and return the window's tree."""
        self.calls.append(
            {
                "windowYears": windowYears,
                "stageYears": stageYears,
                "realizedYears": realizedYears,
                "deterministic": deterministic,
            }
        )

        def windValues(years, factor):
            realized = sso.realizedValues(
                windowEsM, {"Wind": ["operationRateMax"]}, years
            )
            return {
                "Wind": {
                    "operationRateMax": {
                        year: (value * factor).clip(upper=1.0)
                        for year, value in realized["Wind"]["operationRateMax"].items()
                    }
                }
            }

        tree = {"root": {"values": windValues(stageYears[0], 1.0)}}
        if len(stageYears) == 1:
            return tree
        if self.branches and not deterministic:
            tree["calm"] = {
                "parent": "root",
                "probability": 0.4,
                "values": windValues(stageYears[1], 0.7),
            }
            tree["windy"] = {
                "parent": "root",
                "probability": 0.6,
                "values": windValues(stageYears[1], 1.2),
            }
        else:
            tree["realized"] = {
                "parent": "root",
                "probability": 1.0,
                "values": windValues(stageYears[1], 1.0),
            }
        return tree


def _run(esM=None, builder=None, **kwargs):
    kwargs.setdefault("windowLength", 4)
    kwargs.setdefault("stageInvestmentPeriods", [[0, 1], [2, 3]])
    kwargs.setdefault("msspSettings", MSSP_SETTINGS)
    return sso.sequentialStochasticOptimization(
        esM if esM is not None else pathwayEsM(),
        buildTree=builder if builder is not None else RecordingTreeBuilder(),
        **kwargs,
    )


@pytest.fixture(scope="module")
def extendRun():
    """Two-year first stage, four-year windows, decisions implemented up to 2025."""
    builder = RecordingTreeBuilder()
    results = _run(builder=builder, lastYear=2025)
    return results, builder


# --------------------------------------------------------------------- window layout


def test_windows_move_on_by_the_first_stage(extendRun):
    results, builder = extendRun
    assert [window.years for window in results.windows] == [
        [2020, 2021, 2022, 2023],
        [2022, 2023, 2024, 2025],
        [2024, 2025, 2026, 2027],
    ]
    assert [window.implementedYears for window in results.windows] == [
        [2020, 2021],
        [2022, 2023],
        [2024, 2025],
    ]
    assert [call["stageYears"] for call in builder.calls] == [
        [[2020, 2021], [2022, 2023]],
        [[2022, 2023], [2024, 2025]],
        [[2024, 2025], [2026, 2027]],
    ]
    assert [call["realizedYears"] for call in builder.calls] == [
        [],
        [2020, 2021],
        [2020, 2021, 2022, 2023],
    ]
    assert not any(call["deterministic"] for call in builder.calls)


def test_the_implemented_pathway_covers_every_year_once(extendRun):
    results, _ = extendRun
    table = results.implementedCommissioningTable()
    assert list(table.columns) == list(range(2020, 2026))
    assert ("SourceSinkModel", "Wind", "north") in table.index
    assert ("TransmissionModel", "Grid", "north_south") in table.index


def test_a_last_window_shorter_than_the_first_stage_is_cut_at_last_year():
    builder = RecordingTreeBuilder()
    results = _run(builder=builder, lastYear=2026)
    assert [window.startYear for window in results.windows] == [2020, 2022, 2024, 2026]
    assert results.windows[-1].implementedYears == [2026]
    assert sorted({key[3] for key in results.implementedCommissioning}) == list(
        range(2020, 2027)
    )


def test_extend_requires_data_beyond_last_year():
    with pytest.raises(ValueError, match="needs 2 more investment period"):
        _run(lastYear=2029)


@pytest.mark.parametrize(
    "numberOfInvestmentPeriods, lastWindow, lastStages",
    [
        (10, [2028, 2029], [[2028, 2029]]),
        (9, [2026, 2027, 2028], [[2026, 2027], [2028]]),
    ],
    ids=["one-stage", "two-stage"],
)
def test_the_deterministic_last_window_covers_the_remaining_years(
    numberOfInvestmentPeriods, lastWindow, lastStages
):
    builder = RecordingTreeBuilder()
    results = _run(
        pathwayEsM(numberOfInvestmentPeriods),
        builder=builder,
        endOfHorizon="deterministic",
    )
    last = results.windows[-1]
    assert last.deterministic
    assert not any(window.deterministic for window in results.windows[:-1])
    assert all(len(window.years) == 4 for window in results.windows[:-1])
    assert last.years == lastWindow
    assert last.implementedYears == lastWindow
    assert builder.calls[-1]["deterministic"]
    assert builder.calls[-1]["stageYears"] == lastStages
    years = sorted({key[3] for key in results.implementedCommissioning})
    assert years == [START_YEAR + i for i in range(numberOfInvestmentPeriods)]
    assert last.objectiveValue == pytest.approx(
        last.results.scenarioModels[last.tree.scenarioNames[0]].pyM.Obj()
    )
    assert last.convergence == {"converged": True, "stopReason": "deterministic"}


def test_aggregation_without_reduction_matches_full_resolution():
    """With as many typical periods as periods, aggregating every window, the
    deterministic one included, must not change the windows' objective values.
    """
    tsaSpecs = {
        "n_clusters": NUMBER_OF_TIME_STEPS,
        "period_duration": HOURS_PER_TIME_STEP,
        "segments": None,
    }
    full = _run(pathwayEsM(7), endOfHorizon="deterministic")
    aggregated = _run(
        pathwayEsM(7),
        endOfHorizon="deterministic",
        msspSettings={
            **MSSP_SETTINGS,
            "timeSeriesAggregation": True,
            "temporalAggregationSpecs": tsaSpecs,
        },
    )
    assert aggregated.windows[-1].deterministic
    for fullWindow, aggregatedWindow in zip(full.windows, aggregated.windows):
        assert aggregatedWindow.objectiveValue == pytest.approx(
            fullWindow.objectiveValue, rel=1e-6
        )


def test_the_deterministic_last_window_needs_a_single_path():
    class TwoLeaves(RecordingTreeBuilder):
        def __call__(self, windowEsM, **kwargs):
            kwargs["deterministic"] = False
            return super().__call__(windowEsM, **kwargs)

    with pytest.raises(ValueError, match="single path"):
        _run(pathwayEsM(9), builder=TwoLeaves(), endOfHorizon="deterministic")


# ------------------------------------------------------------------- hand-over


def _scenarioModels(window):
    return list(window.results.scenarioModels.values())


def test_the_implemented_decisions_become_the_stock_of_the_next_window(extendRun):
    results, _ = extendRun
    for previous, window in zip(results.windows, results.windows[1:]):
        for esM in _scenarioModels(window):
            for compName in ("Wind", "GasTurbine", "Grid"):
                stock = esM.getComponent(compName).stockCommissioning or {}
                for year in previous.implementedYears:
                    expected = _previousCommissioning(
                        _scenarioModels(previous)[0], compName, year
                    )
                    if expected.sum() > 1e-5:
                        pd.testing.assert_series_equal(
                            stock[year].sort_index(),
                            expected.round(10).sort_index(),
                            check_names=False,
                        )
                    else:
                        assert year not in stock


def test_the_implemented_decisions_are_the_first_stage_decisions(extendRun):
    results, _ = extendRun
    window = results.windows[0]
    for esM in _scenarioModels(window):
        for (_, compName, loc, year), value in window.implementedCommissioning.items():
            if compName != "Wind":
                continue
            ip = window.years.index(year)
            assert pyomo.value(
                esM.pyM.commis_srcSnk[loc, compName, ip]
            ) == pytest.approx(value, abs=1e-6)


def test_the_branches_adapt_after_the_first_stage(extendRun):
    """Wind reaches its capacityMax in the first year, so the calm branch makes up for
    its weaker wind with more gas turbines in the second stage.
    """
    results, _ = extendRun
    window = results.windows[0]
    calm, windy = (window.results.scenarioModels[name] for name in ("calm", "windy"))
    assert any(
        abs(
            pyomo.value(calm.pyM.commis_conv[loc, "GasTurbine", ip])
            - pyomo.value(windy.pyM.commis_conv[loc, "GasTurbine", ip])
        )
        > 1e-3
        for loc in ("north", "south")
        for ip in (2, 3)
    )


def test_a_passed_on_decision_keeps_its_lead_time():
    """Decided in 2020 with a lead time of 3 years, wind arrives in 2023, after the next
    window has started in 2022. Handed over as stock of 2020 with the lead time of 0 that a
    scalar leadTime gives stock years, it would be there in 2022 already.
    """
    years = [START_YEAR + i for i in range(8)]
    esM = pathwayEsM(
        8,
        windKwargs={
            "leadTime": 3,
            "commissioningFix": {
                year: pd.Series({"north": 5.0 if year == 2020 else 0.0, "south": 0.0})
                for year in years
            },
        },
    )
    results = _run(esM, lastYear=2023)
    secondWindow = results.windows[1]
    assert secondWindow.startYear == 2022
    for scenarioEsM in _scenarioModels(secondWindow):
        capacity = scenarioEsM.pyM.cap_srcSnk
        assert pyomo.value(capacity["north", "Wind", 0]) == pytest.approx(0.0, abs=1e-6)
        assert pyomo.value(capacity["north", "Wind", 1]) == pytest.approx(5.0)


# ---------------------------------------------------- equivalence and side effects


@pytest.mark.parametrize(
    "windKwargs",
    [None, {"economicLifetime": 2, "technicalLifetime": 2, "investPerCapacity": 0.2}],
    ids=["long-lived", "retiring-stock"],
)
def test_a_single_path_with_a_one_year_first_stage_is_a_rolling_horizon(windKwargs):
    """With one-year steps and a tree that is the realized data along a single path, every
    window is the window of rollingHorizonOptimization with the same years. With a two-year
    lifetime, the wind built in 2020 retires and is rebuilt in 2022, so the stock handed to
    the last window has to keep the 2022 wind and drop the 2020 wind.
    """
    esM = pathwayEsM(6, windKwargs=windKwargs)
    results = _run(
        esM,
        builder=RecordingTreeBuilder(branches=False),
        windowLength=3,
        stageInvestmentPeriods=[[0], [1, 2]],
        lastYear=2023,
    )
    rolling = rollingHorizonOptimization(
        esM,
        numberOfInvestmentPeriodsForRollingHorizon=3,
        timeSeriesAggregation=False,
        solver=SOLVER,
    )
    assert [window.startYear for window in results.windows] == list(rolling)
    for window in results.windows:
        rollingEsM = rolling[window.startYear]
        assert window.objectiveValue == pytest.approx(
            rollingEsM.objectiveValue, rel=1e-6
        )
        for compName in ("Wind", "GasTurbine", "Grid"):
            expected = _previousCommissioning(rollingEsM, compName, window.startYear)
            actual = _previousCommissioning(
                _scenarioModels(window)[0], compName, window.startYear
            )
            pd.testing.assert_series_equal(
                actual.sort_index(), expected.sort_index(), atol=1e-5
            )


def test_the_passed_esM_is_not_modified():
    esM = pathwayEsM()
    before = copy.deepcopy(fn.dictIO.exportToDict(esM))
    _run(esM, lastYear=2021)
    after = fn.dictIO.exportToDict(esM)
    assert esM.pyM is None
    assert after[1]["Source"]["Wind"]["stockCommissioning"] is None
    assert before[0]["startYear"] == after[0]["startYear"]
    assert esM.rollingHorizonStartYear is None


def test_without_kept_results_only_the_decisions_remain():
    results = _run(lastYear=2023, keepWindowResults=False)
    assert all(window.results is None for window in results.windows)
    assert sorted({key[3] for key in results.implementedCommissioning}) == list(
        range(2020, 2024)
    )


# ------------------------------------------------------------------ evaluation


def _allCommissioning(esM):
    """{(modelingClass, component, location, year): value} of a solved model."""
    commissioning = {}
    for modelingClass, mdl in esM.componentModelingDict.items():
        variable = getattr(esM.pyM, "commis_" + mdl.abbrvName, None)
        if variable is None:
            continue
        for (loc, compName, ip), data in variable.items():
            if ip in esM.investmentPeriods:
                year = esM.investmentPeriodNames[ip]
                commissioning[(modelingClass, compName, loc, year)] = data.value
    return commissioning


def test_fixing_a_perfect_foresight_solution_reproduces_its_objective():
    perfectForesight = pathwayEsM(5)
    perfectForesight.optimize(solver=SOLVER)
    evaluation = sso.evaluateImplementedPathway(
        pathwayEsM(5), _allCommissioning(perfectForesight), solver=SOLVER
    )
    assert evaluation.objectiveValue == pytest.approx(
        perfectForesight.objectiveValue, rel=1e-6
    )


def test_the_evaluation_fixes_the_implemented_decisions(extendRun):
    results, _ = extendRun
    evaluation = sso.evaluateImplementedPathway(pathwayEsM(), results, solver=SOLVER)
    assert list(evaluation.investmentPeriodNames) == list(range(2020, 2026))
    solved = _allCommissioning(evaluation)
    for key, value in results.implementedCommissioning.items():
        assert solved[key] == pytest.approx(max(value, 0.0), abs=1e-6), key


def test_the_evaluation_needs_the_first_years_without_gaps(extendRun):
    results, _ = extendRun
    decisions = {
        key: value
        for key, value in results.implementedCommissioning.items()
        if key[3] != 2021
    }
    with pytest.raises(ValueError, match="without gaps"):
        sso.evaluateImplementedPathway(pathwayEsM(), decisions, solver=SOLVER)


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"stageInvestmentPeriods": [[0, 1, 2, 3]]}, ValueError, "at least two"),
        ({"stageInvestmentPeriods": [[0, 1], [3]]}, ValueError, "exactly once"),
        ({"endOfHorizon": "truncate"}, ValueError, "endOfHorizon must be one of"),
        ({"lastYear": 2019}, ValueError, "not an investment period"),
        ({"msspSettings": {"processResults": False}}, ValueError, "processResults"),
        ({"optimizeSettings": {"solver": SOLVER}}, ValueError, "solver"),
    ],
    ids=["one-stage", "gap", "endOfHorizon", "lastYear", "mssp", "optimize"],
)
def test_invalid_arguments_are_refused(kwargs, error, message):
    with pytest.raises(error, match=message):
        _run(**kwargs)


def test_settings_a_window_cannot_inherit_are_refused():
    with pytest.raises(NotImplementedError, match="annuityPerpetuity"):
        _run(pathwayEsM(annuityPerpetuity=True))


def test_realized_values_read_per_year_and_single_values():
    esM = pathwayEsM(3)
    values = sso.realizedValues(
        esM,
        {"Wind": ["operationRateMax", "investPerCapacity"]},
        [2021, 2022],
    )
    pd.testing.assert_frame_equal(
        values["Wind"]["operationRateMax"][2022], _realizedWind(2022)
    )
    assert values["Wind"]["investPerCapacity"] == {2021: 1.2, 2022: 1.2}
    with pytest.raises(ValueError, match="no value for"):
        sso.realizedValues(esM, {"Wind": ["operationRateMax"]}, [2030])


def test_no_warning_about_disagreeing_scenarios(extendRun):
    results, _ = extendRun
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for window in results.windows:
            window.results.firstStageDecisions()


# ------------------------------------------------------------------ convergence

# One progressive hedging iteration leaves the scenarios of a window apart, if they want
# different first-stage wind: with the default capacityMax, both build the maximum and
# agree from the start, so wind is uncapped here.
_PH_ONE_ITERATION = {
    **MSSP_SETTINGS,
    "method": "ph",
    "mpisppyOptions": {"PHIterLimit": 1, "defaultPHrho": 0.01},
}


def _uncappedWindEsM():
    return pathwayEsM(
        windKwargs={"capacityMax": pd.Series({"north": 1000.0, "south": 1000.0})}
    )


def test_every_window_records_its_convergence(extendRun):
    results, _ = extendRun
    assert all(
        window.convergence == {"converged": True, "stopReason": "extensiveForm"}
        for window in results.windows
    )


def test_a_window_whose_progressive_hedging_has_not_converged_stops_the_run():
    with pytest.raises(
        sso.mssp.ProgressiveHedgingNotConvergedError, match="Window 2020-2023: "
    ):
        _run(_uncappedWindEsM(), lastYear=2021, msspSettings=_PH_ONE_ITERATION)


def test_a_window_that_has_not_converged_is_recorded_on_request():
    with pytest.warns(UserWarning, match="did not converge"):
        results = _run(
            _uncappedWindEsM(),
            lastYear=2021,
            msspSettings={**_PH_ONE_ITERATION, "onNonConvergence": "warn"},
            keepWindowResults=False,
        )
    convergence = results.windows[0].convergence
    assert convergence["converged"] is False
    assert convergence["stopReason"] == "iterationLimit"
