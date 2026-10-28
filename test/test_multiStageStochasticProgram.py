"""Tests for the multi-stage stochastic programming module.

The model is a reduced version of the one in
``examples/08_Stochastic_Optimization/08b_Multi-Stage_Stochastic_Optimization.ipynb``:
two regions, four seasonal time steps, wind, gas turbines, a grid and an uncertain
electricity demand and gas price. The scenario tree has three stages over four
investment periods, so the second stage covers two investment periods.
"""

import copy
import warnings

import pandas as pd
import pyomo.environ as pyomo
import pytest

import fine as fn
from fine.utils import ImplementedSolvers

mssp = pytest.importorskip(
    "fine.expansionModules.multiStageStochasticProgram",
    reason="the multi-stage stochastic programming module needs mpi-sppy",
)
pytest.importorskip("mpisppy", reason="mpi-sppy is not installed")

NUMBER_OF_TIME_STEPS = 4
HOURS_PER_TIME_STEP = 2190
STAGE_INVESTMENT_PERIODS = [[0], [1, 2], [3]]

# Regression values, recorded on 2026-10-08 with the module as committed in 3f72abc0
# (branch mssp, before merging the tsam 4 / optimize refactor of develop), solved by
# Gurobi. Keys are (modelingClass, component, location, investmentPeriod).
EXPECTED_OBJECTIVE = 266.6292734384733
EXPECTED_FIRST_STAGE = {
    ("ConversionModel", "GasTurbine", "north", 0): 10.5,
    ("ConversionModel", "GasTurbine", "south", 0): 10.6,
    ("SourceSinkModel", "Wind", "north", 0): 60.0,
    ("SourceSinkModel", "Wind", "south", 0): 35.0,
}


def _timeSeries(values):
    return pd.DataFrame(values, index=range(NUMBER_OF_TIME_STEPS))


def _demand(northLevel, southLevel):
    """Electricity demand per time step (energy per time step, not average power)."""
    shape = {"north": [1.00, 0.90, 0.85, 1.05], "south": [1.00, 0.92, 0.88, 1.02]}
    return (
        _timeSeries(
            {
                "north": [northLevel * f for f in shape["north"]],
                "south": [southLevel * f for f in shape["south"]],
            }
        )
        * HOURS_PER_TIME_STEP
    )


def _gasPrice(euroPerMWh):
    return euroPerMWh * 1e-6


def _buildBaseModel():
    esM = fn.EnergySystemModel(
        locations={"north", "south"},
        commodities={"electricity", "naturalGas"},
        numberOfTimeSteps=NUMBER_OF_TIME_STEPS,
        commodityUnitsDict={"electricity": "GW_el", "naturalGas": "GW_CH4"},
        hoursPerTimeStep=HOURS_PER_TIME_STEP,
        costUnit="1e9 Euro",
        startYear=2020,
        numberOfInvestmentPeriods=4,
        investmentPeriodInterval=5,
        lengthUnit="km",
        verboseLogLevel=2,
    )
    esM.add(
        fn.Source(
            esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=_timeSeries(
                {"north": [0.45, 0.30, 0.25, 0.40], "south": [0.32, 0.24, 0.20, 0.28]}
            ),
            capacityMax=pd.Series({"north": 60.0, "south": 40.0}),
            investPerCapacity=1.20,
            opexPerCapacity=1.20 * 0.02,
            interestRate=0.05,
            economicLifetime=25,
            technicalLifetime=25,
        )
    )
    esM.add(
        fn.Source(
            esM,
            name="GasPurchase",
            commodity="naturalGas",
            hasCapacityVariable=False,
            commodityCost=_gasPrice(30),
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
            economicLifetime=25,
            technicalLifetime=25,
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
            economicLifetime=40,
            technicalLifetime=40,
        )
    )
    esM.add(
        fn.Sink(
            esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=_demand(30.0, 20.0),
        )
    )
    return esM


def _treeSpec():
    """Three stages: 2020 | 2025, 2030 | 2035, with non-uniform probabilities."""

    def node(parent, probability, demand, gasPrice):
        return {
            "parent": parent,
            "probability": probability,
            "values": {
                "Demand": {"operationRateFix": demand},
                "GasPurchase": {"commodityCost": gasPrice},
            },
        }

    return {
        "today": node(None, 1.0, _demand(30.0, 20.0), _gasPrice(30)),
        "growth": node(
            "today",
            0.65,
            {2025: _demand(33.0, 22.0), 2030: _demand(35.0, 23.5)},
            {2025: _gasPrice(36), 2030: _gasPrice(40)},
        ),
        "stagnation": node("today", 0.35, _demand(30.0, 20.0), _gasPrice(27)),
        "growth_gasCrisis": node("growth", 0.30, _demand(38.0, 25.0), _gasPrice(110)),
        "growth_cheapGas": node("growth", 0.70, _demand(38.0, 25.0), _gasPrice(28)),
        "stagnation_gasCrisis": node(
            "stagnation", 0.20, _demand(31.0, 21.0), _gasPrice(100)
        ),
        "stagnation_cheapGas": node(
            "stagnation", 0.80, _demand(31.0, 21.0), _gasPrice(25)
        ),
    }


def _commissioning(esM):
    """{(variableName, location, component, ip): value} for all commissioning variables."""
    values = {}
    for mdl in esM.componentModelingDict.values():
        for prefix in ("commis_", "commisBin_"):
            variable = getattr(esM.pyM, prefix + mdl.abbrvName, None)
            if variable is None:
                continue
            for key in variable:
                loc, compName, ip = key
                values[(prefix, loc, compName, ip)] = pyomo.value(variable[key])
    return values


def _solve(method="ef", **kwargs):
    return mssp.optimizeMultiStageStochastic(
        _buildBaseModel(),
        _treeSpec(),
        stageInvestmentPeriods=STAGE_INVESTMENT_PERIODS,
        method=method,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
        **kwargs,
    )


@pytest.fixture(scope="module")
def efResults():
    return _solve("ef")


# --------------------------------------------------------------------- scenario tree


def test_scenario_tree_structure():
    tree = mssp.ScenarioTree(_treeSpec())

    assert tree.stageCount == 3
    assert tree.scenarioNames == [
        "growth_cheapGas",
        "growth_gasCrisis",
        "stagnation_cheapGas",
        "stagnation_gasCrisis",
    ]
    assert tree.mpisppyNodeName == {
        "today": "ROOT",
        "growth": "ROOT_0",
        "growth_cheapGas": "ROOT_0_0",
        "growth_gasCrisis": "ROOT_0_1",
        "stagnation": "ROOT_1",
        "stagnation_cheapGas": "ROOT_1_0",
        "stagnation_gasCrisis": "ROOT_1_1",
    }
    assert tree.pathTo("stagnation_gasCrisis") == [
        "today",
        "stagnation",
        "stagnation_gasCrisis",
    ]
    assert tree.probabilityOf("growth_gasCrisis") == pytest.approx(0.65 * 0.30)
    assert sum(tree.probabilityOf(s) for s in tree.scenarioNames) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda t: t["growth"].update(probability=0.5), "must sum to 1"),
        (lambda t: t["stagnation"].update(parent=None), "exactly one root node"),
        (
            lambda t: t.pop("stagnation_cheapGas") and t.pop("stagnation_gasCrisis"),
            "same stage",
        ),
    ],
    ids=["siblingsDoNotSumToOne", "twoRoots", "leavesAtDifferentStages"],
)
def test_scenario_tree_rejects_invalid_trees(mutate, message):
    spec = _treeSpec()
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        mssp.ScenarioTree(spec)


# ------------------------------------------------------------------ input validation


def test_rejects_parameter_not_held_per_investment_period():
    spec = _treeSpec()
    spec["today"]["values"]["Wind"] = {"interestRate": 0.07}
    with pytest.raises(ValueError, match="cannot be uncertain"):
        mssp.optimizeMultiStageStochastic(
            _buildBaseModel(),
            spec,
            stageInvestmentPeriods=STAGE_INVESTMENT_PERIODS,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )


@pytest.mark.parametrize(
    "stageInvestmentPeriods, message",
    [
        ([[0], [1], [3]], "exactly once"),
        # the default of one stage per investment period needs four stages
        (None, "3 stages but the energy system model has 4 investment periods"),
    ],
    ids=["gap", "defaultMapping"],
)
def test_rejects_stage_mapping_that_does_not_cover_all_periods(
    stageInvestmentPeriods, message
):
    with pytest.raises(ValueError, match=message):
        mssp.optimizeMultiStageStochastic(
            _buildBaseModel(),
            _treeSpec(),
            stageInvestmentPeriods=stageInvestmentPeriods,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )


def test_scenario_model_receives_the_values_along_its_path():
    """Each scenario sees the values of every node on its path, per investment period."""
    tree = mssp.ScenarioTree(_treeSpec())
    esM = mssp.buildScenarioEnergySystemModel(
        _buildBaseModel(), tree, "growth_gasCrisis", STAGE_INVESTMENT_PERIODS
    )
    commodityCost = esM.getComponent("GasPurchase").processedCommodityCost
    expected = {0: _gasPrice(30), 1: _gasPrice(36), 2: _gasPrice(40), 3: _gasPrice(110)}
    for ip, price in expected.items():
        assert (commodityCost[ip] == price).all(), ip


def test_base_model_is_not_modified():
    baseModel = _buildBaseModel()
    reference = copy.deepcopy(baseModel)
    mssp.optimizeMultiStageStochastic(
        baseModel,
        _treeSpec(),
        stageInvestmentPeriods=STAGE_INVESTMENT_PERIODS,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    assert baseModel.pyM is None
    # The uncertain parameters are the ones the tree replaces in the scenario copies.
    assert (
        baseModel.getComponent("GasPurchase").commodityCost
        == reference.getComponent("GasPurchase").commodityCost
    )
    pd.testing.assert_frame_equal(
        baseModel.getComponent("Demand").operationRateFix,
        reference.getComponent("Demand").operationRateFix,
    )


# ------------------------------------------------------------------ extensive form


def test_ef_objective_is_probability_weighted_sum_of_scenario_objectives(efResults):
    tree = efResults.tree
    expected = sum(
        tree.probabilityOf(name) * esM.objectiveValue
        for name, esM in efResults.scenarioModels.items()
    )
    assert efResults.objectiveValue == pytest.approx(expected, rel=1e-9)


def test_ef_enforces_non_anticipativity(efResults):
    commis = {
        name: _commissioning(esM) for name, esM in efResults.scenarioModels.items()
    }
    keys = next(iter(commis.values())).keys()

    def values(scenarios, ips):
        return [{k: v for k, v in commis[s].items() if k[3] in ips} for s in scenarios]

    # Stage 1 (2020) is shared by all scenarios.
    stage1 = values(efResults.tree.scenarioNames, STAGE_INVESTMENT_PERIODS[0])
    for other in stage1[1:]:
        assert other == pytest.approx(stage1[0], abs=1e-6)

    # Stage 2 (2025, 2030) is shared within each branch.
    for branch in ("growth", "stagnation"):
        scenarios = [f"{branch}_gasCrisis", f"{branch}_cheapGas"]
        stage2 = values(scenarios, STAGE_INVESTMENT_PERIODS[1])
        assert stage2[1] == pytest.approx(stage2[0], abs=1e-6)

    # The branches genuinely adapt: somewhere after stage 1 the decisions differ.
    assert any(
        abs(commis["growth_gasCrisis"][k] - commis["stagnation_cheapGas"][k]) > 1e-3
        for k in keys
        if k[3] not in STAGE_INVESTMENT_PERIODS[0]
    )


def test_ef_regression_values(efResults):
    """Pin the current results so that refactorings can be checked against them."""
    assert efResults.objectiveValue == pytest.approx(EXPECTED_OBJECTIVE, rel=1e-6)
    decisions = {
        key: value
        for key, value in efResults.firstStageDecisions().items()
        if abs(value) > 1e-6
    }
    assert decisions.keys() == EXPECTED_FIRST_STAGE.keys()
    for key, value in EXPECTED_FIRST_STAGE.items():
        assert decisions[key] == pytest.approx(value, rel=1e-4, abs=1e-6), key


def test_an_infeasible_scenario_is_reported():
    """The 2020 gas turbines are still there in 2035, where one leaf allows none."""
    spec = _treeSpec()
    for name, node in spec.items():
        node["values"]["GasTurbine"] = {
            "capacityMax": 0.0 if name == "growth_gasCrisis" else 1000.0
        }
    with pytest.raises(RuntimeError, match="solvable on its own"):
        mssp.optimizeMultiStageStochastic(
            _buildBaseModel(),
            spec,
            stageInvestmentPeriods=STAGE_INVESTMENT_PERIODS,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )


def _solveWithTwoPeriodFirstStage():
    """Two stages, 2020 and 2025 | 2030 and 2035, with is-built binaries on GasTurbine."""
    baseModel = _buildBaseModel()
    baseModel.updateComponent(
        "GasTurbine",
        {"hasIsBuiltBinaryVariable": True, "bigM": 100, "investIfBuilt": 0.1},
    )

    def node(parent, probability, northDemand, southDemand):
        return {
            "parent": parent,
            "probability": probability,
            "values": {
                "Demand": {"operationRateFix": _demand(northDemand, southDemand)}
            },
        }

    tree = {
        "today": node(None, 1.0, 30.0, 20.0),
        "high": node("today", 0.5, 38.0, 25.0),
        "low": node("today", 0.5, 30.0, 20.0),
    }
    return mssp.optimizeMultiStageStochastic(
        baseModel,
        tree,
        stageInvestmentPeriods=[[0, 1], [2, 3]],
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )


def test_first_stage_decisions_cover_every_first_stage_period():
    results = _solveWithTwoPeriodFirstStage()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        decisions = results.firstStageDecisions()
        isBuilt = results.firstStageDecisions(isBuiltDecisions=True)

    assert {key[3] for key in decisions} == {0, 1}
    for scenarioName, esM in results.scenarioModels.items():
        commis = _commissioning(esM)
        for (_, compName, loc, ip), value in decisions.items():
            assert commis[("commis_", loc, compName, ip)] == pytest.approx(
                value, abs=1e-6
            ), scenarioName

    assert {(key[1], key[3]) for key in isBuilt} == {
        ("GasTurbine", 0),
        ("GasTurbine", 1),
    }
    assert all(
        value == pytest.approx(round(value), abs=1e-6) for value in isBuilt.values()
    )
    # GasTurbine capacity is commissioned in 2020, so it is built there
    assert any(isBuilt[key] == pytest.approx(1.0) for key in isBuilt if key[3] == 0)


def test_first_stage_decisions_warn_when_scenarios_disagree(efResults):
    esM = efResults.scenarioModels[efResults.tree.scenarioNames[-1]]
    variable = esM.pyM.commis_srcSnk["north", "Wind", 0]
    originalValue = variable.value
    variable.set_value(originalValue + 1.0, skip_validation=True)
    try:
        with pytest.warns(UserWarning, match="first-stage decisions .* differ"):
            decisions = efResults.firstStageDecisions()
    finally:
        variable.set_value(originalValue, skip_validation=True)
    assert decisions[("SourceSinkModel", "Wind", "north", 0)] == pytest.approx(
        EXPECTED_FIRST_STAGE[("SourceSinkModel", "Wind", "north", 0)]
    )


def test_first_stage_is_built_decisions_require_non_anticipative_binaries():
    results = mssp.StochasticOptimizationResults(
        tree=None,
        solverObject=None,
        scenarioModels={},
        objectiveValue=None,
        stageInvestmentPeriods=[[0]],
        includeIsBuiltBinaryVariables=False,
    )
    with pytest.raises(ValueError, match="not non-anticipative"):
        results.firstStageDecisions(isBuiltDecisions=True)


def test_ef_scenario_results_are_processed(efResults):
    esM = efResults.scenarioModels["growth_gasCrisis"]
    summary = esM.getOptimizationSummary("SourceSinkModel", ip=2035, outputLevel=0)
    assert ("Wind", "capacity") in {(comp, prop) for comp, prop, *_ in summary.index}


def test_ef_counts_as_converged(efResults):
    assert efResults.converged
    assert efResults.convergence["stopReason"] == "extensiveForm"


# ---------------------------------------------------------------- progressive hedging


def test_ph_agrees_with_ef(efResults):
    # rho has to fit the model's cost scale: with the module default of 1.0, PH reaches
    # consensus on a first-stage decision about 0.24% more expensive than the EF optimum,
    # because its convergence metric only measures agreement between the scenarios.
    phResults = _solve(
        "ph",
        mpisppyOptions={"PHIterLimit": 200, "defaultPHrho": 0.01, "convthresh": 1e-6},
    )
    assert phResults.objectiveValue == pytest.approx(efResults.objectiveValue, rel=1e-4)
    convergence = phResults.convergence
    assert phResults.converged
    assert convergence["stopReason"] == "threshold"
    assert convergence["convergenceMetric"] < 1e-6
    assert convergence["maxRelativeDeviation"] <= 1e-3


# one PH iteration leaves the scenarios far apart (up to 14 GW of wind)
_ONE_ITERATION = {"PHIterLimit": 1, "defaultPHrho": 0.01}


def test_ph_that_has_not_converged_raises():
    with pytest.raises(
        mssp.ProgressiveHedgingNotConvergedError, match="iteration limit"
    ):
        _solve("ph", mpisppyOptions=_ONE_ITERATION)


def test_ph_that_has_not_converged_warns_on_request():
    with pytest.warns(UserWarning, match="did not converge"):
        results = _solve("ph", mpisppyOptions=_ONE_ITERATION, onNonConvergence="warn")
    assert results.converged is False
    assert results.convergence["stopReason"] == "iterationLimit"
    assert results.convergence["iterations"] == 1
    assert results.convergence["maxDeviation"] > 1.0


def test_ph_mean_below_threshold_but_one_decision_off_is_not_converged():
    """mpi-sppy's metric is a mean, so a single deviating decision can hide in it."""
    with pytest.warns(UserWarning, match="did not converge"):
        results = _solve(
            "ph",
            mpisppyOptions={**_ONE_ITERATION, "convthresh": 10.0},
            onNonConvergence="warn",
        )
    assert results.convergence["stopReason"] == "threshold"
    assert results.converged is False


def test_invalid_reaction_to_non_convergence_is_refused():
    with pytest.raises(ValueError, match="onNonConvergence must be one of"):
        _solve("ph", onNonConvergence="ignore")


# ---------------------------------------------------------- time series aggregation


@pytest.mark.parametrize(
    "temporalAggregationSpecs",
    [
        # ETHOS.TSAM 3.x keywords, deprecated in FINE but still accepted.
        {
            "numberOfTypicalPeriods": NUMBER_OF_TIME_STEPS,
            "numberOfTimeStepsPerPeriod": 1,
            "segmentation": False,
        },
        # ETHOS.TSAM 4.x keywords; one period is one time step of 2190 hours.
        {
            "n_clusters": NUMBER_OF_TIME_STEPS,
            "period_duration": HOURS_PER_TIME_STEP,
            "segments": None,
        },
    ],
    ids=["tsam3Keywords", "tsam4Keywords"],
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_tsa_without_reduction_matches_full_resolution(
    efResults, temporalAggregationSpecs
):
    """With as many typical periods as periods, aggregation must not change the result.

    This exercises the per-scenario aggregation path while staying independent of the
    clustering algorithm's details.
    """
    tsaResults = _solve(
        "ef",
        timeSeriesAggregation=True,
        temporalAggregationSpecs=temporalAggregationSpecs,
    )
    assert tsaResults.objectiveValue == pytest.approx(
        efResults.objectiveValue, rel=1e-6
    )


# ------------------------------------------------------- updateComponent fidelity guard


def test_fidelity_guard_ignores_label_order_but_not_values():
    """Rebuilding a component may reorder a location-indexed parameter; that is no change."""
    original = pd.Series({"south": 0.05, "north": 0.05})
    reordered = pd.Series({"north": 0.05, "south": 0.05})
    changed = pd.Series({"north": 0.05, "south": 0.07})

    assert mssp._valuesEqual(original, reordered)
    assert not mssp._valuesEqual(original, changed)
    assert mssp._valuesEqual(
        _timeSeries({"south": [1, 2, 3, 4], "north": [5, 6, 7, 8]}),
        _timeSeries({"north": [5, 6, 7, 8], "south": [1, 2, 3, 4]}),
    )


def test_fidelity_guard_reports_a_parameter_lost_by_updateComponent(monkeypatch):
    """As if a constructor argument were not stored on the component under its name."""
    update = fn.EnergySystemModel.updateComponent

    def lossyUpdate(esM, componentName, updateAttrs):
        update(esM, componentName, updateAttrs)
        esM.getComponent(componentName).interestRate = 0.0

    monkeypatch.setattr(fn.EnergySystemModel, "updateComponent", lossyUpdate)
    with pytest.raises(
        RuntimeError, match=r"changed the parameters \['interestRate'\]"
    ):
        mssp.buildScenarioEnergySystemModel(
            _buildBaseModel(),
            mssp.ScenarioTree(_treeSpec()),
            "growth_gasCrisis",
            STAGE_INVESTMENT_PERIODS,
        )


def test_scenarios_with_different_design_variables_are_refused():
    """The last line of defense for mpi-sppy's positional matching of the variables."""
    tree = mssp.ScenarioTree(_treeSpec())
    northOnly = _buildBaseModel()
    northOnly.updateComponent(
        "GasTurbine", {"locationalEligibility": pd.Series({"north": 1, "south": 0})}
    )
    scenarioModels = {
        name: mssp.buildScenarioEnergySystemModel(
            baseModel, tree, name, STAGE_INVESTMENT_PERIODS
        )
        for name, baseModel in [
            ("growth_gasCrisis", _buildBaseModel()),
            ("growth_cheapGas", northOnly),
        ]
    }
    mssp._verifyStructuralConsistency(
        {"growth_gasCrisis": scenarioModels["growth_gasCrisis"]}
    )
    with pytest.raises(ValueError, match="do not describe the same system"):
        mssp._verifyStructuralConsistency(scenarioModels)
