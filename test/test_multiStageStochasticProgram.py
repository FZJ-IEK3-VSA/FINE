"""Tests for the multi-stage stochastic programming module.

The model is a reduced version of the one in
``examples/08_Stochastic_Optimization/08b_Multi-Stage_Stochastic_Optimization.ipynb``:
two regions, four seasonal time steps, wind, gas turbines, a grid and an uncertain
electricity demand and gas price. The scenario tree has three stages over four
investment periods, so the second stage covers two investment periods.
"""

import copy

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


def test_rejects_stage_mapping_that_does_not_cover_all_periods():
    with pytest.raises(ValueError, match="exactly once"):
        mssp.optimizeMultiStageStochastic(
            _buildBaseModel(),
            _treeSpec(),
            stageInvestmentPeriods=[[0], [1], [3]],
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )


@pytest.mark.parametrize(
    "node, key, message",
    [
        ("today", 2010, "stock years"),
        ("growth", 2035, "not investment periods of this node's stage"),
        ("growth", 2026, "not investment periods of this node's stage"),
    ],
    ids=["stockYear", "periodOfAnotherStage", "typo"],
)
def test_rejects_node_values_for_periods_outside_the_stage(node, key, message):
    """A key the node does not own would otherwise be ignored silently."""
    spec = _treeSpecWithUncertainWindInvest()
    stagePeriods = {"today": [2020], "growth": [2025, 2030]}[node]
    spec[node]["values"]["Wind"]["investPerCapacity"] = {
        **{year: 1.0 for year in stagePeriods},
        key: 1.0,
    }
    with pytest.raises(ValueError, match=message):
        mssp.optimizeMultiStageStochastic(
            _baseModelWithStock(),
            spec,
            stageInvestmentPeriods=STAGE_INVESTMENT_PERIODS,
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


# ---------------------------------------------------------------------- stock years

STOCK_YEAR = 2010
STOCK_IP = -2  # (2010 - 2020) / 5


def _baseModelWithStock(**windUpdates):
    """Build the base model with wind and grid capacity commissioned before 2020."""
    esM = _buildBaseModel()
    esM.updateComponent(
        "Wind",
        {
            "stockCommissioning": {STOCK_YEAR: pd.Series({"north": 5.0, "south": 0.0})},
            **windUpdates,
        },
    )
    esM.updateComponent(
        "Grid",
        {
            "stockCommissioning": {
                STOCK_YEAR: pd.Series({"north_south": 1.0, "south_north": 1.0})
            }
        },
    )
    return esM


def _treeSpecWithUncertain(compName, parameterName, nodeValues):
    spec = _treeSpec()
    for nodeName, value in nodeValues.items():
        spec[nodeName]["values"][compName] = {parameterName: value}
    return spec


def _treeSpecWithUncertainWindInvest():
    return _treeSpecWithUncertain(
        "Wind",
        "investPerCapacity",
        {
            "today": 1.20,
            "growth": 1.10,
            "stagnation": 1.15,
            "growth_gasCrisis": 0.90,
            "growth_cheapGas": 1.00,
            "stagnation_gasCrisis": 1.05,
            "stagnation_cheapGas": 1.10,
        },
    )


@pytest.mark.parametrize(
    "baseInvest, expectedStockInvest",
    [(1.20, 1.20), ({2010: 2.0, 2020: 1.2, 2025: 1.2, 2030: 1.2, 2035: 1.2}, 2.0)],
    ids=["scalarBaseValue", "dictBaseValue"],
)
def test_stock_years_keep_the_base_model_value(baseInvest, expectedStockInvest):
    """The tree covers the investment periods; the stock years come from the base model."""
    baseModel = _baseModelWithStock(investPerCapacity=baseInvest)
    esM = mssp.buildScenarioEnergySystemModel(
        baseModel,
        mssp.ScenarioTree(_treeSpecWithUncertainWindInvest()),
        "growth_gasCrisis",
        STAGE_INVESTMENT_PERIODS,
    )
    invest = esM.getComponent("Wind").processedInvestPerCapacity
    assert (invest[STOCK_IP] == expectedStockInvest).all()
    expected = {0: 1.20, 1: 1.10, 2: 1.10, 3: 0.90}
    for ip, value in expected.items():
        assert (invest[ip] == value).all(), ip


def test_uncertain_lead_time_with_stock():
    spec = _treeSpecWithUncertain(
        "Wind",
        "leadTime",
        {
            "today": 0,
            "growth": 5,
            "stagnation": 5,
            "growth_gasCrisis": 5,
            "growth_cheapGas": 10,
            "stagnation_gasCrisis": 5,
            "stagnation_cheapGas": 5,
        },
    )
    esM = mssp.buildScenarioEnergySystemModel(
        _baseModelWithStock(leadTime=0),
        mssp.ScenarioTree(spec),
        "growth_cheapGas",
        STAGE_INVESTMENT_PERIODS,
    )
    roundedIpLeadTime = esM.getComponent("Wind").roundedIpLeadTime
    expected = {STOCK_IP: 0, 0: 0, 1: 1, 2: 1, 3: 2}
    for ip, value in expected.items():
        assert (roundedIpLeadTime[ip] == value).all(), ip


def test_uncertain_two_dimensional_parameter_with_stock():
    spec = _treeSpecWithUncertain(
        "Grid",
        "investPerCapacity",
        {name: 0.001 if "growth" in name else 0.002 for name in _treeSpec()},
    )
    baseModel = _baseModelWithStock()
    esM = mssp.buildScenarioEnergySystemModel(
        baseModel,
        mssp.ScenarioTree(spec),
        "stagnation_cheapGas",
        STAGE_INVESTMENT_PERIODS,
    )
    # FINE scales a transmission's investPerCapacity by distance, so compare against the
    # base model (0.001 everywhere) rather than against the raw values.
    baseInvest = baseModel.getComponent("Grid").processedInvestPerCapacity
    invest = esM.getComponent("Grid").processedInvestPerCapacity
    pd.testing.assert_series_equal(invest[STOCK_IP], baseInvest[STOCK_IP])
    pd.testing.assert_series_equal(invest[3], 2 * baseInvest[3])


def test_ef_solves_with_stock_and_uncertain_cost():
    results = mssp.optimizeMultiStageStochastic(
        _baseModelWithStock(),
        _treeSpecWithUncertainWindInvest(),
        stageInvestmentPeriods=STAGE_INVESTMENT_PERIODS,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    expected = sum(
        results.tree.probabilityOf(name) * esM.objectiveValue
        for name, esM in results.scenarioModels.items()
    )
    assert results.objectiveValue == pytest.approx(expected, rel=1e-9)


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


def test_ef_scenario_results_are_processed(efResults):
    esM = efResults.scenarioModels["growth_gasCrisis"]
    summary = esM.getOptimizationSummary("SourceSinkModel", ip=2035, outputLevel=0)
    assert ("Wind", "capacity") in {(comp, prop) for comp, prop, *_ in summary.index}


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
