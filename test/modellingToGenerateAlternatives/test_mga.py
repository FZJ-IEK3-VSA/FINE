"""Tests for modelling to generate alternatives (MGA), fn.runMGA in fine/expansionModules/fine_mga.py.

Full MGA runs go through run_mga(); the unit tests call the helpers setMGAObjective and
addCostSlackConstraint via the `mga` import. These are the places to adapt if the MGA
functions move.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyomo.environ as pyomo
import pytest
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.repn import generate_standard_repn

import fine as fn
from fine.expansionModules import fine_mga as mga
from fine.utils import ImplementedSolvers

SLACK = 0.1

# persistent=True works with Gurobi only
requires_gurobi = pytest.mark.requires_solver("gurobi")

MAX_WIND = {
    "name": "max wind",
    "sense": "max",
    "terms": [{"component": "Wind", "variable": "capacity"}],
}
MIN_WIND = dict(MAX_WIND, name="min wind", sense="min")
MAX_IMPORT = {
    "name": "max import",
    "sense": "max",
    "terms": [{"component": "Electricity import", "variable": "operation"}],
}


def run_mga(esM, objectives, slack=SLACK, **kwargs):
    """Run the MGA of an already optimized energy system model for the given objectives."""
    kwargs.setdefault("solver", ImplementedSolvers.STANDARD_SOLVER.value)
    return fn.runMGA(esM, {"slack": slack, "objectives": objectives}, **kwargs)


def build_esM(**esMKwargs):
    """Two regions, six time steps; electricity import keeps the model feasible despite the limited wind capacity.

    esMKwargs are passed on to the EnergySystemModel, e.g. for investment periods.
    """
    locations = ["Region1", "Region2"]
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities={"electricity", "heat"},
        numberOfTimeSteps=6,
        commodityUnitsDict={"electricity": r"GW$_{el}$", "heat": r"GW$_{heat}$"},
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        **esMKwargs,
    )

    profile = [0.6, 0.7, 0.9, 1.0, 0.9, 0.8]
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                [[u * 10, u * 20] for u in profile], columns=locations
            ),
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Heat demand",
            commodity="heat",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                [[u * 5, u * 5] for u in profile], columns=locations
            ),
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity import",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.2,
        )
    )
    windProfile = [0.05, 0.15, 0.2, 0.0, 0.8, 0.7]
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame(
                [[u, u] for u in windProfile], columns=locations
            ),
            capacityMax=pd.Series([40, 40], index=locations),
            investPerCapacity=1.1,
            opexPerCapacity=0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Heat pump",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1 / 2.5, "heat": 1},
            hasCapacityVariable=True,
            investPerCapacity=0.76,
            opexPerCapacity=0.00646,
            interestRate=0.06,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.95,
            dischargeEfficiency=0.95,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
        )
    )
    esM.add(
        fn.Transmission(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=pd.DataFrame(
                [[0, 10], [10, 0]], index=locations, columns=locations
            ),
            investPerCapacity=0.05,
            interestRate=0.08,
            economicLifetime=40,
        )
    )
    return esM


def build_optimized_esM(**esMKwargs):
    esM = build_esM(**esMKwargs)
    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    return esM


def linear_terms(expr):
    """Return the constant and the coefficients per variable name of a linear expression."""
    repn = generate_standard_repn(expr)
    assert repn.is_linear()
    coefs = {}
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        coefs[var.name] = coefs.get(var.name, 0) + coef
    return repn.constant, coefs


def variables(esM, *varNames):
    """Map the names of all variables of the given Pyomo variable components to the variables."""
    return {
        var.name: var
        for varName in varNames
        for var in getattr(esM.pyM, varName).values()
    }


def objective_terms(esM, objSpec):
    """Declare the MGA objective and return its coefficients per variable name."""
    mga.setMGAObjective(esM, objSpec)
    constant, coefs = linear_terms(esM.pyM.Obj.expr)
    assert constant == 0
    return coefs


def expected_coefficients(esM, varName, component, weight, locations=None):
    """Coefficients of a weighted term without time series aggregation.

    Operation variables are weighted with their annual sum, i.e. divided by the number of
    modeled years, capacity variables with the weight itself.
    """
    isOperation = varName.startswith(("op_", "chargeOp_", "dischargeOp_"))
    factor = 1 / esM.numberOfYears if isOperation else 1
    return {
        name: weight * factor
        for name, var in variables(esM, varName).items()
        if var.index()[1] == component
        and (locations is None or var.index()[0] in locations)
    }


def summary_value(summary, model, component, property):
    """Sum of a property of a component over all locations in an optimization summary."""
    return summary[model].loc[(component, property)].sum().sum()


@pytest.fixture(scope="module")
def optimized():
    """Optimized model with its cost expression and cost, for declaring MGA objectives and constraints."""
    esM = build_optimized_esM()
    costExpr = esM.pyM.Obj.expr
    return SimpleNamespace(
        esM=esM,
        costExpr=costExpr,
        optimalCost=pyomo.value(costExpr),
        costTerms=linear_terms(costExpr),
    )


@pytest.fixture(scope="module")
def mga_run():
    """Optimized model after an MGA run with several alternatives, its optimum and the results."""
    esM = build_optimized_esM()
    optimum = esM.pyM.Obj()
    # The first alternative differs from the optimum (at maximum wind capacity), so a
    # result taken before solving it would be noticed
    results = run_mga(esM, [MIN_WIND, MAX_WIND, MAX_IMPORT])
    return esM, optimum, results


################################################################################
#                               Cost constraint                                #
################################################################################


def test_costConstraint(optimized):
    esM = optimized.esM
    mga.addCostSlackConstraint(esM, optimized.costExpr, optimized.optimalCost, SLACK)

    constraint = esM.pyM.mgaCostConstraint
    assert pyomo.value(constraint.upper) == pytest.approx(
        optimized.optimalCost * (1 + SLACK)
    )
    assert constraint.lower is None

    # The constrained cost is the objective function of the original optimization
    costConstant, costCoefs = optimized.costTerms
    constant, coefs = linear_terms(constraint.body)
    assert constant == pytest.approx(costConstant)
    assert coefs == pytest.approx(costCoefs)


def test_costConstraint_negative_optimum():
    # Revenues can make the optimal cost negative, the slack still loosens the bound
    pyM = pyomo.ConcreteModel()
    pyM.cost = pyomo.Var()
    esM = SimpleNamespace(pyM=pyM)

    mga.addCostSlackConstraint(esM, pyM.cost, -10.0, SLACK)

    assert pyomo.value(pyM.mgaCostConstraint.upper) == pytest.approx(-9.0)


################################################################################
#                                MGA objective                                 #
################################################################################


@pytest.mark.parametrize(
    "sense, pyomoSense",
    [
        ("max", pyomo.maximize),
        ("maximize", pyomo.maximize),
        ("min", pyomo.minimize),
        ("minimize", pyomo.minimize),
    ],
)
def test_objective_sense(optimized, sense, pyomoSense):
    mga.setMGAObjective(optimized.esM, dict(MAX_WIND, sense=sense))
    assert optimized.esM.pyM.Obj.sense == pyomoSense


@pytest.mark.xfail(strict=True, reason="an unknown sense silently minimizes")
def test_objective_invalid_sense(optimized):
    with pytest.raises(ValueError):
        mga.setMGAObjective(optimized.esM, dict(MAX_WIND, sense="foo"))


def test_objective_custom_terms(optimized):
    esM = optimized.esM
    coefs = objective_terms(
        esM,
        {
            "name": "custom",
            "terms": [
                {"component": "Wind", "variable": "capacity", "weight": 0.3},
                {"component": "Heat pump", "variable": "operation", "weight": 0.5},
                {
                    "component": "Batteries",
                    "variable": "discharge",
                    "weight": 2.0,
                    "locations": ["Region1"],
                },
            ],
        },
    )

    assert coefs == pytest.approx(
        {
            **expected_coefficients(esM, "cap_srcSnk", "Wind", 0.3),
            **expected_coefficients(esM, "op_conv", "Heat pump", 0.5),
            **expected_coefficients(
                esM, "dischargeOp_stor", "Batteries", 2.0, locations=["Region1"]
            ),
        }
    )


def test_objective_allCapacity(optimized):
    esM = optimized.esM
    coefs = objective_terms(
        esM,
        {
            "name": "all capacity",
            "variableSelection": "allCapacity",
            "defaultWeight": 0.5,
            "weights": {"Wind": 2.0, "Batteries": {"capacity": 3.0}},
            "excludeComponents": ["AC cables"],
        },
    )

    # Every capacity variable is weighted once, the storage capacity included
    assert coefs == pytest.approx(
        {
            **expected_coefficients(esM, "cap_srcSnk", "Wind", 2.0),
            **expected_coefficients(esM, "cap_conv", "Heat pump", 0.5),
            **expected_coefficients(esM, "cap_stor", "Batteries", 3.0),
        }
    )


def test_objective_allCapacityAndOperation(optimized):
    esM = optimized.esM
    coefs = objective_terms(
        esM,
        {
            "name": "all capacity and operation",
            "variableSelection": "allCapacityAndOperation",
            "weights": {"Batteries": {"charge": 0.2}},
        },
    )

    expected = {}
    for varName, component, weight in [
        ("cap_srcSnk", "Wind", 1.0),
        ("op_srcSnk", "Electricity demand", 1.0),
        ("op_srcSnk", "Heat demand", 1.0),
        ("op_srcSnk", "Electricity import", 1.0),
        ("op_srcSnk", "Wind", 1.0),
        ("cap_conv", "Heat pump", 1.0),
        ("op_conv", "Heat pump", 1.0),
        ("cap_stor", "Batteries", 1.0),
        ("chargeOp_stor", "Batteries", 0.2),
        ("dischargeOp_stor", "Batteries", 1.0),
        ("cap_trans", "AC cables", 1.0),
        ("op_trans", "AC cables", 1.0),
    ]:
        expected.update(expected_coefficients(esM, varName, component, weight))
    assert coefs == pytest.approx(expected)


@pytest.mark.parametrize(
    "objSpec, error, match",
    [
        (
            {"terms": [{"component": "Nuclear", "variable": "capacity"}]},
            KeyError,
            "Unknown component",
        ),
        (
            {"terms": [{"component": "Wind", "variable": "charging"}]},
            KeyError,
            "Unknown variable",
        ),
        (
            {"terms": [{"component": "Wind", "variable": "charge"}]},
            ValueError,
            "not available",
        ),
        (
            {"terms": [{"component": "Electricity import", "variable": "capacity"}]},
            ValueError,
            "No variables matched",
        ),
        (
            {
                "terms": [
                    {"component": "Wind", "variable": "capacity", "locations": ["X"]}
                ]
            },
            ValueError,
            "No variables matched",
        ),
        ({"terms": []}, ValueError, "has no terms"),
        ({"variableSelection": "everything"}, ValueError, "variableSelection"),
        (
            {"variableSelection": "allCapacity", "weights": {"Nuclear": 1.0}},
            KeyError,
            "Unknown components",
        ),
        (
            {"variableSelection": "allCapacity", "excludeComponents": ["Nuclear"]},
            KeyError,
            "Unknown components",
        ),
    ],
)
def test_objective_invalid_spec(optimized, objSpec, error, match):
    with pytest.raises(error, match=match):
        mga.setMGAObjective(optimized.esM, {"name": "invalid", **objSpec})


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="term investmentPeriods are compared with the investment period index, "
    "not with the year as documented",
)
def test_objective_investment_periods_as_years():
    esM = build_optimized_esM(
        numberOfInvestmentPeriods=2, investmentPeriodInterval=5, startYear=2020
    )
    term = {"component": "Wind", "variable": "capacity", "investmentPeriods": [2025]}
    coefs = objective_terms(esM, {"name": "wind 2025", "terms": [term]})

    capacities = variables(esM, "cap_srcSnk")
    assert {capacities[name].index()[2] for name in coefs} == {1}


################################################################################
#                                    runMGA                                    #
################################################################################


def test_runMGA_results(mga_run):
    _, optimum, results = mga_run
    assert list(results) == ["costOptimal", "min wind", "max wind", "max import"]
    assert results["costOptimal"]["cost"] == pytest.approx(optimum)
    # The cost-optimal solution of the optimized model is reused, not solved again
    assert results["costOptimal"]["status"] == "reused"
    for name in ["min wind", "max wind", "max import"]:
        assert {"cost", "status", "mgaObjective", "summary"} <= set(results[name]), name
        assert results[name]["status"] == "optimal", name


def test_runMGA_cost_within_slack(mga_run):
    _, optimum, results = mga_run
    for name, result in results.items():
        assert result["cost"] <= optimum * (1 + SLACK) * (1 + 1e-6), name


def test_runMGA_alternatives_follow_their_objective(mga_run):
    _, _, results = mga_run
    windCapacity = {
        name: summary_value(result["summary"], "SourceSinkModel", "Wind", "capacity")
        for name, result in results.items()
    }

    # Each summary belongs to its own alternative: the MGA objective is its wind capacity
    for name in ["max wind", "min wind"]:
        assert results[name]["mgaObjective"] == pytest.approx(windCapacity[name])
    assert windCapacity["max wind"] >= windCapacity["costOptimal"] - 1e-6
    assert windCapacity["min wind"] < windCapacity["costOptimal"] - 1e-3


def test_runMGA_operation_weighted_with_annual_sum(mga_run):
    _, _, results = mga_run
    result = results["max import"]
    assert result["mgaObjective"] == pytest.approx(
        summary_value(
            result["summary"],
            "SourceSinkModel",
            "Electricity import",
            "operation_annual",
        )
    )


def test_runMGA_zero_slack_keeps_optimal_cost():
    esM = build_optimized_esM()
    optimum = esM.pyM.Obj()
    results = run_mga(esM, [MIN_WIND], slack=0)
    assert results["min wind"]["cost"] == pytest.approx(optimum, rel=1e-6)


@pytest.mark.parametrize(
    "persistent", [False, pytest.param(True, marks=requires_gurobi)]
)
def test_runMGA_time_series_aggregation(persistent):
    # Three periods in two typical periods, so the typical periods occur differently often
    esM = build_esM()
    esM.aggregateTemporally(n_clusters=2, period_duration=2, segments=None)
    esM.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    optimum = esM.pyM.Obj()
    assert sorted(esM.periodOccurrences[0]) == [1, 2]

    results = run_mga(
        esM, [MAX_IMPORT], timeSeriesAggregation=True, persistent=persistent
    )

    # The operation of the typical periods is weighted with their occurrences, so the
    # objective is the annual operation
    _, coefs = linear_terms(esM.pyM.Obj.expr)
    operations = variables(esM, "op_srcSnk")
    assert coefs == pytest.approx(
        {
            name: esM.periodOccurrences[var.index()[2]][var.index()[3]]
            / esM.numberOfYears
            for name, var in operations.items()
            if var.index()[1] == "Electricity import"
        }
    )
    result = results["max import"]
    assert result["cost"] <= optimum * (1 + SLACK) * (1 + 1e-6)
    assert result["mgaObjective"] == pytest.approx(
        summary_value(
            result["summary"],
            "SourceSinkModel",
            "Electricity import",
            "operation_annual",
        )
    )


def test_runMGA_multiple_investment_periods():
    esM = build_optimized_esM(numberOfInvestmentPeriods=2, investmentPeriodInterval=5)
    optimum = esM.pyM.Obj()

    results = run_mga(esM, [MAX_WIND])

    # The capacities of both investment periods are part of the objective
    _, coefs = linear_terms(esM.pyM.Obj.expr)
    capacities = variables(esM, "cap_srcSnk")
    assert {capacities[name].index()[2] for name in coefs} == {0, 1}
    assert results["max wind"]["cost"] <= optimum * (1 + SLACK) * (1 + 1e-6)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="the summaries are requested for ip=0, which does not exist for "
    "startYear != 0 unless summaryKwargs names the investment period",
)
def test_runMGA_start_year():
    esM = build_optimized_esM(startYear=2020)
    results = run_mga(esM, [MAX_WIND])
    assert list(results) == ["costOptimal", "max wind"]


def test_runMGA_twice():
    # The second call must restore FINE's cost objective instead of taking the MGA
    # objective of the first call as the cost
    esM = build_optimized_esM()
    optimum = esM.pyM.Obj()
    first = run_mga(esM, [MAX_WIND])
    second = run_mga(esM, [MAX_WIND])
    assert second["costOptimal"]["cost"] == pytest.approx(optimum)
    assert second["max wind"]["cost"] == pytest.approx(first["max wind"]["cost"])
    assert second["max wind"]["mgaObjective"] == pytest.approx(
        first["max wind"]["mgaObjective"]
    )


@pytest.mark.xfail(
    strict=True, reason="objectives with the same name overwrite each other's results"
)
def test_runMGA_duplicate_objective_names():
    esM = build_optimized_esM()
    with pytest.raises(ValueError):
        run_mga(esM, [MAX_WIND, dict(MAX_WIND, sense="min")])


def test_runMGA_unoptimized_model(mga_run):
    # An undeclared model is declared and solved cost-optimally by runMGA itself
    _, optimum, expected = mga_run
    results = run_mga(build_esM(), [MAX_WIND], timeSeriesAggregation=False)
    assert results["costOptimal"]["cost"] == pytest.approx(optimum)
    assert results["max wind"]["cost"] == pytest.approx(expected["max wind"]["cost"])
    assert results["max wind"]["mgaObjective"] == pytest.approx(
        expected["max wind"]["mgaObjective"]
    )


@pytest.mark.xfail(
    strict=True,
    reason="slack is not validated; the infeasible alternatives are reported with "
    "the values of the previous solve",
)
def test_runMGA_negative_slack():
    esM = build_optimized_esM()
    with pytest.raises(ValueError):
        run_mga(esM, [MAX_WIND], slack=-0.1)


################################################################################
#                       Cost-optimal start and optimalCost                     #
################################################################################


def count_solves(monkeypatch):
    """Count the solves that go through esM.optimize (persistent solves bypass it)."""
    solves = []
    original = fn.EnergySystemModel._runOptimization

    def _runOptimization(self, *args, **kwargs):
        solves.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(fn.EnergySystemModel, "_runOptimization", _runOptimization)
    return solves


@pytest.mark.parametrize(
    "persistent", [False, pytest.param(True, marks=requires_gurobi)]
)
def test_runMGA_declared_model(mga_run, persistent):
    # A declared but not solved model is solved cost-optimally by runMGA itself
    _, optimum, expected = mga_run
    esM = build_esM()
    esM.declareOptimizationProblem(timeSeriesAggregation=False)
    results = run_mga(
        esM, [MAX_WIND], timeSeriesAggregation=False, persistent=persistent
    )
    assert results["costOptimal"]["cost"] == pytest.approx(optimum)
    assert results["costOptimal"]["status"] == "optimal"
    assert results["costOptimal"]["summary"] is not None
    assert results["max wind"]["mgaObjective"] == pytest.approx(
        expected["max wind"]["mgaObjective"]
    )


def test_runMGA_optimalCost(mga_run, monkeypatch):
    _, optimum, expected = mga_run
    esM = build_optimized_esM()
    solves = count_solves(monkeypatch)

    results = run_mga(esM, [MAX_WIND], optimalCost=optimum)

    # Only the alternative is solved; the cost-optimal entry holds the given value
    assert len(solves) == 1
    assert results["costOptimal"] == {
        "cost": optimum,
        "status": "provided",
        "summary": None,
    }
    assert results["max wind"]["mgaObjective"] == pytest.approx(
        expected["max wind"]["mgaObjective"]
    )


def test_runMGA_optimalCost_sets_bound():
    # The cost constraint is built from the given value, not from the solved optimum
    esM = build_optimized_esM()
    optimum = esM.pyM.Obj()
    results = run_mga(esM, [MAX_IMPORT], optimalCost=1.2 * optimum, slack=0)
    assert pyomo.value(esM.pyM.mgaCostConstraint.upper) == pytest.approx(1.2 * optimum)
    # With slack 0 and the solved optimum the cost could not rise
    assert optimum * (1 + 1e-6) < results["max import"]["cost"]
    assert results["max import"]["cost"] <= 1.2 * optimum * (1 + 1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="without persistent=True an undeclared model is solved cost-optimally "
    "although optimalCost is given",
)
def test_runMGA_optimalCost_skips_cost_optimal_solve(mga_run, monkeypatch):
    _, optimum, _ = mga_run
    solves = count_solves(monkeypatch)
    run_mga(build_esM(), [MAX_WIND], optimalCost=optimum, timeSeriesAggregation=False)
    assert len(solves) == 1


################################################################################
#                            Persistent Gurobi solver                          #
################################################################################


@requires_gurobi
def test_runMGA_persistent_matches_default(mga_run):
    # Same alternatives as the mga_run fixture, solved with the persistent solver
    _, optimum, expected = mga_run
    esM = build_optimized_esM()
    results = run_mga(esM, [MIN_WIND, MAX_WIND, MAX_IMPORT], persistent=True)

    assert list(results) == list(expected)
    assert results["costOptimal"]["cost"] == pytest.approx(optimum)
    assert results["costOptimal"]["status"] == "reused"
    for name in ["min wind", "max wind", "max import"]:
        result = results[name]
        assert result["status"] == "optimal", name
        assert result["mgaObjective"] == pytest.approx(
            expected[name]["mgaObjective"]
        ), name
        assert result["cost"] <= optimum * (1 + SLACK) * (1 + 1e-6), name
    # The summaries are post-processed from the persistent solve
    for name in ["min wind", "max wind"]:
        assert summary_value(
            results[name]["summary"], "SourceSinkModel", "Wind", "capacity"
        ) == pytest.approx(results[name]["mgaObjective"]), name
    assert esM.solverSpecs["solver"] == "gurobi_persistent"


@requires_gurobi
def test_runMGA_persistent_unoptimized_model(mga_run, monkeypatch):
    # An undeclared model is declared and also solved cost-optimally with the
    # persistent solver, so esM.optimize is never called
    _, optimum, expected = mga_run
    solves = count_solves(monkeypatch)
    results = run_mga(build_esM(), [MAX_WIND], persistent=True)
    assert solves == []
    assert results["costOptimal"]["cost"] == pytest.approx(optimum)
    assert results["costOptimal"]["status"] == "optimal"
    assert results["max wind"]["mgaObjective"] == pytest.approx(
        expected["max wind"]["mgaObjective"]
    )


@requires_gurobi
def test_runMGA_persistent_twice(mga_run):
    _, optimum, expected = mga_run
    esM = build_optimized_esM()
    run_mga(esM, [MAX_WIND], persistent=True)
    results = run_mga(esM, [MAX_WIND], persistent=True)
    assert results["costOptimal"]["cost"] == pytest.approx(optimum)
    assert results["max wind"]["mgaObjective"] == pytest.approx(
        expected["max wind"]["mgaObjective"]
    )


@requires_gurobi
def test_runMGA_persistent_infeasible_status():
    # An infeasible alternative is reported by its status instead of silently
    esM = build_optimized_esM()
    results = run_mga(esM, [MAX_WIND], slack=-0.1, persistent=True)
    assert results["max wind"]["status"].startswith("gurobi status ")


@requires_gurobi
def test_runMGA_persistent_writes_solverModelFile(tmp_path):
    gurobipy = pytest.importorskip("gurobipy")
    esM = build_optimized_esM()
    optimum = esM.pyM.Obj()
    modelFile = tmp_path / "model.mps"
    run_mga(esM, [MAX_WIND], persistent=True, solverModelFile=str(modelFile))

    # The file holds the cost-optimal model as transferred to Gurobi
    with gurobipy.Env(params={"OutputFlag": 0}) as env:
        model = gurobipy.read(str(modelFile), env)
        model.optimize()
        assert model.ObjVal == pytest.approx(optimum)
        model.dispose()


@requires_gurobi
def test_runMGA_persistent_solver_options(tmp_path, monkeypatch):
    # The optimize arguments are passed on to the persistent solver
    created = []
    create = mga._createPersistentSolver

    def _createPersistentSolver(*args, **kwargs):
        created.append(create(*args, **kwargs))
        return created[-1]

    monkeypatch.setattr(mga, "_createPersistentSolver", _createPersistentSolver)
    # A relative name: Pyomo's option string parsing drops the backslashes of a
    # Windows path, in esM.optimize as well
    monkeypatch.chdir(tmp_path)
    run_mga(
        build_optimized_esM(),
        [MAX_WIND],
        persistent=True,
        threads=1,
        timeLimit=100,
        logFileName="gurobi.log",
    )
    (opt,) = created
    assert opt.options["Threads"] == 1
    assert opt.options["timelimit"] == 100
    assert (tmp_path / "gurobi.log").stat().st_size > 0


def test_runMGA_solverModelFile_requires_persistent(tmp_path):
    esM = build_optimized_esM()
    with pytest.raises(ValueError, match="persistent=True"):
        run_mga(esM, [MAX_WIND], solverModelFile=str(tmp_path / "model.mps"))


def test_runMGA_persistent_requires_gurobi():
    esM = build_optimized_esM()
    with pytest.raises(ValueError, match="gurobi"):
        run_mga(esM, [MAX_WIND], persistent=True, solver="glpk")


def test_runMGA_persistent_unknown_optimize_argument():
    esM = build_optimized_esM()
    with pytest.raises(TypeError, match="unknownArgument"):
        run_mga(esM, [MAX_WIND], persistent=True, unknownArgument=1)


class SuboptimalGurobi:
    """Persistent Gurobi solver that reports every solve as SUBOPTIMAL (status 13).

    This is what Gurobi reports when its barrier stalls close to the optimum. The
    real solve is optimal, so a loaded solution is the correct one.
    """

    class _Model:
        Status = 13

        def __init__(self, model):
            self._model = model

        def __getattr__(self, name):
            return getattr(self._model, name)

    def __init__(self, opt):
        self._opt = opt

    def __getattr__(self, name):
        return getattr(self._opt, name)

    @property
    def _solver_model(self):
        return self._Model(self._opt._solver_model)

    def solve(self, *args, **kwargs):
        """Solve and report the result as Pyomo translates Gurobi's SUBOPTIMAL status."""
        solverInfo = self._opt.solve(*args, **kwargs)
        solverInfo.solver.status = SolverStatus.warning
        solverInfo.solver.termination_condition = TerminationCondition.other
        return solverInfo


@pytest.fixture
def suboptimal_gurobi(monkeypatch):
    create = mga._createPersistentSolver
    monkeypatch.setattr(
        mga,
        "_createPersistentSolver",
        lambda *args, **kwargs: SuboptimalGurobi(create(*args, **kwargs)),
    )


@requires_gurobi
def test_runMGA_persistent_accepts_suboptimal(mga_run, suboptimal_gurobi):
    _, _, expected = mga_run
    esM = build_optimized_esM()
    results = run_mga(esM, [MAX_WIND], persistent=True)

    # The solution is loaded and post-processed despite the SUBOPTIMAL status
    result = results["max wind"]
    assert result["status"].startswith("suboptimal (accepted")
    assert result["mgaObjective"] == pytest.approx(expected["max wind"]["mgaObjective"])
    assert summary_value(
        result["summary"], "SourceSinkModel", "Wind", "capacity"
    ) == pytest.approx(expected["max wind"]["mgaObjective"])


@requires_gurobi
def test_runMGA_persistent_rejects_suboptimal(suboptimal_gurobi):
    esM = build_optimized_esM()
    results = run_mga(esM, [MAX_WIND], persistent=True, acceptSuboptimal=False)
    # The solution is not loaded, which the status makes visible
    assert results["max wind"]["status"] == "suboptimal"


################################################################################
#                                Excel output                                  #
################################################################################


def test_runMGA_writes_excel(tmp_path):
    esM = build_optimized_esM()
    outdir = Path(tmp_path) / "mga"
    results = run_mga(
        esM,
        [MAX_WIND, MIN_WIND],
        save_mga_results_in_excel=True,
        excelOutputDir=str(outdir),
        excelFilePrefix="Test",
    )

    # One file per solution, named after the objective with file-system safe characters
    fileNames = {
        "costOptimal": "Test_costOptimal.xlsx",
        "max wind": "Test_max_wind.xlsx",
        "min wind": "Test_min_wind.xlsx",
    }
    assert sorted(path.name for path in outdir.iterdir()) == sorted(fileNames.values())

    # Every file holds its own solution, compared by the wind capacity
    for name, result in results.items():
        assert Path(result["excelFile"]) == outdir / fileNames[name]
        summary = pd.read_excel(
            result["excelFile"],
            sheet_name="SourceSinkOptSummary_1dim",
            index_col=[0, 1, 2],
        )
        assert summary.loc[("Wind", "capacity")].sum().sum() == pytest.approx(
            summary_value(result["summary"], "SourceSinkModel", "Wind", "capacity")
        ), name
