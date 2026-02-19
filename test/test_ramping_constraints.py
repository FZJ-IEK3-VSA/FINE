import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import fine as fn


def build_test_system(rampUp=None, rampDown=None):
    """Create a simple electrolyzer system for testing ramp constraints."""
    esM = fn.EnergySystemModel(
        locations={"test"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={"electricity": "GW", "hydrogen": "GW"},
        numberOfTimeSteps=48,
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    # Source and sink for feasibility
    esM.add(
        fn.Source(
            esM=esM,
            name="grid",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=10.0,
            opexPerOperation=0.0,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="H2_sink",
            commodity="hydrogen",
            hasCapacityVariable=False,
            opexPerOperation=0.0,
        )
    )

    # Conversion component with optional ramp limits
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit="GW",
            commodityConversionFactors={"electricity": -1, "hydrogen": 1},
            hasCapacityVariable=True,
            capacityMax=10.0,
            investPerCapacity=0.0,
            opexPerCapacity=0.0,
            rampUpMax=rampUp,
            rampDownMax=rampDown,
        )
    )
    return esM


def maximize_period_boundary_jump_up(esM):
    """Force optimization to push the inter-period ramp constraint to its limit."""
    esM.aggregateTemporally(numberOfTypicalPeriods=2)

    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    pyM = esM.pyM
    last_timestep_p0 = max(t for (p, t) in pyM.intraYearTimeSet if p == 0)
    key_before = ("test", "Electrolyzer", 0, 0, last_timestep_p0)
    key_after = ("test", "Electrolyzer", 0, 1, 0)

    for obj in pyM.component_objects(pyo.Objective, active=True):
        obj.deactivate()

    pyM.obj = pyo.Objective(
        expr=pyM.op_conv[key_after] - pyM.op_conv[key_before], sense=pyo.maximize
    )

    pyo.SolverFactory("glpk").solve(pyM)
    return pyM, key_before, key_after


def maximize_period_boundary_jump_down(esM):
    """Build TSA model and choose an objective that maximizes the downward jump at the period boundary: op_before - op_after."""
    esM.aggregateTemporally(numberOfTypicalPeriods=2)
    esM.optimize(timeSeriesAggregation=True, solver="glpk")  # or your equivalent
    pyM = esM.pyM

    # Last timestep of period 0, first timestep of period 1
    last_timestep_p0 = max(t for (p, t) in pyM.intraYearTimeSet if p == 0)
    key_before = ("test", "Electrolyzer", 0, 0, last_timestep_p0)
    key_after = ("test", "Electrolyzer", 0, 1, 0)

    # Replace existing objectives
    for obj in pyM.component_objects(pyo.Objective, active=True):
        obj.deactivate()

    # Maximize the drop: op_before - op_after
    pyM.obj = pyo.Objective(
        expr=pyM.op_conv[key_before] - pyM.op_conv[key_after],
        sense=pyo.maximize,
    )

    pyo.SolverFactory("glpk").solve(pyM)
    return pyM, key_before, key_after


def test_real_esm_tsa_interperiod_rampup_enforcement():
    """Verify that inter-period ramping limits the operation jump correctly, with debugging prints to inspect intermediate values."""
    # --- Unconstrained case (no ramping limits) ---
    system_A = build_test_system(rampUp=None, rampDown=None)
    model_A, key_before, key_after = maximize_period_boundary_jump_up(system_A)

    capacity_A = pyo.value(model_A.cap_conv["test", "Electrolyzer", 0])
    op_before_A = pyo.value(model_A.op_conv[key_before])
    op_after_A = pyo.value(model_A.op_conv[key_after])
    jump_unconstrained = op_after_A - op_before_A

    # --- Constrained case ---
    system_B = build_test_system(rampUp=0.2, rampDown=0.2)
    model_B, key_before, key_after = maximize_period_boundary_jump_up(system_B)

    capacity_B = pyo.value(model_B.cap_conv["test", "Electrolyzer", 0])
    op_before_B = pyo.value(model_B.op_conv[key_before])
    op_after_B = pyo.value(model_B.op_conv[key_after])
    jump_constrained = op_after_B - op_before_B

    # Inter-period ramp-up constraint
    constraint_key = ("test", "Electrolyzer", 0, 1, 0)

    assert hasattr(model_B, "ConstrInterPeriod_rampUpMax_conv")
    assert not hasattr(model_A, "ConstrInterPeriod_rampUpMax_conv")

    constraint = model_B.ConstrInterPeriod_rampUpMax_conv[constraint_key]
    repn = generate_standard_repn(constraint.body)

    # ---- Extract coefficient of cap_var in constraint ----
    cap_coefficient = None
    for var, coeff in zip(repn.linear_vars, repn.linear_coefs):
        if var is model_B.cap_conv["test", "Electrolyzer", 0]:
            cap_coefficient = coeff
            break

    # rampUpMax = 0.2 → dt = |coef| / rampUpMax
    timestep = abs(cap_coefficient) / 0.2
    expected_limit = 0.2 * timestep * capacity_B

    # --- Assertions ---
    assert np.isclose(capacity_A, 10.0)
    assert jump_unconstrained > 9.9, (
        "Without constraints, jump should be ~full capacity"
    )
    assert jump_constrained < jump_unconstrained, "Constraint must reduce the jump"
    assert np.isclose(jump_constrained, expected_limit, atol=1e-6), (
        f"Jump {jump_constrained:.6f} should match limit {expected_limit:.6f}"
    )


def test_real_esm_tsa_interperiod_rampdown_enforcement():
    """Mirror test for ramp-down:
    - Unconstrained: drop across period boundary should be ~full capacity.
    - Constrained: drop is reduced and matches rampDownMax * dt * cap.
    """
    # --- Unconstrained case (no ramp limits) ---
    system_A = build_test_system(rampUp=None, rampDown=None)
    model_A, key_before, key_after = maximize_period_boundary_jump_down(system_A)

    # No inter-period ramp-down constraint should exist
    assert not hasattr(model_A, "ConstrInterPeriod_rampDownMax_conv")

    op_before_A = pyo.value(model_A.op_conv[key_before])
    op_after_A = pyo.value(model_A.op_conv[key_after])
    drop_unconstrained = op_before_A - op_after_A

    # --- Constrained case (rampDownMax enforced) ---
    ramp_down = 0.2  # per hour
    system_B = build_test_system(rampUp=ramp_down, rampDown=ramp_down)
    model_B, key_before, key_after = maximize_period_boundary_jump_down(system_B)

    # Inter-period ramp-down constraint must exist
    assert hasattr(model_B, "ConstrInterPeriod_rampDownMax_conv")

    cap_B = pyo.value(model_B.cap_conv["test", "Electrolyzer", 0])
    op_before_B = pyo.value(model_B.op_conv[key_before])
    op_after_B = pyo.value(model_B.op_conv[key_after])
    drop_constrained = op_before_B - op_after_B

    # Active inter-period ramp-down constraint at (ip=0, p=1, t=0)
    constraint_key = ("test", "Electrolyzer", 0, 1, 0)
    constraint = model_B.ConstrInterPeriod_rampDownMax_conv[constraint_key]

    # Read coefficient in front of capacity from linear repn
    repn = generate_standard_repn(constraint.body)

    cap_coefficient = None
    for var, coeff in zip(repn.linear_vars, repn.linear_coefs):
        if var is model_B.cap_conv["test", "Electrolyzer", 0]:
            cap_coefficient = coeff
            break

    assert cap_coefficient is not None, "Could not find capacity term in constraint."

    # Coefficient should be -ramp_down * dt → dt = |coef| / ramp_down
    timestep = abs(cap_coefficient) / ramp_down
    expected_limit = ramp_down * timestep * cap_B

    # Sanity checks
    assert np.isclose(cap_B, 10.0)
    assert np.isclose(timestep, system_B.hoursPerSegment[0][1, 0])

    # Behaviour checks:
    # Unconstrained: optimizer should use ~full drop (cap → 0)
    assert drop_unconstrained > 9.9, (
        "Without constraints, drop should be ~full capacity"
    )

    # Constraint must reduce the drop
    assert drop_constrained < drop_unconstrained, (
        "Ramp-down constraint must reduce the drop"
    )

    # And the drop should match the theoretical limit
    assert np.isclose(drop_constrained, expected_limit, atol=1e-6), (
        f"Drop {drop_constrained:.3f} should match limit {expected_limit:.3f}"
    )


# test edge case: no ramp limits lead to no interperiod constraint
def test_no_interperiod_constraint_if_no_ramp_limits():
    esM = build_test_system(rampUp=None, rampDown=None)
    esM.aggregateTemporally(numberOfTypicalPeriods=2)
    esM.optimize(esM, esM.pyM)
    pyM = esM.pyM

    assert not hasattr(pyM, "ConstrInterPeriod_rampUpMax_conv")
    assert not hasattr(pyM, "ConstrInterPeriod_rampDownMax_conv")
