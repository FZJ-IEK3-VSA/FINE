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
    esM.add(fn.Source(
        esM=esM, name="grid", commodity="electricity",
        hasCapacityVariable=False, operationRateMax=10.0, opexPerOperation=0.0
    ))

    esM.add(fn.Sink(
        esM=esM, name="H2_sink", commodity="hydrogen",
        hasCapacityVariable=False, opexPerOperation=0.0
    ))

    # Conversion component with optional ramp limits
    esM.add(fn.Conversion(
        esM=esM, name="Electrolyzer", physicalUnit="GW",
        commodityConversionFactors={"electricity": -1, "hydrogen": 1},
        hasCapacityVariable=True, capacityMax=10.0,
        investPerCapacity=0.0, opexPerCapacity=0.0,
        rampUpMax=rampUp, rampDownMax=rampDown,
    ))
    return esM


def maximize_period_boundary_jump(esM):
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
        expr=pyM.op_conv[key_after] - pyM.op_conv[key_before],
        sense=pyo.maximize
    )

    pyo.SolverFactory("glpk").solve(pyM)
    return pyM, key_before, key_after


def test_real_esm_tsa_interperiod_ramp_enforcement():
    """Verify that inter-period ramping limits the operation jump correctly."""

    # --- Unconstrained case (no ramping limits) ---
    system_A = build_test_system(rampUp=None, rampDown=None)
    model_A, key_before, key_after = maximize_period_boundary_jump(system_A)

    capacity = pyo.value(model_A.cap_conv["test", "Electrolyzer", 0])
    op_before_A = pyo.value(model_A.op_conv[key_before])
    op_after_A = pyo.value(model_A.op_conv[key_after])
    jump_unconstrained = op_after_A - op_before_A

    print("\n--- UNCONSTRAINED ---")
    print(f"Capacity: {capacity:.4f} GW")
    print(f"op_before: {op_before_A:.6f}")
    print(f"op_after:  {op_after_A:.6f}")
    print(f"Jump (unconstrained): {jump_unconstrained:.6f}")


    system_B = build_test_system(rampUp=0.2, rampDown=0.2)
    model_B, key_before, key_after = maximize_period_boundary_jump(system_B)

    op_before_B = pyo.value(model_B.op_conv[key_before])
    op_after_B = pyo.value(model_B.op_conv[key_after])
    jump_constrained = op_after_B - op_before_B

    print("\n--- CONSTRAINED ---")
    print(f"op_before: {op_before_B:.6f}")
    print(f"op_after:  {op_after_B:.6f}")
    print(f"Jump (constrained): {jump_constrained:.6f}")


    constraint_key = ("test", "Electrolyzer", 0, 1, 0)
    print("DEBUG: num periods =", getattr(system_B, "numberOfTypicalPeriods", "MISSING"))


    constraint = model_B.ConstrInterPeriod_rampUpMax_conv[constraint_key]
    repn = generate_standard_repn(constraint.body)

    cap_coefficient = None
    for var, coeff in zip(repn.linear_vars, repn.linear_coefs):
        if var is model_B.cap_conv["test", "Electrolyzer", 0]:
            cap_coefficient = coeff
            break

    timestep = abs(cap_coefficient) / 0.2  # rampUpMax = 0.2
    expected_limit = 0.2 * timestep * capacity

    print("\n--- CONSTRAINT ANALYSIS ---")
    print(f"Constraint expression: {constraint.body}")
    print(f"Coefficient on capacity: {cap_coefficient}")
    print(f"Derived timestep: {timestep}")
    print(f"Expected ramp limit: {expected_limit:.6f}")
    print(f"Actual jump: {jump_constrained:.6f}")
    print(f"Difference: {abs(jump_constrained - expected_limit):.6e}")


    assert np.isclose(capacity, 10.0)
    assert jump_unconstrained > 9.9, "Without constraints, jump should be ~full capacity"
    assert jump_constrained < jump_unconstrained, "Constraint must reduce the jump"
    assert np.isclose(jump_constrained, expected_limit, atol=1e-6), \
        f"Jump {jump_constrained:.3f} should match limit {expected_limit:.3f}"




