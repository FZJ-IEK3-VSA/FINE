"""Compare the optimization models of test fixtures with stored references.

A refactoring that must not change the optimization model keeps these
references identical. The model is written in a canonical form instead of an
LP file, so the comparison does not depend on the Pyomo LP writer, the order of
Python sets or the last digits of floating point numbers. Set UPDATE_GOLDEN=1 to
regenerate the references intentionally.
"""

import difflib
import gzip
import math
import os
from itertools import islice
from pathlib import Path

import numpy as np
import pyomo.environ as pyomo
import pytest
from pyomo.repn import generate_standard_repn

LP_GOLDEN_DIR = Path(__file__).parent / "data" / "golden" / "lp"


def _number(x):
    """Format a number with 12 significant digits; None means unbounded."""
    if x is None or math.isinf(x):
        return "inf" if x is None or x > 0 else "-inf"
    return f"{x + 0.0:.12g}"  # + 0.0 turns -0.0 into 0.0


def _terms(expr):
    """Return the constant and the sorted linear and quadratic terms of expr."""
    repn = generate_standard_repn(expr, compute_values=True)
    assert repn.nonlinear_expr is None, f"Nonlinear expression: {expr}"
    coefficients = {}
    for var, coef in zip(repn.linear_vars, repn.linear_coefs):
        coefficients[var.name] = coefficients.get(var.name, 0) + coef
    for (var1, var2), coef in zip(repn.quadratic_vars, repn.quadratic_coefs):
        key = "*".join(sorted((var1.name, var2.name)))
        coefficients[key] = coefficients.get(key, 0) + coef
    terms = " ".join(
        f"{_number(coef)} {name}" for name, coef in sorted(coefficients.items())
    )
    return pyomo.value(repn.constant), terms


def _canonical_model(pyM):
    """Return the model as sorted lines: variables, constraints and objective."""
    lines = []
    for var in pyM.component_data_objects(pyomo.Var):
        domain = "B" if var.is_binary() else "I" if var.is_integer() else "C"
        if var.fixed:
            bounds = f"fixed {_number(var.value)}"
        else:
            bounds = f"{_number(var.lb)} {_number(var.ub)}"
        lines.append(f"var {var.name} {domain} {bounds}")
    for con in pyM.component_data_objects(pyomo.Constraint, active=True):
        constant, terms = _terms(con.body)
        lower = None if con.lower is None else pyomo.value(con.lower) - constant
        upper = None if con.upper is None else pyomo.value(con.upper) - constant
        lines.append(f"con {con.name}: {_number(lower)} <= {terms} <= {_number(upper)}")
    for obj in pyM.component_data_objects(pyomo.Objective, active=True):
        constant, terms = _terms(obj.expr)
        lines.append(f"obj {obj.name} {obj.sense}: {_number(constant)} + {terms}")
    return "\n".join(sorted(lines)) + "\n"


def _assert_model_snapshot(esM, reference_name, tmp_path):
    """Compare the canonical model of an esM with a stored reference.

    References ending in .gz are stored compressed.
    """
    esM.declareOptimizationProblem()
    actual = _canonical_model(esM.pyM)
    reference_path = LP_GOLDEN_DIR / reference_name
    compressed = reference_path.suffix == ".gz"

    if os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true", "yes"}:
        LP_GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        if compressed:
            # mtime=0 keeps the file identical when nothing changed.
            with gzip.GzipFile(reference_path, "wb", mtime=0) as f:
                f.write(actual.encode())
        else:
            reference_path.write_text(actual, newline="\n")
        return

    regenerate = (
        "Regenerate it intentionally with: UPDATE_GOLDEN=1 pytest "
        f"test/test_lpSnapshot.py -k {reference_name.split('.')[0]}"
    )
    assert reference_path.exists(), (
        f"Missing model reference {reference_path}. {regenerate}"
    )
    if compressed:
        with gzip.open(reference_path, "rt") as f:
            expected = f.read()
    else:
        expected = reference_path.read_text()

    if actual != expected:
        actual_path = tmp_path / reference_name.removesuffix(".gz")
        actual_path.write_text(actual, newline="\n")
        diff = difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            "reference",
            "actual",
            lineterm="",
        )
        pytest.fail(
            f"Model differs from {reference_path}.\n"
            f"Actual model: {actual_path}\n{regenerate}\n" + "\n".join(islice(diff, 40))
        )


@pytest.mark.parametrize(
    "fixture_name, reference_name",
    [
        ("minimal_test_esM", "minimal_test_esM.txt"),
        ("single_node_test_esM", "single_node_test_esM.txt"),
        ("dsm_test_esM", "dsm_test_esM.txt"),
        ("balanceLimitConstraint_test_esM", "balanceLimitConstraint_test_esM.txt.gz"),
        ("perfectForesight_test_esM", "perfectForesight_test_esM.txt"),
    ],
)
def test_lp_snapshot(fixture_name, reference_name, request, tmp_path):
    # balanceLimitConstraint_test_esM draws random input data.
    np.random.seed(0)
    esM = request.getfixturevalue(fixture_name)
    if isinstance(esM, tuple):
        esM = esM[0]
    _assert_model_snapshot(esM, reference_name, tmp_path)
