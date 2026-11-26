"""Check the componentLimit constraints against the specification.

Each case builds the componentLimit test model (see the fixture
componentLimit_esM_factory in conftest.py) with a single limit row, declares
the optimization problem and compares the coefficients of the limit constraint
with the ones the specification in
docs/user_guide/mathematical_descriptions/inter_component.md prescribes.
No model is solved.

The test model has two investment periods (internal indices 0 and 1, five
years apart) and two time steps of one hour without time series aggregation,
so every period occurs once per year.
"""

import warnings

import pandas as pd
import pyomo.environ as pyomo
import pytest
from pyomo.repn import generate_standard_repn

LOCATIONS = ["R1", "R2", "R3"]
YEARS_PER_PERIOD = 5
# share of a flow that arrives: 1 - losses (0.001 per km) * distance (100 km)
ARRIVING = 0.9


def _limitRow(typ, value=10.0, bound="upper", commodity=None, ip=2020, ipEnd=None):
    return pd.DataFrame(
        [[value, bound, typ, commodity, ip, ipEnd]],
        index=["L"],
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
    )


def _build(factory, row, members, locs=None, connections=None):
    """Build the test model with one limit "L" over locations or connections."""
    eligibility = eligibility2dim = None
    if connections is not None:
        eligibility2dim = pd.DataFrame(
            1, index=pd.MultiIndex.from_tuples(connections), columns=["L"]
        )
    else:
        eligibility = pd.DataFrame(0, index=LOCATIONS, columns=["L"])
        eligibility.loc[locs, "L"] = 1
    return factory(
        row,
        eligibility,
        eligibility2dim,
        members={name: ["L"] for name in members},
    )


def _limitConstraints(esM, ID="L"):
    """Return (lower, upper, coefficients) of every componentLimit constraint of ID."""
    constraints = []
    for con in esM.pyM.component_data_objects(pyomo.Constraint, active=True):
        name = con.parent_component().name
        if "omponentLimit" not in name and "omponentShareLimit" not in name:
            continue
        index = con.index()
        if (index[0] if isinstance(index, tuple) else index) != ID:
            continue
        repn = generate_standard_repn(con.body, compute_values=True)
        coefficients = {}
        for var, coef in zip(repn.linear_vars, repn.linear_coefs):
            coefficients[var.name] = coefficients.get(var.name, 0) + coef
        constant = pyomo.value(repn.constant)
        lower = None if con.lower is None else pyomo.value(con.lower) - constant
        upper = None if con.upper is None else pyomo.value(con.upper) - constant
        constraints.append((lower, upper, coefficients))
    return constraints


def _declare(esM):
    esM.declareOptimizationProblem(timeSeriesAggregation=False)
    return esM


def _cap(cls, loc, comp, ip=0, coef=1.0, var="cap"):
    return {f"{var}_{cls}[{loc},{comp},{ip}]": coef}


def _op(var, loc, comp, ip=0, coef=1.0):
    """Return the operation variable of both time steps of one investment period."""
    return {f"{var}[{loc},{comp},{ip},0,{t}]": coef for t in (0, 1)}


def _merge(*dicts):
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


# (case, row arguments, members, where, expected coefficients)
# "where" is ("locs", [...]) or ("connections", [...]).
CASES = [
    # --- capacity ---
    (
        "capacity_skips_ineligible_location",
        dict(typ="capacity"),
        ["Src"],
        ("locs", ["R1", "R3"]),
        _cap("srcSnk", "R1", "Src"),
    ),
    (
        "capacity_conversion",
        dict(typ="capacity"),
        ["Conv"],
        ("locs", ["R1", "R2"]),
        _merge(_cap("conv", "R1", "Conv"), _cap("conv", "R2", "Conv")),
    ),
    (
        "capacity_storage",
        dict(typ="capacity", ip=2025),
        ["Stor"],
        ("locs", ["R2"]),
        _cap("stor", "R2", "Stor", ip=1),
    ),
    (
        "capacity_line_on_one_connection",
        dict(typ="capacity"),
        ["Line"],
        ("connections", [("R1", "R2")]),
        _merge(
            _cap("trans", "R1_R2", "Line", coef=0.5),
            _cap("trans", "R2_R1", "Line", coef=0.5),
        ),
    ),
    (
        "capacity_line_listed_in_both_directions_counts_once",
        dict(typ="capacity"),
        ["Line"],
        ("connections", [("R1", "R2"), ("R2", "R1")]),
        _merge(
            _cap("trans", "R1_R2", "Line", coef=0.5),
            _cap("trans", "R2_R1", "Line", coef=0.5),
        ),
    ),
    (
        "capacity_line_over_locations",
        dict(typ="capacity"),
        ["Line"],
        ("locs", ["R1", "R2"]),
        _merge(
            # both ends inside: the line counts fully
            _cap("trans", "R1_R2", "Line", coef=0.5),
            _cap("trans", "R2_R1", "Line", coef=0.5),
            # one end inside: the line counts half
            _cap("trans", "R1_R3", "Line", coef=0.25),
            _cap("trans", "R3_R1", "Line", coef=0.25),
            _cap("trans", "R2_R3", "Line", coef=0.25),
            _cap("trans", "R3_R2", "Line", coef=0.25),
        ),
    ),
    (
        "capacity_commodity_filter",
        dict(typ="capacity", commodity="gas"),
        ["Src", "Conv"],
        ("locs", LOCATIONS),
        _merge(*(_cap("conv", loc, "Conv") for loc in LOCATIONS)),
    ),
    # --- commissioning ---
    (
        "commissioning_range_is_not_scaled",
        dict(typ="commissioning", ip=2020, ipEnd=2025),
        ["Src"],
        ("locs", ["R1"]),
        _merge(
            _cap("srcSnk", "R1", "Src", ip=0, var="commis"),
            _cap("srcSnk", "R1", "Src", ip=1, var="commis"),
        ),
    ),
    (
        "commissioning_line_on_one_connection",
        dict(typ="commissioning"),
        ["Line"],
        ("connections", [("R1", "R2")]),
        _merge(
            _cap("trans", "R1_R2", "Line", coef=0.5, var="commis"),
            _cap("trans", "R2_R1", "Line", coef=0.5, var="commis"),
        ),
    ),
    # --- operation ---
    (
        "operation_source",
        dict(typ="operation"),
        ["Src"],
        ("locs", ["R1"]),
        _op("op_srcSnk", "R1", "Src"),
    ),
    (
        "operation_sink_counts_positive",
        dict(typ="operation"),
        ["Snk"],
        ("locs", ["R1"]),
        _op("op_srcSnk", "R1", "Snk"),
    ),
    (
        "operation_range_counts_years",
        dict(typ="operation", ip=2020, ipEnd=2025),
        ["Src"],
        ("locs", ["R1"]),
        _merge(
            _op("op_srcSnk", "R1", "Src", ip=0, coef=YEARS_PER_PERIOD),
            _op("op_srcSnk", "R1", "Src", ip=1, coef=YEARS_PER_PERIOD),
        ),
    ),
    (
        "operation_conversion_in_physical_unit",
        dict(typ="operation"),
        ["Conv"],
        ("locs", ["R1"]),
        _op("op_conv", "R1", "Conv"),
    ),
    (
        "operation_conversion_in_input_commodity_is_positive",
        dict(typ="operation", commodity="gas"),
        ["Conv"],
        ("locs", ["R1"]),
        _op("op_conv", "R1", "Conv", coef=2.0),
    ),
    (
        "operation_storage_counts_discharge",
        dict(typ="operation"),
        ["Stor"],
        ("locs", ["R1"]),
        _op("dischargeOp_stor", "R1", "Stor"),
    ),
    (
        "operation_line_on_directed_connection",
        dict(typ="operation"),
        ["Line"],
        ("connections", [("R1", "R2")]),
        _op("op_trans", "R1_R2", "Line"),
    ),
    (
        "operation_commodity_filter",
        dict(typ="operation", commodity="gas"),
        ["Src", "Gas"],
        ("locs", ["R1"]),
        _op("op_srcSnk", "R1", "Gas"),
    ),
    # --- balance ---
    (
        "balance_source_and_sink",
        dict(typ="balance", commodity="electricity"),
        ["Src", "Snk"],
        ("locs", ["R1"]),
        _merge(_op("op_srcSnk", "R1", "Src"), _op("op_srcSnk", "R1", "Snk", coef=-1)),
    ),
    (
        "balance_conversion_input",
        dict(typ="balance", commodity="gas"),
        ["Conv"],
        ("locs", ["R1"]),
        _op("op_conv", "R1", "Conv", coef=-2.0),
    ),
    (
        "balance_conversion_output",
        dict(typ="balance", commodity="electricity"),
        ["Conv"],
        ("locs", ["R1"]),
        _op("op_conv", "R1", "Conv"),
    ),
    (
        "balance_storage",
        dict(typ="balance", commodity="electricity"),
        ["Stor"],
        ("locs", ["R1"]),
        _merge(
            _op("dischargeOp_stor", "R1", "Stor"),
            _op("chargeOp_stor", "R1", "Stor", coef=-1),
        ),
    ),
    (
        "balance_line_into_one_location",
        dict(typ="balance", commodity="electricity"),
        ["Line"],
        ("locs", ["R1"]),
        _merge(
            _op("op_trans", "R2_R1", "Line", coef=ARRIVING),
            _op("op_trans", "R3_R1", "Line", coef=ARRIVING),
            _op("op_trans", "R1_R2", "Line", coef=-1),
            _op("op_trans", "R1_R3", "Line", coef=-1),
        ),
    ),
    (
        "balance_line_counts_only_the_group_boundary",
        dict(typ="balance", commodity="electricity"),
        ["Line"],
        ("locs", ["R1", "R2"]),
        _merge(
            _op("op_trans", "R3_R1", "Line", coef=ARRIVING),
            _op("op_trans", "R3_R2", "Line", coef=ARRIVING),
            _op("op_trans", "R1_R3", "Line", coef=-1),
            _op("op_trans", "R2_R3", "Line", coef=-1),
        ),
    ),
    (
        "balance_without_commodity_counts_own_commodity",
        dict(typ="balance"),
        ["Src", "Snk", "Stor"],
        ("locs", ["R1"]),
        _merge(
            _op("op_srcSnk", "R1", "Src"),
            _op("op_srcSnk", "R1", "Snk", coef=-1),
            _op("dischargeOp_stor", "R1", "Stor"),
            _op("chargeOp_stor", "R1", "Stor", coef=-1),
        ),
    ),
    (
        "balance_range_counts_years",
        dict(typ="balance", commodity="electricity", ip=2020, ipEnd=2025),
        ["Src"],
        ("locs", ["R1"]),
        _merge(
            _op("op_srcSnk", "R1", "Src", ip=0, coef=YEARS_PER_PERIOD),
            _op("op_srcSnk", "R1", "Src", ip=1, coef=YEARS_PER_PERIOD),
        ),
    ),
    # --- no contributing variable: an upper bound builds no constraint ---
    (
        "no_constraint_for_component_without_capacity_variable",
        dict(typ="capacity"),
        ["Snk"],
        ("locs", ["R1"]),
        None,
    ),
    (
        "no_constraint_for_line_balance_over_all_locations",
        dict(typ="balance", commodity="electricity"),
        ["Line"],
        ("locs", LOCATIONS),
        None,
    ),
]


@pytest.mark.parametrize(
    "rowArgs, members, where, expected",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_componentLimit_coefficients(
    componentLimit_esM_factory, rowArgs, members, where, expected
):
    kind, entries = where
    locs, connections = (entries, None) if kind == "locs" else (None, entries)
    esM = _declare(
        _build(
            componentLimit_esM_factory,
            _limitRow(**rowArgs),
            members,
            locs=locs,
            connections=connections,
        )
    )
    constraints = _limitConstraints(esM)
    if expected is None:
        assert constraints == []
        return
    assert len(constraints) == 1
    _, _, coefficients = constraints[0]
    assert coefficients.keys() == expected.keys()
    for name, coef in expected.items():
        assert coefficients[name] == pytest.approx(coef), name


@pytest.mark.parametrize(
    "bound, expectedBounds",
    [("lower", (10.0, None)), ("upper", (None, 10.0)), ("fixed", (10.0, 10.0))],
)
def test_componentLimit_bounds(componentLimit_esM_factory, bound, expectedBounds):
    esM = _declare(
        _build(
            componentLimit_esM_factory,
            _limitRow("capacity", bound=bound),
            ["Src"],
            locs=["R1"],
        )
    )
    ((lower, upper, _),) = _limitConstraints(esM)
    assert (lower, upper) == expectedBounds


def test_componentLimit_builds_one_constraint_per_row(componentLimit_esM_factory):
    """Rows of one ID are separate constraints, even if they are identical."""
    rows = pd.concat([_limitRow("capacity"), _limitRow("capacity")])
    esM = _declare(_build(componentLimit_esM_factory, rows, ["Src"], locs=["R1"]))
    assert len(_limitConstraints(esM)) == 2


@pytest.mark.parametrize(
    "rowArgs, members, where",
    [
        (dict(typ="operation"), ["Line"], ("locs", ["R1"])),
        (dict(typ="balance"), ["Conv"], ("locs", ["R1"])),
        (dict(typ="capacity", bound="lower"), ["Snk"], ("locs", ["R1"])),
        (dict(typ="capacity", bound="fixed"), ["Snk"], ("locs", ["R1"])),
    ],
    ids=[
        "operation_of_line_over_locations",
        "balance_without_commodity_with_conversion",
        "lower_bound_without_variable",
        "fixed_bound_without_variable",
    ],
)
def test_componentLimit_rejected_when_declared(
    componentLimit_esM_factory, rowArgs, members, where
):
    esM = _build(
        componentLimit_esM_factory, _limitRow(**rowArgs), members, locs=where[1]
    )
    with pytest.raises(ValueError):
        _declare(esM)


@pytest.mark.parametrize(
    "rowArgs",
    [
        dict(typ="capacity", bound="lower", value=-1.0),
        dict(typ="capacity", bound="fixed", value=0.0),
    ],
    ids=["lower_bound_below_zero", "fixed_zero"],
)
def test_componentLimit_without_variable_satisfied_by_zero(
    componentLimit_esM_factory, rowArgs
):
    """A bound that 0 satisfies builds no constraint when nothing contributes."""
    esM = _declare(
        _build(componentLimit_esM_factory, _limitRow(**rowArgs), ["Snk"], locs=["R1"])
    )
    assert _limitConstraints(esM) == []


def test_componentLimit_without_member_warns(componentLimit_esM_factory):
    esM = _build(componentLimit_esM_factory, _limitRow("capacity"), [], locs=["R1"])
    with pytest.warns(UserWarning, match="no component"):
        _declare(esM)
    assert _limitConstraints(esM) == []


@pytest.mark.parametrize(
    "rowArgs, where, message",
    [
        (
            dict(typ="balance", commodity="electricity"),
            ("connections", [("R1", "R2")]),
            "balance",
        ),
        (dict(typ="capacity", commodity="hydrogen"), ("locs", ["R1"]), "hydrogen"),
    ],
    ids=["balance_over_connections", "unknown_commodity"],
)
def test_componentLimit_rejected_when_created(
    componentLimit_esM_factory, rowArgs, where, message
):
    kind, entries = where
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=message):
            _build(
                componentLimit_esM_factory,
                _limitRow(**rowArgs),
                ["Line"],
                locs=entries if kind == "locs" else None,
                connections=entries if kind == "connections" else None,
            )


def _shareConstraints(esM):
    """Return {group: (lower, upper, coefficients)} of the share constraints of "L"."""
    constraints = {}
    for con in esM.pyM.componentShareLimitConstraint.values():
        repn = generate_standard_repn(con.body, compute_values=True)
        coefficients = {
            var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)
        }
        constant = pyomo.value(repn.constant)
        lower = None if con.lower is None else pyomo.value(con.lower) - constant
        upper = None if con.upper is None else pyomo.value(con.upper) - constant
        constraints[con.index()[2]] = (lower, upper, coefficients)
    return constraints


def _normalised(lower, upper, coefficients):
    """Scale a one-sided constraint to the form  terms <= 0."""
    sign = 1 if lower is None else -1
    bound = upper if lower is None else lower
    assert bound == pytest.approx(0)
    return {name: pytest.approx(sign * coef) for name, coef in coefficients.items()}


def test_componentShareLimit_per_location(componentLimit_esM_factory):
    """shareMax: each location holds at most the share of the eligible total."""
    esM = _declare(
        _build(
            componentLimit_esM_factory,
            _limitRow("capacity", value=0.4, bound="shareMax"),
            ["Src"],
            locs=["R1", "R2"],
        )
    )
    constraints = _shareConstraints(esM)
    assert set(constraints) == {"R1", "R2"}
    # cap[R1] <= 0.4 * (cap[R1] + cap[R2])
    assert _normalised(*constraints["R1"]) == {
        "cap_srcSnk[R1,Src,0]": 0.6,
        "cap_srcSnk[R2,Src,0]": -0.4,
    }


def test_componentShareLimit_grouped(componentLimit_esM_factory):
    """With a grouping, the share bounds each group instead of each location."""
    grouping = pd.DataFrame({"L": ["A", "A", "B"]}, index=LOCATIONS)
    esM = _declare(
        componentLimit_esM_factory(
            _limitRow("capacity", value=0.7, bound="shareMax"),
            pd.DataFrame(1, index=LOCATIONS, columns=["L"]),
            members={"Conv": ["L"]},
            grouping=grouping,
        )
    )
    constraints = _shareConstraints(esM)
    assert set(constraints) == {"A", "B"}
    # cap[R1] + cap[R2] <= 0.7 * (cap[R1] + cap[R2] + cap[R3])
    assert _normalised(*constraints["A"]) == {
        "cap_conv[R1,Conv,0]": 0.3,
        "cap_conv[R2,Conv,0]": 0.3,
        "cap_conv[R3,Conv,0]": -0.7,
    }


def test_componentShareMin_group_without_variable(componentLimit_esM_factory):
    """A shareMin group that cannot contribute forces the total to 0."""
    esM = _declare(
        _build(
            componentLimit_esM_factory,
            _limitRow("capacity", value=0.3, bound="shareMin"),
            ["Src"],
            # Src is not eligible in R3
            locs=["R1", "R2", "R3"],
        )
    )
    constraints = _shareConstraints(esM)
    # 0 >= 0.3 * (cap[R1] + cap[R2])
    assert _normalised(*constraints["R3"]) == {
        "cap_srcSnk[R1,Src,0]": 0.3,
        "cap_srcSnk[R2,Src,0]": 0.3,
    }
