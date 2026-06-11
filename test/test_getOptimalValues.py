import pytest

from fine.utils import ImplementedSolvers


@pytest.mark.parametrize(
    "name, expected_time_dependent, expected_has_values",
    [
        ("capacityVariablesOptimum", False, False),
        ("isBuiltVariablesOptimum", False, False),
        ("operationVariablesOptimum", True, True),
        ("commissioningVariablesOptimum", False, False),
        ("decommissioningVariablesOptimum", False, False),
    ],
)
def test_getOptimalValues_returns_requested_variable_from_esm(
    minimal_test_esM,
    name,
    expected_time_dependent,
    expected_has_values,
):
    """Test getOptimalValues for a requested optimum variable on a real optimized
    EnergySystemModel.

    The test uses the minimal_test_esM fixture from conftest.py, optimizes it,
    and calls getOptimalValues on the real component modeling class stored in
    esM.componentModelingDict.

    It verifies that getOptimalValues returns the expected dictionary structure,
    the correct time-dependency flag, the component model dimension, and real
    values for result categories that are active for the selected component.
    """
    esM = minimal_test_esM

    esM.optimize(
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    comp_name = "Electricity market"
    component_model = esM.componentModelingDict[esM.componentNames[comp_name]]

    result = component_model.getOptimalValues(name=name, ip=0)

    assert set(result.keys()) == {"values", "timeDependent", "dimension"}
    assert result["timeDependent"] is expected_time_dependent
    assert result["dimension"] == component_model.dimension

    if expected_has_values:
        assert result["values"] is not None
        assert not result["values"].empty
    else:
        assert result["values"] is None


@pytest.mark.parametrize("name", ["all", "unknownVariableName"])
def test_getOptimalValues_returns_all_variables_from_esm(minimal_test_esM, name):
    """Test getOptimalValues when all optimum variables are requested on a real
    optimized EnergySystemModel.

    If name is 'all' or not part of the supported variable mapping,
    getOptimalValues should return all supported result categories.
    """
    esM = minimal_test_esM

    esM.optimize(
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    comp_name = "Electricity market"
    component_model = esM.componentModelingDict[esM.componentNames[comp_name]]

    result = component_model.getOptimalValues(name=name, ip=0)

    expected_variable_names = {
        "capacityVariablesOptimum",
        "isBuiltVariablesOptimum",
        "operationVariablesOptimum",
        "commissioningVariablesOptimum",
        "decommissioningVariablesOptimum",
    }

    assert set(result.keys()) == expected_variable_names

    expected_time_dependent = {
        "capacityVariablesOptimum": False,
        "isBuiltVariablesOptimum": False,
        "operationVariablesOptimum": True,
        "commissioningVariablesOptimum": False,
        "decommissioningVariablesOptimum": False,
    }

    expected_has_values = {
        "capacityVariablesOptimum": False,
        "isBuiltVariablesOptimum": False,
        "operationVariablesOptimum": True,
        "commissioningVariablesOptimum": False,
        "decommissioningVariablesOptimum": False,
    }

    for variable_name in expected_variable_names:
        entry = result[variable_name]

        assert set(entry.keys()) == {"values", "timeDependent", "dimension"}
        assert entry["timeDependent"] is expected_time_dependent[variable_name]
        assert entry["dimension"] == component_model.dimension

        if expected_has_values[variable_name]:
            assert entry["values"] is not None
            assert not entry["values"].empty
        else:
            assert entry["values"] is None

