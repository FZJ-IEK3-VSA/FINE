import pytest

import fine as fn
from fine.utils import ImplementedSolvers


def build_test_system():
    """Create a minimal EnergySystemModel for testing getOptimalValues."""
    esM = fn.EnergySystemModel(
        locations={"test"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "GW"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="electricity_source",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=10.0,
            investPerCapacity=1.0,
            opexPerOperation=0.0,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="electricity_sink",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=1.0,
        )
    )

    return esM


@pytest.mark.parametrize(
    "name, expected_time_dependent, expected_has_values",
    [
        ("capacityVariablesOptimum", False, True),
        ("isBuiltVariablesOptimum", False, False),
        ("operationVariablesOptimum", True, True),
        ("commissioningVariablesOptimum", False, True),
        ("decommissioningVariablesOptimum", False, True),
    ],
)
def test_getOptimalValues_returns_requested_variable_from_esm(
    name,
    expected_time_dependent,
    expected_has_values,
):
    """Test getOptimalValues for a requested optimum variable on a real optimized
    EnergySystemModel.

    The test builds a minimal EnergySystemModel with one source and one sink,
    optimizes it, and calls getOptimalValues on the real component modeling
    class stored in esM.componentModelingDict.

    It verifies that getOptimalValues returns the expected dictionary structure,
    the correct time-dependency flag, the component model dimension, and real
    values for the result categories that are active in this minimal model.

    Some result categories, such as isBuilt, commissioning, and decommissioning
    variables, may be None because the corresponding model features are not
    active in this simple single-period test system.
    """
    esM = build_test_system()

    esM.optimize(
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    component_model = esM.componentModelingDict[
        esM.componentNames["electricity_source"]
    ]

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
def test_getOptimalValues_returns_all_variables_from_esm(name):
    """Test getOptimalValues when all optimum variables are requested on a real
    optimized EnergySystemModel.

    If name is 'all' or not part of the supported variable mapping,
    getOptimalValues should return all supported result categories.

    The test verifies that every expected result category is present, that each
    entry contains values, time-dependency information, and the component model
    dimension, and that active result categories contain real values.
    """
    esM = build_test_system()

    esM.optimize(
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    component_model = esM.componentModelingDict[
        esM.componentNames["electricity_source"]
    ]

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
        "capacityVariablesOptimum": True,
        "isBuiltVariablesOptimum": False,
        "operationVariablesOptimum": True,
        "commissioningVariablesOptimum": True,
        "decommissioningVariablesOptimum": True,
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
