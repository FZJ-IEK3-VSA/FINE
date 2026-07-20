import fine as fn
import pandas as pd
import pytest

# Test material demand constraint in a simple model


def test_material_demand_constraint():
    # Define the Energy System Model

    initial_material_cost = {
        "steel": pd.Series(
            {
                "A": 0.1,
            }
        ),
        "copper": pd.Series(
            {
                "A": 0.1,
            }
        ),
    }

    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"GW$_{el}$"},
        materials={"steel", "copper"},
        materialUnitsDict={"steel": r"tons", "copper": r"tons"},
        initialMaterialCost=initial_material_cost,
    )

    # Add electricity source with material intensity
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind Turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            materialIntensity={
                0: {"steel": pd.Series({"A": 3.1}), "copper": pd.Series({"A": 5.3})}
            },
        )
    )

    # Add electricity sink with fixed operation rate
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=50,
        )
    )

    # Add material sources for all materials
    esM.add(
        fn.Source(
            esM=esM,
            name="Steel supply",
            hasCapacityVariable=False,
            commodity="steel",
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Copper supply",
            hasCapacityVariable=False,
            commodity="copper",
        )
    )

    # Generate material sinks for all materials
    esM.generationMaterialSinks()

    esM.optimize(solver="glpk")

    # Expected values for operation rate of the material sink = material intensity * commissioning
    expected_copper = 50 * 5.3
    expected_steel = 50 * 3.1

    result_copper = esM.pyM.initialMaterialSupply["A", "copper"].value
    result_steel = esM.pyM.initialMaterialSupply["A", "steel"].value

    assert result_copper == pytest.approx(expected_copper)
    assert result_steel == pytest.approx(expected_steel)
