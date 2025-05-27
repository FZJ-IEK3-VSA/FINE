import fine as fn
import pandas as pd
import pytest

# Test material demand constraint in a simple model
# Here, material intensity and commissioning are multiplied and result in the operation rate of the automatically generated material sink (copper/steel demand)

def test_material_demand_constraint():

    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        onlycommodities={"electricity"},
        onlycommodityUnitsDict={"electricity": r"GW$_{el}$"},
        onlymaterials={"steel", "copper"},
        onlymaterialUnitsDict={"steel": r"tons", "copper": r"tons"}
    )

    # Add electricity source with material intensity
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind Turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            materialIntensity={
                'A': {
                    'steel': pd.Series({0: 3.1}),
                    'copper': pd.Series({0: 5.3})
                }
            }
        )
    )

    # Add electricity sink with fixed operation rate
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=50
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

    res_copper = esM.pyM.op_srcSnk["A", "Copper demand", 0, 0, 0]()
    res_steel = esM.pyM.op_srcSnk["A", "Steel demand", 0, 0, 0]()

    # Check whether the constraint performs the calculation correctly 
    assert res_copper == pytest.approx(expected_copper)
    assert res_steel == pytest.approx(expected_steel)
