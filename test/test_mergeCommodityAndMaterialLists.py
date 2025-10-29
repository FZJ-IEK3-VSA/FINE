import fine as fn
import pandas as pd

# Test if merge of the onlymaterials and onlycommodities list works correctly (same also for UnitsDict)


def test_commodity_and_material_merging():
    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={
            "electricity": r"GW$_{el}$",
            "hydrogen": r"GW$_{H_{2},LHV}$",
        },
        materials={"steel", "copper", "lithium"},
        materialUnitsDict={"steel": r"tons", "copper": r"tons", "lithium": r"tons"},
    )

    # Expected merged commodity list
    expected_commodities = {"electricity", "hydrogen", "copper", "lithium", "steel"}

    # Expected merged commodityUnitsDict list
    expected_units = {
        "electricity": r"GW$_{el}$",
        "hydrogen": r"GW$_{H_{2},LHV}$",
        "copper": r"tons",
        "lithium": r"tons",
        "steel": r"tons",
    }

    # Check whether the expected list matches the automatically generated list
    assert esM.commodities == expected_commodities
    assert esM.commodityUnitsDict == expected_units





def test_automatic_generation_scrap_materials():
    # Define minimal Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={
            "electricity": r"GW$_{el}$",
            "hydrogen": r"GW$_{H_{2},LHV}$",
        },
        materials={"steel", "copper", "lithium"},
        materialUnitsDict={"steel": r"tons", "copper": r"tons", "lithium": r"tons"},
        numberOfTimeSteps=8760,
        hoursPerTimeStep=1,
    )

    # Add components using different materials

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=5,
            materialIntensity={
                "A": {
                    "steel": pd.Series(
                        {0: 5.1}, dtype="float64"
                    ),
                    "copper": pd.Series(
                        {0: 5.1}, dtype="float64"
                    ),
                },
            },
            )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=5,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=5,
            materialIntensity={
                "A": {
                    "lithium": pd.Series(
                        {0: 5.1}, dtype="float64"
                    ),
                },
            },
        )
    )

    # Expected merged commodity list
    expected_commodities = {"electricity", "hydrogen", "copper", "lithium", "steel", "Batteries_lithium_scrap", "Wind_copper_scrap", "Wind_steel_scrap"}

    # Expected merged commodityUnitsDict list
    expected_units = {
        "electricity": r"GW$_{el}$",
        "hydrogen": r"GW$_{H_{2},LHV}$",
        "copper": r"tons",
        "lithium": r"tons",
        "steel": r"tons",
        "Batteries_lithium_scrap": r"tons",
        "Wind_copper_scrap": r"tons",
        "Wind_steel_scrap": r"tons"
    }

    # Check whether the expected list matches the automatically generated list
    assert esM.commodities == expected_commodities
    assert esM.commodityUnitsDict == expected_units
