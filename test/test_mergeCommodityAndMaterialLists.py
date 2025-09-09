import fine as fn

# Test if merge of the onlymaterials and onlycommodities list works correctly (same also for UnitsDict)

def test_commodity_and_material_merging():

    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={"electricity": r"GW$_{el}$","hydrogen": r"GW$_{H_{2},LHV}$"},
        materials={"steel", "copper", "lithium"},
        materialUnitsDict={"steel": r"tons", "copper": r"tons", "lithium": r"tons"},
    )

    # Expected merged commodity list 
    expected_commodities = ["electricity", "hydrogen", "copper", "lithium", "steel"]

    # Expected merged commodityUnitsDict list
    expected_units = {
        "electricity": r"GW$_{el}$",
        "hydrogen": r"GW$_{H_{2},LHV}$",
        "copper": r"tons",
        "lithium": r"tons",
        "steel": r"tons"
    }

    # Check whether the expected list matches the automatically generated list 
    assert set(esM.commodities) == set(expected_commodities)
    assert esM.commodityUnitsDict == expected_units
