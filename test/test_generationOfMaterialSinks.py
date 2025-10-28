import fine as fn

# Test if automatic generation of material sinks works correctly


def test_generation_material_sinks():
    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"GW$_{el}$"},
        materials={"steel", "copper", "lithium"},
        materialUnitsDict={"steel": r"tons", "copper": r"tons", "lithium": r"tons"},
    )

    # Add material sink for steel
    steel_sink = fn.Sink(
        esM=esM,
        name="Steel demand",
        commodity="steel",
        hasCapacityVariable=False,
        material=True,
    )
    esM.add(steel_sink)

    # Generate missing material sinks automatically
    esM.generationMaterialSinks()

    # Test whether both sinks are now present in all_sink_names after generation of missing sinks
    all_sink_names = [
        comp.name
        for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()
    ]

    assert "Steel demand" in all_sink_names
    assert "Copper demand" in all_sink_names
    assert "Lithium demand" in all_sink_names

    # Test whether properties of the sink are correct e.g. copper sink
    copper_sink = next(
        c
        for c in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()
        if c.name == "Copper demand"
    )
    assert copper_sink.commodity == "copper"
    assert copper_sink.material is True
