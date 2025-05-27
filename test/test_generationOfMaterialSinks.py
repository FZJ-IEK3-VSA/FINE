import fine as fn

# Test if automatic generation of material sinks works correctly

def test_generation_material_sinks():
    
    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        onlycommodities={"electricity"},
        onlycommodityUnitsDict={"electricity": r"GW$_{el}$"},
        onlymaterials={"steel", "copper", "lithium"},
        onlymaterialUnitsDict={"steel": r"tons", "copper": r"tons", "lithium": r"tons"},
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
    all_sink_names = [comp.name for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()]
    
    assert "Steel demand" in all_sink_names
    assert "Copper demand" in all_sink_names 
    assert "Lithium demand" in all_sink_names 

    # Test whether properties of the sink are correct e.g. copper sink
    copper_sink = next(c for c in esM.componentModelingDict["SourceSinkModel"].componentsDict.values() if c.name == "Copper demand")
    assert copper_sink.commodity == "copper"
    assert copper_sink.material is True


# Test if automatic generation of secondary material sources works correctly

def test_generation_secondary_material_sources():
    
    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        onlycommodities={"electricity"},
        onlycommodityUnitsDict={"electricity": r"GW$_{el}$"},
        onlymaterials={"steel", "copper", "lithium"},
        onlymaterialUnitsDict={"steel": r"tons", "copper": r"tons", "lithium": r"tons"},
    )


    # Add material source for steel
    steel_recovery = fn.Source(
        esM=esM,
        name="Steel recovery",
        commodity="steel",
        hasCapacityVariable=False,
        material=True,
    )
    esM.add(steel_recovery)

    # Generate missing material sources automatically
    esM.generationSecondaryMaterialSources()

    # Test whether both sources are now present in all_source_names after generation of missing sources
    all_source_names = [comp.name for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()]
    
    assert "Steel recovery" in all_source_names
    assert "Copper recovery" in all_source_names 
    assert "Lithium recovery" in all_source_names 

    # Test whether properties of the sources are correct e.g. copper source
    copper_recovery = next(c for c in esM.componentModelingDict["SourceSinkModel"].componentsDict.values() if c.name == "Copper recovery")
    assert copper_recovery.commodity == "copper"
    assert copper_recovery.material is True