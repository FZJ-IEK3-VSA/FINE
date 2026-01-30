import fine as fn
import pandas as pd
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


# Test if automatic generation of secondary material sources and sinks works correctly


def test_generation_material_sources():
    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"GW$_{el}$"},
        materials={"steel", "copper", "aluminum"},
        materialUnitsDict={"steel": r"tons", "copper": r"tons", "aluminum": r"tons"},
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            materialIntensity={
                0: {
                    "steel": pd.Series({"A": 3.1}),
                    "copper": pd.Series({"A": 5.3}),
                    "aluminum": pd.Series({"A": 2.3}),
                }
            },
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind_aluminum_scrap",
            commodity="Wind_aluminum_scrap",
            hasCapacityVariable=False,
            material=True,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Wind_aluminum_scrap_rec",
            commodity="Wind_aluminum_scrap",
            hasCapacityVariable=False,
        )
    )

    # Generate missing material sinks automatically
    esM.generationSecondaryMaterialSources()

    # Collect all source component names in the SourceSinkModel
    scrap_sources = [
        comp.name
        for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()
        if comp.__class__.__name__ == "Source" and comp.commodity.endswith("_scrap")
    ]

    scrap_sinks = [
        comp.name
        for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()
        if comp.__class__.__name__ == "Sink" and comp.commodity.endswith("_scrap")
    ]

    # Test that both scrap sources were generated
    assert "Wind_steel_scrap" in scrap_sources
    assert "Wind_copper_scrap" in scrap_sources
    assert "Wind_aluminum_scrap" in scrap_sources

    # Inspect the copper scrap source to confirm correct properties
    copper_scrap_source = next(
        comp
        for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()
        if comp.name == "Wind_copper_scrap"
    )

    assert copper_scrap_source.commodity == "Wind_copper_scrap"
    assert copper_scrap_source.hasCapacityVariable is False
    assert copper_scrap_source.material is True

    # Test that both scrap sources were generated
    assert "Wind_steel_scrap_rec" in scrap_sinks
    assert "Wind_copper_scrap_rec" in scrap_sinks
    assert "Wind_aluminum_scrap_rec" in scrap_sinks

    # Inspect the copper scrap source to confirm correct properties
    copper_scrap_sink = next(
        comp
        for comp in esM.componentModelingDict["SourceSinkModel"].componentsDict.values()
        if comp.name == "Wind_copper_scrap_rec"
    )

    assert copper_scrap_sink.commodity == "Wind_copper_scrap"
    assert copper_scrap_sink.hasCapacityVariable is False
    assert copper_scrap_sink.material is False
