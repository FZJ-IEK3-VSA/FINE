import fine as fn


def test_updateComponent(minimal_test_esM):
    _invest_before = minimal_test_esM.getComponentAttribute(
        componentName="Electrolyzers", attributeName="investPerCapacity"
    )
    _opex_before = minimal_test_esM.getComponentAttribute(
        componentName="Electrolyzers", attributeName="opexPerCapacity"
    )

    # double the invest and determine a economic lifetime, set both as updated values
    _new_invest = _invest_before * 2
    _new_capacityMax = 5
    minimal_test_esM.updateComponent(
        componentName="Electrolyzers",
        updateAttrs={"investPerCapacity": _new_invest, "capacityMax": _new_capacityMax},
    )

    # make sure the new value has ben set as invest
    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="Electrolyzers", attributeName="investPerCapacity"
        )
        == _new_invest
    )

    # make sure the new value has ben set as invest
    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="Electrolyzers", attributeName="capacityMax"
        )
        == _new_capacityMax
    )

    # other values must not have changed
    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="Electrolyzers", attributeName="opexPerCapacity"
        )
        == _opex_before
    )

    # Change name of component
    _new_name = "New Electrolyzer"
    minimal_test_esM.updateComponent(
        componentName="Electrolyzers", updateAttrs={"name": _new_name}
    )

    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="Electrolyzers", attributeName="name"
        )
        == "Electrolyzers"
    )

    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="New Electrolyzer", attributeName="name"
        )
        == _new_name
    )


def test_updateComponent_conversionDynamic():
    esM = fn.EnergySystemModel(
        locations={"region1"},
        numberOfTimeSteps=10,
        hoursPerTimeStep=1,
        commodities={"electricity", "heat"},
        commodityUnitsDict={"electricity": "kW", "heat": "kW"},
    )

    esM.add(
        fn.ConversionDynamic(
            esM=esM,
            name="Methane heater",
            physicalUnit="kW",
            capacityFix=1,
            commodityConversionFactors={"electricity": -1, "heat": 1},
        )
    )
    assert esM.getComponent("Methane heater").capacityFix == 1
    assert (
        esM.getComponent("Methane heater").processedCapacityFix[0].loc["region1"] == 1
    )

    esM.updateComponent("Methane heater", {"capacityFix": 2})
    assert esM.getComponent("Methane heater").capacityFix == 2
    assert (
        esM.getComponent("Methane heater").processedCapacityFix[0].loc["region1"] == 2
    )
