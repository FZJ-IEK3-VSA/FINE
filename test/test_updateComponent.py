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
