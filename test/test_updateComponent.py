import FINE as fn
import numpy as np
import pytest


def test_updateComponent(minimal_test_esM):
    _invest_before = minimal_test_esM.getComponentAttribute(
        componentName="Electrolyzers", attributeName="investPerCapacity"
    )
    _opex_before = minimal_test_esM.getComponentAttribute(
        componentName="Electrolyzers", attributeName="opexPerCapacity"
    )

    # double the invest and set as updated values
    _new_invest = _invest_before * 2
    minimal_test_esM.updateComponent(
        componentName="Electrolyzers", updateAttrs={"investPerCapacity": _new_invest}
    )

    # make sure the new value has ben set as invest
    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="Electrolyzers", attributeName="investPerCapacity"
        )
        == _new_invest
    )
    # other values must not have changed
    assert (
        minimal_test_esM.getComponentAttribute(
            componentName="Electrolyzers", attributeName="opexPerCapacity"
        )
        == _opex_before
    )
