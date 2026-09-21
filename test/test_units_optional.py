import builtins

import pytest

import fine as fn


def test_plain_model_without_pint(monkeypatch):
    original_import = builtins.__import__

    def without_pint(name, *args, **kwargs):
        if name == "pint" or name.startswith("pint."):
            raise ModuleNotFoundError("No module named 'pint'", name="pint")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pint)
    model = fn.EnergySystemModel(
        locations={"DE"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "MW"},
        numberOfTimeSteps=2,
    )
    component = fn.Source(model, "plain", "electricity", True, capacityMax=5)
    assert component.capacityMax == 5
    assert model.unitRegistry._ureg is None
    with pytest.raises(ImportError, match="optional 'pint' dependency"):
        model.unitRegistry.convert(5, "GW", "MW")
