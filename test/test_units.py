import pytest
import pandas as pd
import pint

import fine as fn


pytest.importorskip("pint")


@pytest.fixture
def ureg():
    r = pint.UnitRegistry()
    # Define a simple currency dimension for tests
    try:
        r.define("EUR = [currency]")
    except Exception:
        # ignore if already defined
        pass
    return r


@pytest.fixture
def esm():
    return fn.EnergySystemModel(
        locations={"DE"},
        commodities={"electricity", "water"},
        commodityUnitsDict={"electricity": "MW", "water": "m**3/h"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
    )


def test_plain_float_is_unchanged(esm):
    comp = fn.Source(
        esM=esm,
        name="src",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityMax=5,
    )
    assert comp.capacityMax == 5


def test_capacity_conversion(esm, ureg):
    comp = fn.Source(
        esM=esm,
        name="src2",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityMax=5,
        units={"capacityMax": ureg.GW},
    )
    # capacityMax stored on component (preprocessed may be in processedCapacityMax)
    assert pytest.approx(5000, rel=1e-6) == comp.capacityMax


def test_series_conversion(esm, ureg):
    capacity = pd.Series({"DE": 2.0})
    comp = fn.Source(
        esM=esm,
        name="src3",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityMax=capacity,
        units={"capacityMax": ureg.GW},
    )
    assert comp.processedCapacityMax[0]["DE"] == pytest.approx(2000)


def test_investment_cost_conversion(esm, ureg):
    comp = fn.Source(
        esM=esm,
        name="src4",
        commodity="electricity",
        hasCapacityVariable=True,
        investPerCapacity=800,
        units={"investPerCapacity": ureg.EUR / ureg.kW},
    )
    assert comp.processedInvestPerCapacity[0]["DE"] == pytest.approx(800_000)


def test_incompatible_unit_raises(esm, ureg):
    with pytest.raises(Exception):
        fn.Source(
            esM=esm,
            name="bad",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=5,
            units={"capacityMax": ureg.km},
        )


def test_no_pint_quantity_leakage(esm, ureg):
    # pint imported at module level for linting

    comp = fn.Source(
        esM=esm,
        name="nod",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityMax=5,
        investPerCapacity=800,
        units={
            "capacityMax": ureg.GW,
            "investPerCapacity": ureg.EUR / ureg.kW,
        },
    )
    # Ensure public attributes are plain numbers or pandas objects, not pint.Quantity
    assert not isinstance(comp.capacityMax, pint.Quantity)
    assert not isinstance(comp.investPerCapacity, pint.Quantity)


def test_storage_and_transmission_conversions(esm, ureg):
    # storage: capacity in GWh -> MWh
    st = fn.Storage(
        esM=esm,
        name="stor",
        commodity="electricity",
        hasCapacityVariable=True,
        capacityMax=1,
        units={"capacityMax": ureg.GWh},
    )
    # 1 GWh -> 1000 MWh (since esM commodityUnit is MW * hour steps)
    assert st.capacityMax == pytest.approx(1000)

    # distance conversion via unit registry (DataFrame) miles -> km
    miles = pd.DataFrame([[100.0]])
    converted = esm.unitRegistry.convert(miles, ureg.mile, esm.lengthUnit, "distances")
    val = float(converted.values.flatten()[0])
    assert val == pytest.approx(160.934, rel=1e-3)
