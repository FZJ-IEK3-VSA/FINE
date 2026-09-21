import pytest
import pandas as pd

import fine as fn


pint = pytest.importorskip("pint")


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


@pytest.mark.parametrize("cls", [fn.Source, fn.Sink, fn.Conversion])
@pytest.mark.parametrize("has_capacity", [True, False])
def test_operation_rates(cls, has_capacity, esm, ureg):
    kwargs = (
        {"physicalUnit": "MW", "commodityConversionFactors": {"electricity": 1}}
        if cls is fn.Conversion
        else {"commodity": "electricity"}
    )
    data = pd.DataFrame({"DE": [0.2, 0.4]})
    comp = cls(
        esm,
        "rates",
        hasCapacityVariable=has_capacity,
        operationRateFix=data,
        units={"operationRateFix": ureg.percent if has_capacity else ureg.GW},
        **kwargs,
    )
    pd.testing.assert_frame_equal(
        comp.operationRateFix, data * (0.01 if has_capacity else 1000)
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "commodityCost",
        "commodityRevenue",
        "commodityCostTimeSeries",
        "commodityRevenueTimeSeries",
    ],
)
def test_commodity_costs(parameter, esm, ureg):
    value = (
        pd.DataFrame({"DE": [2.0, 3.0]}) if parameter.endswith("TimeSeries") else 2.0
    )
    comp = fn.Source(
        esm,
        "costs",
        "electricity",
        True,
        **{parameter: value},
        units={parameter: ureg.EUR / ureg.kWh},
    )
    if isinstance(value, pd.DataFrame):
        pd.testing.assert_frame_equal(getattr(comp, parameter), value * 1000)
    else:
        assert getattr(comp, parameter) == pytest.approx(2000)


@pytest.mark.parametrize("has_capacity", [True, False])
def test_storage_operation_rates(esm, ureg, has_capacity):
    data = pd.DataFrame({"DE": [0.2, 0.4]})
    comp = fn.Storage(
        esm,
        "storage_rates",
        "electricity",
        hasCapacityVariable=has_capacity,
        chargeOpRateFix=data,
        units={"chargeOpRateFix": ureg.Unit("1/day") if has_capacity else ureg.GW},
    )
    pd.testing.assert_frame_equal(
        comp.chargeOpRateFix, data * (1 / 24 if has_capacity else 1000)
    )


def test_transmission_constructor(ureg):
    model = fn.EnergySystemModel(
        locations={"A", "B"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": "MW"},
        numberOfTimeSteps=2,
        costUnit="EUR",
        lengthUnit="km",
    )
    comp = fn.Transmission(
        model,
        "line",
        "electricity",
        capacityMax=2,
        distances=1000,
        investPerCapacity=3,
        investIfBuilt=4,
        operationRateFix=pd.DataFrame({"A_B": [50.0, 50.0], "B_A": [50.0, 50.0]}),
        units={
            "capacityMax": ureg.GW,
            "distances": ureg.m,
            "investPerCapacity": ureg.EUR / ureg.kW / ureg.m,
            "investIfBuilt": ureg.EUR / ureg.m,
            "operationRateFix": ureg.percent,
        },
    )
    assert (comp.processedCapacityMax[0] == 2000).all()
    assert (comp.distances == 1).all()
    assert comp.investPerCapacity == pytest.approx(3e6)
    assert comp.investIfBuilt == pytest.approx(4000)
    assert (comp.operationRateFix == 0.5).all().all()


def test_scaled_base_units(esm, ureg):
    assert esm.unitRegistry.convert(2e6, ureg.EUR, "1e6 Euro") == pytest.approx(2)
    assert esm.unitRegistry.convert(1000, ureg.kg / ureg.h, "t_CO2/h") == pytest.approx(
        1
    )


@pytest.mark.parametrize("units", [[], False, "", {"unknown": None}])
def test_invalid_units_mapping(esm, units):
    with pytest.raises((TypeError, ValueError)):
        fn.Source(esm, "bad", "electricity", True, units=units)


@pytest.mark.parametrize("descriptor", ["GW", 5])
def test_reject_non_unit_descriptors(esm, descriptor):
    with pytest.raises(TypeError):
        fn.Source(
            esm,
            "bad",
            "electricity",
            True,
            capacityMax=2,
            units={"capacityMax": descriptor},
        )


def test_export_does_not_repeat_conversion(esm, ureg):
    esm.add(
        fn.Source(
            esm,
            "exported",
            "electricity",
            True,
            capacityMax=2,
            units={"capacityMax": ureg.GW},
        )
    )
    from fine.IOManagement.dictIO import exportToDict, importFromDict  # noqa: PLC0415

    restored = importFromDict(*exportToDict(esm))
    assert restored.getComponent("exported").capacityMax == pytest.approx(2000)


def test_legacy_positional_arguments(esm):
    conversion = fn.Conversion(esm, "positional", "MW", {"electricity": 1}, False)
    assert conversion.hasCapacityVariable is False
