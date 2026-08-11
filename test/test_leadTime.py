import math

import pandas as pd
import pytest

import fine as fn


@pytest.fixture
def leadtime_test_esM():
    return fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"GW$_{el}$"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        lengthUnit="km",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        verboseLogLevel=2,
    )


def _add_source(esM, name, **kwargs):
    esM.add(
        fn.Source(
            esM=esM,
            name=name,
            commodity="electricity",
            hasCapacityVariable=True,
            **kwargs,
        )
    )
    return esM.getComponent(name)


def test_default_leadtime_is_zero(leadtime_test_esM):
    comp = _add_source(leadtime_test_esM, "default")
    assert comp.leadTime == 0
    for ip in leadtime_test_esM.investmentPeriods:
        assert (comp.processedLeadTime[ip] == 0).all()
        assert (comp.ipLeadTime[ip] == 0).all()
        assert (comp.roundedIpLeadTime[ip] == 0).all()


def test_none_leadtime_matches_zero(leadtime_test_esM):
    comp_none = _add_source(leadtime_test_esM, "none_lt", leadTime=None)
    comp_zero = _add_source(leadtime_test_esM, "zero_lt", leadTime=0)
    for ip in leadtime_test_esM.investmentPeriods:
        assert comp_none.processedLeadTime[ip].equals(comp_zero.processedLeadTime[ip])
        assert comp_none.ipLeadTime[ip].equals(comp_zero.ipLeadTime[ip])
        assert comp_none.roundedIpLeadTime[ip].equals(comp_zero.roundedIpLeadTime[ip])


def test_scalar_leadtime_broadcast_and_conversion(leadtime_test_esM):
    # interval = 5 years, leadTime = 2 years -> ipLeadTime = 0.4 -> ceil -> 1
    comp = _add_source(leadtime_test_esM, "scalar", leadTime=2)
    assert comp.leadTime == 2
    for ip in leadtime_test_esM.investmentPeriods:
        assert (comp.processedLeadTime[ip] == 2.0).all()
        assert (comp.ipLeadTime[ip] == 2 / 5).all()
        assert (comp.roundedIpLeadTime[ip] == math.ceil(2 / 5)).all()


def test_per_location_series_leadtime(leadtime_test_esM):
    series = pd.Series({"loc1": 1, "loc2": 3})
    comp = _add_source(leadtime_test_esM, "series", leadTime=series)
    for ip in leadtime_test_esM.investmentPeriods:
        assert comp.processedLeadTime[ip]["loc1"] == 1.0
        assert comp.processedLeadTime[ip]["loc2"] == 3.0
        assert comp.roundedIpLeadTime[ip]["loc1"] == math.ceil(1 / 5)
        assert comp.roundedIpLeadTime[ip]["loc2"] == math.ceil(3 / 5)


def test_per_investment_period_dict_leadtime_uses_calendar_year_keys(
    leadtime_test_esM,
):
    # keys are calendar years (2020, 2025, 2030), matching investPerCapacity's convention,
    # not raw investment-period indices (0, 1, 2).
    comp = _add_source(
        leadtime_test_esM, "dict_lt", leadTime={2020: 0, 2025: 1, 2030: 6}
    )
    assert comp.roundedIpLeadTime[0]["loc1"] == 0
    assert comp.roundedIpLeadTime[1]["loc1"] == math.ceil(1 / 5)
    assert comp.roundedIpLeadTime[2]["loc1"] == math.ceil(6 / 5)


def test_dict_missing_year_raises_clear_error(leadtime_test_esM):
    with pytest.raises(ValueError):
        _add_source(leadtime_test_esM, "missing_year", leadTime={2020: 0, 2025: 1})


def test_dict_wrong_year_raises_clear_error(leadtime_test_esM):
    with pytest.raises(ValueError):
        _add_source(
            leadtime_test_esM, "wrong_year", leadTime={2020: 0, 2025: 1, 2031: 2}
        )


def test_negative_scalar_raises(leadtime_test_esM):
    with pytest.raises(ValueError):
        _add_source(leadtime_test_esM, "neg", leadTime=-1)


def test_negative_series_entry_raises(leadtime_test_esM):
    with pytest.raises(ValueError):
        _add_source(
            leadtime_test_esM,
            "neg_series",
            leadTime=pd.Series({"loc1": -1, "loc2": 2}),
        )


def test_nan_scalar_raises_and_is_not_silently_zeroed(leadtime_test_esM):
    with pytest.raises(ValueError):
        _add_source(leadtime_test_esM, "nan_lt", leadTime=float("nan"))


def test_dict_input_is_not_mutated(leadtime_test_esM):
    user_dict = {2020: 0, 2025: 1, 2030: 2}
    user_dict_copy = dict(user_dict)
    _add_source(leadtime_test_esM, "no_mutate", leadTime=user_dict)
    assert user_dict == user_dict_copy


def test_two_dimensional_leadtime_for_transmission(leadtime_test_esM):
    esM = leadtime_test_esM
    esM.add(
        fn.Transmission(
            esM=esM,
            name="tx",
            commodity="electricity",
            hasCapacityVariable=True,
            leadTime=pd.Series({"loc1_loc2": 2, "loc2_loc1": 4}),
        )
    )
    tx = esM.getComponent("tx")
    for ip in esM.investmentPeriods:
        assert tx.processedLeadTime[ip]["loc1_loc2"] == 2.0
        assert tx.processedLeadTime[ip]["loc2_loc1"] == 4.0
        assert tx.roundedIpLeadTime[ip]["loc1_loc2"] == math.ceil(2 / 5)
        assert tx.roundedIpLeadTime[ip]["loc2_loc1"] == math.ceil(4 / 5)


@pytest.mark.parametrize(
    "leadTime_kwargs",
    [
        {},
        {"leadTime": 2},
        {"leadTime": pd.Series({"loc1": 1, "loc2": 3})},
        {"leadTime": {2020: 0, 2025: 1, 2030: 6}},
    ],
)
def test_leadtime_survives_dict_export_import_round_trip(
    leadtime_test_esM, leadTime_kwargs
):
    """Regression test: fn.dictIO.exportToDict/importFromDict round-trips a component
    by reading raw (unprocessed) constructor-argument-shaped attributes (getattr(component,
    "leadTime")) and feeding them straight back into the constructor. self.leadTime must
    therefore hold the same shape the user originally passed in (calendar-year-keyed dict,
    Series, or scalar) -- not the internally processed, investment-period-indexed dict --
    or the reconstructed component fails validation (or silently differs from the original).
    """
    esM = leadtime_test_esM
    comp = _add_source(esM, "src", **leadTime_kwargs)

    esm_dict, comp_dict = fn.dictIO.exportToDict(esM)
    rebuilt_esM = fn.dictIO.importFromDict(esm_dict, comp_dict)
    rebuilt_comp = rebuilt_esM.getComponent("src")

    for ip in esM.investmentPeriods:
        assert rebuilt_comp.processedLeadTime[ip].equals(comp.processedLeadTime[ip])
        assert rebuilt_comp.roundedIpLeadTime[ip].equals(comp.roundedIpLeadTime[ip])


@pytest.fixture
def stochastic_leadtime_test_esM():
    return fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"GW$_{el}$"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        lengthUnit="km",
        stochasticModel=True,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=1,
        startYear=2020,
        verboseLogLevel=2,
    )


@pytest.mark.parametrize(
    "leadTime_kwargs",
    [
        {"leadTime": 2},
        {"leadTime": pd.Series({"loc1": 0, "loc2": 1})},
        {"leadTime": {2020: 0, 2021: 3}},
    ],
)
def test_stochastic_model_with_nonzero_leadtime_raises(
    stochastic_leadtime_test_esM, leadTime_kwargs
):
    with pytest.raises(NotImplementedError):
        _add_source(stochastic_leadtime_test_esM, "x", **leadTime_kwargs)


@pytest.mark.parametrize(
    "leadTime_kwargs",
    [
        {},
        {"leadTime": 0},
        {"leadTime": pd.Series({"loc1": 0, "loc2": 0})},
        {"leadTime": {2020: 0, 2021: 0}},
    ],
)
def test_stochastic_model_with_zero_leadtime_does_not_raise(
    stochastic_leadtime_test_esM, leadTime_kwargs
):
    # value-aware fix: an all-zero Series/dict must NOT trigger the guard just because
    # the container itself is truthy (the old WIP's `leadTime != 0` check on a non-empty
    # dict/Series was always True regardless of its actual values).
    _add_source(stochastic_leadtime_test_esM, "x", **leadTime_kwargs)


_ETL_PARAMETERS = {
    "etlParameters": {
        "initCost": 1,
        "learningRate": 0.18,
        "initCapacity": 10,
        "maxCapacity": 50,
        "noSegments": 4,
    }
}


def test_pwlcf_parameters_with_nonzero_leadtime_raises(leadtime_test_esM):
    with pytest.raises(NotImplementedError):
        _add_source(
            leadtime_test_esM,
            "x",
            leadTime=2,
            pwlcfParameters=_ETL_PARAMETERS,
        )


def test_pwlcf_parameters_with_zero_leadtime_does_not_raise(leadtime_test_esM):
    comp = _add_source(
        leadtime_test_esM, "x", leadTime=0, pwlcfParameters=_ETL_PARAMETERS
    )
    assert comp.pwlcf is not None


def test_inactive_pwlcf_parameters_with_nonzero_leadtime_does_not_raise(
    leadtime_test_esM,
):
    # a pwlcfParameters dict whose values are all None is treated as inactive by the
    # component itself (no pwlcf module gets instantiated), so it must not trigger the
    # leadTime guard either.
    comp = _add_source(
        leadtime_test_esM,
        "x",
        leadTime=2,
        pwlcfParameters={"etlParameters": None, "eosParameters": None},
    )
    assert comp.pwlcf is None


def test_no_pwlcf_parameters_with_nonzero_leadtime_does_not_raise(leadtime_test_esM):
    _add_source(leadtime_test_esM, "x", leadTime=2)
