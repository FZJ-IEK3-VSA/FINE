import copy

import pytest
import fine as fn
from fine.expansionModules.rollingHorizon import (
    rollingHorizonOptimization,
    _DEFAULT_TSA_SETTINGS,
    _cachedGroupExists,
    _cachedIntervalConfigMismatches,
    _cachedIntervalChainMismatches,
    _stockCommissioningDiffers,
)
import numpy as np
import pandas as pd


_YEARS = [2020, 2025, 2030, 2035]


def _ts(value, n_steps=2):
    """Return a fixed operation-rate DataFrame for a single location."""
    return pd.DataFrame(
        np.full(n_steps, value),
        columns=["PerfectLand"],
        index=list(range(n_steps)),
    )


def _build_esM(edemand2020=2190):
    """Construct an esM whose components exercise all rolling horizon code paths.

    Source_cheap_then_expensive : stock accumulation + non-PerOperation dict param filtering
    Source_expensive_then_cheap : zero-commissioning guard (line 96 False branch)
    Source_short_lifetime       : outdated stock cleanup (lines 113-132); dedicated heat commodity
    Electrolyzer                : ip-dependent CCF (line 162)
    FuelCell                    : time-constant CCF (line 196 else:pass)
    EDemand / H2Demand          : electricity and hydrogen sinks
    HeatDemand                  : growing demand forces new commissioning every period

    edemand2020 lets a caller perturb the 2020 electricity demand (and thus the
    resulting 2020 commissioning) while keeping every other component identical,
    to build a "different chain" cache for the resume staleness tests below.
    """
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity", "hydrogen", "heat"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
            "heat": r"kW$_{th}$",
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=7860,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=4,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    # year-keyed investPerCapacity covers the non-PerOperation dict param branch (line 202)
    esM.add(
        fn.Source(
            esM=esM,
            name="Source_cheap_then_expensive",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity={2020: 1000, 2025: 900, 2030: 800, 2035: 700},
            interestRate=0.02,
            opexPerOperation={2020: 1, 2025: 1, 2030: 1, 2035: 100},
            economicLifetime=15,
            technicalLifetime=15,
        )
    )

    # opex=100 in 2020-2030 → optimizer commissions 0 → covers line 96 False branch
    esM.add(
        fn.Source(
            esM=esM,
            name="Source_expensive_then_cheap",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e3,
            interestRate=0.02,
            opexPerOperation={2020: 100, 2025: 100, 2030: 100, 2035: 1},
            economicLifetime=15,
        )
    )

    # Dedicated heat commodity forces commissioning in every period via growing HeatDemand.
    # technicalLifetime=9 → cleanup condition 2020 < 2030-9=2021 fires in [2030,2035].
    # Covers lines 113-132 (non-empty outdatedStockYears path).
    esM.add(
        fn.Source(
            esM=esM,
            name="Source_short_lifetime",
            commodity="heat",
            hasCapacityVariable=True,
            investPerCapacity=1e3,
            interestRate=0.02,
            opexPerOperation={2020: 1, 2025: 1, 2030: 1, 2035: 1},
            economicLifetime=9,
            technicalLifetime=9,
        )
    )

    # ip-dependent CCF: firstKey is a year → covers line 162 branch
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                2020: {"electricity": -1, "hydrogen": 0.65},
                2025: {"electricity": -1, "hydrogen": 0.67},
                2030: {"electricity": -1, "hydrogen": 0.69},
                2035: {"electricity": -1, "hydrogen": 0.71},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,
            interestRate=0.02,
            economicLifetime=15,
        )
    )

    # tuple-keyed CCF: firstKey is (commisYear, opYear) → covers lines 173-195 branch.
    # Exactly 9 valid pairs for technicalLifetime=15 across [2020,2025,2030,2035].
    # Varying efficiency per commissioning year makes FINE set isCommisDepending=True.
    esM.add(
        fn.Conversion(
            esM=esM,
            name="ElectrolyzerTuple",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2020, 2020): {"electricity": -1, "hydrogen": 0.60},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.59},
                (2020, 2030): {"electricity": -1, "hydrogen": 0.58},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.65},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.64},
                (2025, 2035): {"electricity": -1, "hydrogen": 0.63},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.70},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.69},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.75},
            },
            hasCapacityVariable=True,
            investPerCapacity=550,
            interestRate=0.02,
            economicLifetime=15,
            technicalLifetime=15,
        )
    )

    # time-constant CCF: firstKey is a commodity string → covers line 196 else:pass branch
    esM.add(
        fn.Conversion(
            esM=esM,
            name="FuelCell",
            physicalUnit=r"kW$_{H_{2},LHV}$",
            commodityConversionFactors={"hydrogen": -1, "electricity": 0.5},
            hasCapacityVariable=True,
            investPerCapacity=300,
            interestRate=0.02,
            economicLifetime=15,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix={
                2020: _ts(edemand2020),
                2025: _ts(4380),
                2030: _ts(6570),
                2035: _ts(8760),
            },
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix={year: _ts(100) for year in _YEARS},
        )
    )

    # Growing demand forces new heat capacity in every period, guaranteeing commissioning.
    esM.add(
        fn.Sink(
            esM=esM,
            name="HeatDemand",
            commodity="heat",
            hasCapacityVariable=False,
            operationRateFix={
                2020: _ts(500),
                2025: _ts(1000),
                2030: _ts(1500),
                2035: _ts(2000),
            },
        )
    )

    return esM


@pytest.fixture(scope="module")
def rh_results():
    esM = _build_esM()
    return rollingHorizonOptimization(
        esM=esM,
        scenario_name="test",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )


def _minimal_esM(n_periods):
    return fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=n_periods,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )


# ─── Error path tests (lines 26-32) ───────────────────────────────────────────


def test_raises_on_single_investment_period():
    """Line 26: numberOfInvestmentPeriods < 2 raises ValueError."""
    with pytest.raises(ValueError, match="At least two"):
        rollingHorizonOptimization(
            esM=_minimal_esM(1),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=1,
        )


def test_raises_when_window_not_smaller_than_periods():
    """Line 28: window >= numberOfInvestmentPeriods raises ValueError."""
    with pytest.raises(ValueError, match="at least one more"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=4,
        )


def test_raises_typeerror_for_non_int_window():
    """Line 34: non-integer window raises TypeError via isStrictlyPositiveInt."""
    with pytest.raises(TypeError, match="integer"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=2.0,
        )


def test_raises_valueerror_for_non_positive_window():
    """Line 34: zero/negative window raises ValueError via isStrictlyPositiveInt."""
    with pytest.raises(ValueError, match="strictly positive"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=0,
        )


def test_rolling_horizon_start_year_set_as_side_effect():
    """Lines 22-23: esM.rollingHorizonStartYear defaults to esM.startYear.

    This happens before the input checks raise, so it is observable even
    when the call errors out afterwards.
    """
    esM = _minimal_esM(1)
    assert esM.rollingHorizonStartYear is None
    with pytest.raises(ValueError):
        rollingHorizonOptimization(
            esM=esM,
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=1,
        )
    assert esM.rollingHorizonStartYear == esM.startYear == 2020


def test_raises_when_write_excel_output_without_export_path():
    """writeExcelOutput=True requires resultExportPath to be set."""
    with pytest.raises(ValueError, match="resultExportPath"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=2,
            writeExcelOutput=True,
        )


def test_raises_when_write_excel_output_without_scenario_name():
    """writeExcelOutput=True requires scenario_name to be set."""
    with pytest.raises(ValueError, match="scenario_name"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            numberOfInvestmentPeriodsForRollingHorizon=2,
            writeExcelOutput=True,
            resultExportPath="some/path",
        )


def test_raises_when_write_netcdf_output_without_export_path():
    """writeNetCDFOutput=True requires resultExportPath to be set."""
    with pytest.raises(ValueError, match="resultExportPath"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=2,
            writeNetCDFOutput=True,
        )


def test_raises_when_write_netcdf_output_without_scenario_name():
    """writeNetCDFOutput=True requires scenario_name to be set."""
    with pytest.raises(ValueError, match="scenario_name"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            numberOfInvestmentPeriodsForRollingHorizon=2,
            writeNetCDFOutput=True,
            resultExportPath="some/path",
        )


def test_raises_when_resume_without_export_path():
    """resume=True implies netCDF caching and requires resultExportPath to be set."""
    with pytest.raises(ValueError, match="resultExportPath"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=2,
            resume=True,
        )


def test_raises_when_resume_without_scenario_name():
    """resume=True implies netCDF caching and requires scenario_name to be set."""
    with pytest.raises(ValueError, match="scenario_name"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            numberOfInvestmentPeriodsForRollingHorizon=2,
            resume=True,
            resultExportPath="some/path",
        )


# ─── Results structure tests ───────────────────────────────────────────────────


def test_results_keys(rh_results):
    """Line 299: results keyed by first year of each interval (3 intervals for 4 periods, window=2)."""
    assert set(rh_results.keys()) == {2020, 2025, 2030}


def test_sub_esm_start_year(rh_results):
    """Line 219: each sub-esM has the correct startYear."""
    assert rh_results[2025].startYear == 2025


def test_sub_esm_number_of_investment_periods(rh_results):
    """Line 220: each sub-esM has numberOfInvestmentPeriods equal to the window size."""
    assert rh_results[2025].numberOfInvestmentPeriods == 2


def test_sub_esm_rolling_horizon_start_year_is_global_start_year(rh_results):
    """Lines 22-23/218: every sub-esM keeps the original overall startYear (2020)
    as rollingHorizonStartYear, independent of its own local startYear.

    This is what NPV reporting (component.py/sourceSink.py/storage.py/transmission.py)
    relies on to discount sub-esM results back to the global start year.
    """
    for year, sub_esM in rh_results.items():
        assert sub_esM.rollingHorizonStartYear == 2020
        assert sub_esM.startYear == year


# ─── Stock logic tests ─────────────────────────────────────────────────────────


def test_stock_from_first_to_second_interval(rh_results):
    """Lines 98-109 (branch a): 2020 commissioning stored as stock in [2025,2030]."""
    commis_2020 = (
        rh_results[2020]
        .getOptimizationSummary("SourceSinkModel", ip=2020)
        .loc["Source_cheap_then_expensive", "commissioning"]
        .iloc[0, 0]
    )
    stock_2020 = (
        rh_results[2025]
        .getComponent("Source_cheap_then_expensive")
        .stockCommissioning[2020]["PerfectLand"]
    )
    assert commis_2020 == stock_2020


def test_stock_accumulates_across_intervals(rh_results):
    """Lines 111-114 (branch b): both 2020 and 2025 commissioning present in [2030,2035] stock."""
    stock = (
        rh_results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning
    )
    assert 2020 in stock
    assert 2025 in stock


def test_stock_values_match_commissioning(rh_results):
    """StockCommissioning values must equal the optimization results they came from."""
    commis_2020 = (
        rh_results[2020]
        .getOptimizationSummary("SourceSinkModel", ip=2020)
        .loc["Source_cheap_then_expensive", "commissioning"]
        .iloc[0, 0]
    )
    commis_2025 = (
        rh_results[2025]
        .getOptimizationSummary("SourceSinkModel", ip=2025)
        .loc["Source_cheap_then_expensive", "commissioning"]
        .iloc[0, 0]
    )
    stock = (
        rh_results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning
    )
    assert stock[2020]["PerfectLand"] == commis_2020
    assert stock[2025]["PerfectLand"] == commis_2025


def test_zero_commissioning_not_added_to_stock(rh_results):
    """Line 96 False branch: zero commissioning in 2020 produces no stock entry in [2025,2030]."""
    stock = (
        rh_results[2025].getComponent("Source_expensive_then_cheap").stockCommissioning
    )
    assert stock is None or 2020 not in stock


def test_outdated_stock_removed(rh_results):
    """Lines 113-132: stock older than technicalLifetime removed.

    Source_short_lifetime (technicalLifetime=9): 2020 < 2030-9=2021 → cleaned in [2030,2035].
    """
    stock = rh_results[2030].getComponent("Source_short_lifetime").stockCommissioning
    assert stock is None or 2020 not in stock


# ─── Parameter filtering tests ─────────────────────────────────────────────────


def test_operation_params_filtered_to_window(rh_results):
    """Lines 200-201: PerOperation params filtered to rolling horizon years only.

    [2025,2030] sub-esM opexPerOperation must only contain {2025, 2030}.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    opex = comp_dict["Source"]["Source_cheap_then_expensive"]["opexPerOperation"]
    assert set(opex.keys()) == {2025, 2030}


def test_non_operation_dict_params_include_stock_years(rh_results):
    """Lines 202-203: non-PerOperation dict params filtered to window + stockYears.

    In [2025,2030], 2020 is a stockYear → investPerCapacity keeps key 2020.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    invest = comp_dict["Source"]["Source_cheap_then_expensive"]["investPerCapacity"]
    assert 2020 in invest
    assert 2025 in invest
    assert 2030 in invest
    assert 2035 not in invest


def test_ccf_ip_dependent_filtered_to_window(rh_results):
    """Lines 162-171: ip-dependent CCF filtered to rolling horizon years only.

    [2025,2030] sub-esM Electrolyzer CCF must only contain {2025, 2030}.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    ccf = comp_dict["Conversion"]["Electrolyzer"]["commodityConversionFactors"]
    assert set(ccf.keys()) == {2025, 2030}


def test_ccf_tuple_keyed_filtered_to_window(rh_results):
    """Lines 173-195: tuple (commisYear, opYear) CCF filtered so every opYear is in the window.

    [2025,2030] sub-esM ElectrolyzerTuple CCF must only contain tuples with opYear in {2025, 2030}.
    Pairs with opYear=2035 (e.g. (2025,2035)) and opYear=2020 must be absent.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    ccf = comp_dict["Conversion"]["ElectrolyzerTuple"]["commodityConversionFactors"]
    assert len(ccf) > 0
    assert all(isinstance(k, tuple) for k in ccf.keys())
    assert all(op_year in {2025, 2030} for (_, op_year) in ccf.keys())


def test_ccf_time_constant_unchanged_across_windows(rh_results):
    """Line 196 (else: pass): a CCF keyed directly by commodity name (no year/tuple
    dependency) must be passed through unchanged into every rolling horizon window.
    """
    expected = {"hydrogen": -1, "electricity": 0.5}
    for year in rh_results:
        _, comp_dict = fn.dictIO.exportToDict(rh_results[year])
        assert (
            comp_dict["Conversion"]["FuelCell"]["commodityConversionFactors"]
            == expected
        )


# ─── Myopic mode (window size 1) ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def rh_results_myopic():
    """numberOfInvestmentPeriodsForRollingHorizon=1 is the 'pure foresight'/myopic
    extreme mentioned in the module docstring: every investment period is optimized
    on its own, one at a time, chaining through all 4 periods (3 handoffs instead
    of 1), which exercises the stock persistence logic over more iterations.
    """
    esM = _build_esM()
    return rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_myopic",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=1,
    )


def test_myopic_results_keys(rh_results_myopic):
    """With window=1 and 4 investment periods, every period gets its own interval."""
    assert set(rh_results_myopic.keys()) == {2020, 2025, 2030, 2035}


def test_myopic_sub_esm_has_single_investment_period(rh_results_myopic):
    assert rh_results_myopic[2030].numberOfInvestmentPeriods == 1


def test_myopic_stock_accumulates_over_three_handoffs(rh_results_myopic):
    """Regression coverage for the persistedStock bug fix (commit 10c8900a):
    by the last (2035) iteration, commissioning from all three earlier periods
    must have survived being carried forward across three separate handoffs.
    """
    stock = (
        rh_results_myopic[2035]
        .getComponent("Source_cheap_then_expensive")
        .stockCommissioning
    )
    assert stock is not None
    assert {2020, 2025, 2030}.issubset(set(stock.keys()))


def test_myopic_outdated_stock_still_removed(rh_results_myopic):
    """Lines 113-132 also apply when chaining single-period windows:
    Source_short_lifetime (technicalLifetime=9) must have its 2020 stock
    dropped by the time the 2030 window is built (2020 < 2030-9=2021).
    """
    stock = (
        rh_results_myopic[2030].getComponent("Source_short_lifetime").stockCommissioning
    )
    assert stock is None or 2020 not in stock


# ─── Myopic parity with the retired simple myopic module (issue #640) ─────────
#
# fine.expansionModules.transformationPath.optimizeSimpleMyopic has been
# removed (it relies on utils.setNewCO2ReductionTarget, which no longer
# exists) and now raises NotImplementedError. These
# tests port its two use cases - a CO2 reduction pathway and technical-lifetime
# expiry of installed stock - onto rollingHorizonOptimization with
# numberOfInvestmentPeriodsForRollingHorizon=1, showing the rolling horizon
# module fully covers the old myopic use cases.


@pytest.fixture(scope="module")
def rh_results_co2_targets():
    """CO2 reduction targets expressed as a per-investment-period balanceLimit.

    A cheap, emitting gas plant is preferred over expensive wind whenever the
    CO2 budget allows it. The reduction target tightens from 25% (2020) to
    100% (2030), so emissions should shrink to exactly zero once no budget
    is left, forcing a switch to wind despite its higher cost.
    """
    years = [2020, 2025, 2030]
    CO2Reference = 100
    reductionTargets = {2020: 0.25, 2025: 0.5, 2030: 1.0}
    balanceLimit = {
        year: pd.DataFrame(
            index=["CO2 limit"],
            columns=["Total", "lowerBound"],
            data=[[-CO2Reference * (1 - reductionTargets[year]), True]],
        )
        for year in years
    }

    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity", "naturalGas", "CO2"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "naturalGas": r"kW$_{CH_{4},LHV}$",
            "CO2": r"t$_{CO_2}$",
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
        balanceLimit=balanceLimit,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame(
                np.array([[0.5], [0.2]]), columns=["PerfectLand"]
            ),
            investPerCapacity=2000,
            interestRate=0.05,
            economicLifetime=20,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas import",
            commodity="naturalGas",
            hasCapacityVariable=False,
            commodityCost=0.02,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Gas power plant",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": 1, "naturalGas": -2, "CO2": 1},
            hasCapacityVariable=True,
            investPerCapacity=200,
            opexPerCapacity=200 * 0.03,
            interestRate=0.05,
            economicLifetime=20,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                np.array([[6], [4]]), columns=["PerfectLand"]
            ),
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to environment",
            commodity="CO2",
            hasCapacityVariable=False,
            balanceLimitID="CO2 limit",
        )
    )

    return rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_co2_targets",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=1,
    )


def _co2_emissions(esM, year):
    return (
        esM.getOptimizationSummary("SourceSinkModel", ip=year)
        .loc["CO2 to environment"]
        .loc["operation"]
        .sum()
        .sum()
    )


def test_co2_target_binding_forces_zero_emissions(rh_results_co2_targets):
    """A 100% reduction target (2030) leaves no CO2 budget, so emissions must be 0."""
    assert _co2_emissions(rh_results_co2_targets[2030], 2030) == pytest.approx(0)


def test_co2_target_loose_allows_emissions(rh_results_co2_targets):
    """A loose 25% reduction target (2020) does not bind the cheap gas plant,
    so it is used and emissions are non-zero.
    """
    assert _co2_emissions(rh_results_co2_targets[2020], 2020) > 0


def test_co2_target_forces_wind_investment_once_binding(rh_results_co2_targets):
    """Once the gas plant is priced out by the CO2 constraint, capacity must be
    installed in the emission-free alternative (wind) to still meet demand.
    """
    windCapacity = (
        rh_results_co2_targets[2030]
        .getOptimizationSummary("SourceSinkModel", ip=2030)
        .loc["Wind"]
        .loc["capacity"]
        .sum()
        .sum()
    )
    assert windCapacity > 0


@pytest.fixture(scope="module")
def rh_results_exceeded_lifetime():
    """Technical lifetime shorter than the modeled time horizon (issue #640,
    test_exceededLifetime): an electrolyzer commissioned in 2020 with a
    technicalLifetime of 7 years must have fallen out of the stock by 2030
    (2020 < 2030 - 7 = 2023), the same behaviour the retired simple myopic
    module verified via a "_stock_2020" component that no longer existed.
    """
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    esM = fn.EnergySystemModel(
        locations={"OneLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )

    costs = pd.DataFrame([np.array([0.05, 0.0, 0.1, 0.051])], index=["OneLocation"]).T
    revenues = pd.DataFrame([np.array([0.0, 0.01, 0.0, 0.0])], index=["OneLocation"]).T
    maxpurchase = (
        pd.DataFrame([np.array([1e6, 1e6, 1e6, 1e6])], index=["OneLocation"]).T
        * hoursPerTimeStep
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=maxpurchase,
            commodityCostTimeSeries=costs,
            commodityRevenueTimeSeries=revenues,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=7,
            technicalLifetime=7,
        )
    )

    esM.add(
        fn.Storage(
            esM=esM,
            name="Pressure tank",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    demand = (
        pd.DataFrame([np.array([6e3, 6e3, 6e3, 6e3])], index=["OneLocation"]).T
        * hoursPerTimeStep
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    return rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_exceeded_lifetime",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=1,
    )


def test_exceeded_lifetime_commissions_stock_in_2020(rh_results_exceeded_lifetime):
    """Sanity check: electrolyzer capacity is actually commissioned in 2020,
    otherwise the removal assertion below would be vacuously true.
    """
    commis_2020 = (
        rh_results_exceeded_lifetime[2020]
        .getOptimizationSummary("ConversionModel", ip=2020)
        .loc["Electrolyzers", "commissioning"]
        .sum()
        .sum()
    )
    assert commis_2020 > 0


def test_exceeded_lifetime_stock_dropped_by_2030(rh_results_exceeded_lifetime):
    """2020 stock (technicalLifetime=7) must be gone from the 2030 window:
    2020 < 2030 - 7 = 2023.
    """
    stock = (
        rh_results_exceeded_lifetime[2030]
        .getComponent("Electrolyzers")
        .stockCommissioning
    )
    assert stock is None or 2020 not in stock


# ─── timeSeriesAggregationSettings passthrough ─────────────────────────────────
#
# numberOfTypicalPeriods/numberOfTimeStepsPerPeriod/numberOfSegments/clusterMethod
# used to be individual rollingHorizonOptimization parameters, which restricted
# callers to only those tsam settings. They are now a single
# timeSeriesAggregationSettings dict passed straight through to
# EnergySystemModel.aggregateTemporally, so any tsam kwarg is reachable.


def test_default_tsa_settings_match_original_hardcoded_values():
    """_DEFAULT_TSA_SETTINGS preserves the values that used to be hardcoded,
    so callers who don't pass timeSeriesAggregationSettings see unchanged
    behavior.
    """
    assert _DEFAULT_TSA_SETTINGS == {
        "numberOfTypicalPeriods": 7,
        "numberOfTimeStepsPerPeriod": 24,
        "numberOfSegmentsPerPeriod": 16,
        "segmentation": True,
        "clusterMethod": "hierarchical",
        "sortValues": True,
        "rescaleClusterPeriods": True,
        "representationMethod": None,
    }


def test_partial_tsa_settings_override_merges_with_defaults():
    """Passing only one key in timeSeriesAggregationSettings must not reset
    the other tsam settings. numberOfTypicalPeriods is left at its default
    (7), which the 2-time-step test system cannot satisfy (7*1 > 2),
    proving the default is still active alongside the override.
    """
    esM = _build_esM()
    with pytest.raises(
        ValueError, match="product of the numberOfTypicalPeriods"
    ):
        rollingHorizonOptimization(
            esM=esM,
            scenario_name="test_partial_tsa",
            timeSeriesAggregation=True,
            timeSeriesAggregationSettings={"numberOfTimeStepsPerPeriod": 1},
            numberOfInvestmentPeriodsForRollingHorizon=2,
        )


@pytest.fixture(scope="module")
def rh_results_tsa_custom():
    """Override numberOfTypicalPeriods/numberOfTimeStepsPerPeriod/
    numberOfSegmentsPerPeriod via timeSeriesAggregationSettings to reach
    aggregateTemporally: the default values (7 typical periods, 24 time
    steps per period) are impossible to satisfy for this 2-time-step test
    system, so a successful run here proves the override was applied.
    """
    esM = _build_esM()
    return rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_tsa_custom",
        timeSeriesAggregation=True,
        timeSeriesAggregationSettings={
            "numberOfTypicalPeriods": 2,
            "numberOfTimeStepsPerPeriod": 1,
            "numberOfSegmentsPerPeriod": 1,
        },
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )


def test_tsa_settings_passthrough_controls_clustering(rh_results_tsa_custom):
    """The overridden values are the ones actually used for clustering."""
    esM = rh_results_tsa_custom[2020]
    assert len(esM.typicalPeriods) == 2
    assert len(esM.timeStepsPerPeriod) == 1
    assert len(esM.segmentsPerPeriod) == 1


# ─── netCDF output (xarrayIO) and resume ───────────────────────────────────────
#
# writeExcelOutput used to be the only way to persist rolling horizon results.
# writeNetCDFOutput saves every interval's full esM (input and output) into a
# single shared netCDF file, one group per interval keyed by its start year
# (consistent with perfect foresight's single-file output, unlike Excel's one
# file per interval), which also enables resuming an interrupted run
# (resume=True) instead of re-solving already completed intervals.


def _shared_netcdf_path(dir_path, scenario_name="netcdf_cache"):
    return dir_path / f"{scenario_name}_rollingHorizon.nc"


def _write_cached_group(esM_obj, dir_path, startYear, scenario_name="netcdf_cache"):
    """Write a single interval's esM into its own group of the shared netCDF
    file, without touching any other group already there. Lets tests build
    specific (partial, mixed-origin, ...) cache scenarios directly from
    already-solved esM objects, without re-solving or copying files.
    """
    fn.writeEnergySystemModelToNetCDF(
        esM_obj,
        outputFilePath=str(_shared_netcdf_path(dir_path, scenario_name)),
        overwriteExisting=False,
        groupPrefix=str(startYear),
    )


@pytest.fixture(scope="module")
def rh_netcdf_cache(tmp_path_factory):
    """Run rolling horizon once with writeNetCDFOutput=True, producing one
    shared netCDF file with one group per interval. Shared across the
    netCDF/resume tests below to avoid re-solving the same model repeatedly.
    """
    export_dir = tmp_path_factory.mktemp("rh_netcdf_cache")
    esM = _build_esM()
    results = rollingHorizonOptimization(
        esM=esM,
        scenario_name="netcdf_cache",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        writeNetCDFOutput=True,
        resultExportPath=str(export_dir),
    )
    return results, export_dir


def _commissioning(esM, year):
    return (
        esM.getOptimizationSummary("SourceSinkModel", ip=year)
        .loc["Source_cheap_then_expensive", "commissioning"]
        .iloc[0, 0]
    )


def test_write_netcdf_output_creates_one_shared_file_with_one_group_per_interval(
    rh_netcdf_cache,
):
    """writeNetCDFOutput=True writes a single netCDF file, named after
    scenario_name, holding one group per rolling horizon interval, keyed by
    its start year -- mirroring perfect foresight's single-file output
    instead of writing one file per interval.
    """
    _, export_dir = rh_netcdf_cache
    netCDFPath = _shared_netcdf_path(export_dir)
    assert netCDFPath.is_file()
    for year in (2020, 2025, 2030):
        assert _cachedGroupExists(netCDFPath, str(year))


def test_netcdf_output_round_trips_optimization_summary(rh_netcdf_cache):
    """A cached interval group, read back via xarrayIO, reproduces the same
    commissioning values as the in-memory result it was written from.
    """
    results, export_dir = rh_netcdf_cache
    loaded = fn.readNetCDFtoEnergySystemModel(
        str(_shared_netcdf_path(export_dir)), groupPrefix="2020"
    )
    assert _commissioning(loaded, 2020) == pytest.approx(
        _commissioning(results[2020], 2020)
    )


def _track_optimize_calls(monkeypatch):
    """Patch EnergySystemModel.optimize to record which sub-esM's startYear
    it was called on, without changing its behavior.
    """
    calls = []
    original_optimize = fn.EnergySystemModel.optimize

    def _tracking_optimize(self, *args, **kwargs):
        calls.append(self.startYear)
        return original_optimize(self, *args, **kwargs)

    monkeypatch.setattr(fn.EnergySystemModel, "optimize", _tracking_optimize)
    return calls


def test_resume_skips_optimize_when_all_intervals_cached(
    rh_netcdf_cache, monkeypatch
):
    """resume=True must load every interval from its cached group instead
    of rebuilding and re-solving it, once the cache is fully populated.
    This is the point of resuming a finished/interrupted run: no interval
    should be solved twice.
    """
    original_results, export_dir = rh_netcdf_cache
    optimize_calls = _track_optimize_calls(monkeypatch)

    resumed_esM = _build_esM()
    resumed_results = rollingHorizonOptimization(
        esM=resumed_esM,
        scenario_name="netcdf_cache",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        resume=True,
        resultExportPath=str(export_dir),
    )

    assert optimize_calls == []
    assert set(resumed_results.keys()) == {2020, 2025, 2030}
    for year in resumed_results:
        assert _commissioning(resumed_results[year], year) == pytest.approx(
            _commissioning(original_results[year], year)
        )


def test_resume_partial_cache_solves_only_missing_intervals(
    rh_netcdf_cache, tmp_path, monkeypatch
):
    """If only the first interval's group exists (simulating a run
    interrupted right after it), resuming must load that interval from
    cache and only solve the remaining ones. The final results must match
    an uninterrupted run exactly, proving stock bookkeeping is unaffected
    by reloading (rather than re-solving) the earlier interval.
    """
    original_results, _ = rh_netcdf_cache
    _write_cached_group(original_results[2020], tmp_path, 2020)
    optimize_calls = _track_optimize_calls(monkeypatch)

    resumed_esM = _build_esM()
    resumed_results = rollingHorizonOptimization(
        esM=resumed_esM,
        scenario_name="netcdf_cache",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        resume=True,
        resultExportPath=str(tmp_path),
    )

    assert optimize_calls == [2025, 2030]
    for year in (2020, 2025, 2030):
        assert _commissioning(resumed_results[year], year) == pytest.approx(
            _commissioning(original_results[year], year)
        )


def test_resume_raises_on_mismatched_cache(rh_netcdf_cache, tmp_path):
    """If a cached interval's window size doesn't match what the current
    call expects (e.g. numberOfInvestmentPeriodsForRollingHorizon changed
    between runs), resuming from it must fail loudly instead of silently
    producing an inconsistent result.
    """
    original_results, _ = rh_netcdf_cache
    _write_cached_group(
        original_results[2020], tmp_path, 2020, scenario_name="mismatch"
    )
    esM = _build_esM()
    with pytest.raises(ValueError, match="does not match this call's configuration"):
        rollingHorizonOptimization(
            esM=esM,
            scenario_name="mismatch",
            numberOfInvestmentPeriodsForRollingHorizon=1,
            resume=True,
            resultExportPath=str(tmp_path),
        )


# ─── Cache validation helpers (unit-level, no solves) ──────────────────────────
#
# _cachedIntervalConfigMismatches / _cachedIntervalChainMismatches implement the
# two safety checks resume relies on: a hard check that this call's own window
# configuration matches the cache, and a soft check that the cache was actually
# built from the same component set + accumulated stock as what was just
# recomputed for it. Tested directly here since constructing real "stale cache"
# scenarios end-to-end requires two full solves (covered separately below).


class _FakeCachedEsm:
    def __init__(self, startYear, numberOfInvestmentPeriods):
        self.startYear = startYear
        self.numberOfInvestmentPeriods = numberOfInvestmentPeriods


def test_config_mismatch_detects_start_year_difference():
    reasons = _cachedIntervalConfigMismatches(
        _FakeCachedEsm(startYear=2020, numberOfInvestmentPeriods=2),
        rollingHorizonYears=[2025, 2030],
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert any("startYear" in r for r in reasons)


def test_config_mismatch_detects_window_size_difference():
    reasons = _cachedIntervalConfigMismatches(
        _FakeCachedEsm(startYear=2020, numberOfInvestmentPeriods=2),
        rollingHorizonYears=[2020, 2025],
        numberOfInvestmentPeriodsForRollingHorizon=1,
    )
    assert any("numberOfInvestmentPeriods" in r for r in reasons)


def test_config_mismatch_empty_when_matching():
    reasons = _cachedIntervalConfigMismatches(
        _FakeCachedEsm(startYear=2020, numberOfInvestmentPeriods=2),
        rollingHorizonYears=[2020, 2025],
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert reasons == []


def test_stock_commissioning_differs_none_vs_none():
    assert _stockCommissioningDiffers(None, None) is False


def test_stock_commissioning_differs_none_vs_value():
    stock = {2020: pd.Series({"PerfectLand": 1.0})}
    assert _stockCommissioningDiffers(None, stock) is True
    assert _stockCommissioningDiffers(stock, None) is True


def test_stock_commissioning_differs_within_tolerance_is_not_a_difference():
    fresh = {2020: pd.Series({"PerfectLand": 1.0000001})}
    cached = {2020: pd.Series({"PerfectLand": 1.0000002})}
    assert _stockCommissioningDiffers(fresh, cached) is False


def test_stock_commissioning_differs_beyond_tolerance():
    fresh = {2020: pd.Series({"PerfectLand": 1.0})}
    cached = {2020: pd.Series({"PerfectLand": 1.1})}
    assert _stockCommissioningDiffers(fresh, cached) is True


def test_stock_commissioning_differs_on_different_years():
    fresh = {2020: pd.Series({"PerfectLand": 1.0})}
    cached = {2025: pd.Series({"PerfectLand": 1.0})}
    assert _stockCommissioningDiffers(fresh, cached) is True


def _tiny_esM_with_source(**sourceKwargs):
    esM = _minimal_esM(2)
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            **sourceKwargs,
        )
    )
    return esM


def test_chain_mismatch_detects_component_set_difference():
    esM = _tiny_esM_with_source(hasCapacityVariable=False)
    freshCompDict = {"Source": {}}
    reasons = _cachedIntervalChainMismatches(esM, freshCompDict)
    assert any("component set" in r for r in reasons)


def test_chain_mismatch_detects_stock_difference():
    esM = _tiny_esM_with_source(
        hasCapacityVariable=True,
        investPerCapacity=1,
        interestRate=0.02,
        economicLifetime=10,
        stockCommissioning={2015: pd.Series({"PerfectLand": 5.0})},
    )
    _, cachedCompDict = fn.dictIO.exportToDict(esM)
    freshCompDict = copy.deepcopy(cachedCompDict)
    freshCompDict["Source"]["Src"]["stockCommissioning"] = {
        2015: pd.Series({"PerfectLand": 999.0})
    }
    reasons = _cachedIntervalChainMismatches(esM, freshCompDict)
    assert any("stockCommissioning" in r for r in reasons)


def test_chain_mismatch_empty_when_matching():
    esM = _tiny_esM_with_source(hasCapacityVariable=False)
    _, cachedCompDict = fn.dictIO.exportToDict(esM)
    reasons = _cachedIntervalChainMismatches(esM, cachedCompDict)
    assert reasons == []


# ─── Stale cache: end-to-end warn-discard-resolve, and monotonic fallback ──────


@pytest.fixture(scope="module")
def rh_netcdf_cache_perturbed(tmp_path_factory):
    """Run a second full rolling horizon with a different 2020 electricity
    demand, so its resulting 2020 commissioning -- and therefore the
    stockCommissioning baked into its *later* cached intervals -- differs
    from a chain that starts at rh_netcdf_cache's 2020 result. Used to
    construct a genuinely stale cache group below, rather than just a
    structurally-invalid one.
    """
    export_dir = tmp_path_factory.mktemp("rh_netcdf_cache_perturbed")
    esM = _build_esM(edemand2020=2190 * 3)
    results = rollingHorizonOptimization(
        esM=esM,
        scenario_name="netcdf_cache",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        writeNetCDFOutput=True,
        resultExportPath=str(export_dir),
    )
    return results, export_dir


def test_resume_discards_stale_cache_and_solves_fresh(
    rh_netcdf_cache, rh_netcdf_cache_perturbed, tmp_path, monkeypatch
):
    """Build a shared cache file where 2020 is consistent but 2025 was
    actually produced by a *different* 2020 (the perturbed run) -- as if
    an earlier interval got re-solved with different inputs between runs,
    leaving a stale downstream cache group in place. 2030's group is left
    as the ORIGINAL (2020-consistent) one.

    Resuming must: (1) load 2020 from cache, (2) detect 2025's cache is
    stale, warn, discard it, and solve 2025 fresh from the correct (2020-
    consistent) chain, and (3) per the monotonic-fallback guard, also solve
    2030 fresh even though its own cached group is individually consistent
    -- because it was never validated against the freshly-solved 2025.
    The final results must match an uninterrupted single run exactly,
    proving the discard-and-resolve path is fully self-correcting.
    """
    original_results, _ = rh_netcdf_cache
    perturbed_results, _ = rh_netcdf_cache_perturbed

    _write_cached_group(original_results[2020], tmp_path, 2020)
    _write_cached_group(perturbed_results[2025], tmp_path, 2025)
    _write_cached_group(original_results[2030], tmp_path, 2030)

    optimize_calls = _track_optimize_calls(monkeypatch)

    resumed_esM = _build_esM()
    with pytest.warns(UserWarning, match="stale"):
        resumed_results = rollingHorizonOptimization(
            esM=resumed_esM,
            scenario_name="netcdf_cache",
            timeSeriesAggregation=False,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            resume=True,
            resultExportPath=str(tmp_path),
        )

    assert optimize_calls == [2025, 2030]
    for year in (2020, 2025, 2030):
        assert _commissioning(resumed_results[year], year) == pytest.approx(
            _commissioning(original_results[year], year)
        )
