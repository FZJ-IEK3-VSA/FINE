import copy
import inspect
import warnings

import pytest
import fine as fn
from fine.expansionModules.rollingHorizon import (
    rollingHorizonOptimization,
    _STOCK_YEAR_PARAMETERS,
    _buildIntervalComponentDict,
    _cachedGroupExists,
    _cachedIntervalConfigMismatches,
    _cachedIntervalChainMismatches,
    _filterComponentParametersForInterval,
    _stockCommissioningDiffers,
)
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from fine.utils import ImplementedSolvers
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

    Source_cheap_then_expensive : stock accumulation + stock-year parameter filtering
    Source_expensive_then_cheap : commissioning below stockCommissioningThreshold
    Source_short_lifetime       : outdated stock cleanup; dedicated heat commodity
    Electrolyzer                : ip-dependent commodityConversionFactors
    FuelCell                    : time-constant commodityConversionFactors
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

    # year-keyed investPerCapacity covers the stock-year parameter branch
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

    # opex=100 in 2020-2030 → optimizer commissions 0 → covers the branch of
    # _updateStockCommissioningForInterval that carries nothing over as stock
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
    # Covers the non-empty outdatedStockYears path of
    # _updateStockCommissioningForInterval.
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

    # ip-dependent CCF: firstKey is a year → covers that branch of
    # _filterComponentParametersForInterval
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

    # tuple-keyed CCF: firstKey is (commisYear, opYear) → covers that branch of
    # _filterComponentParametersForInterval.
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

    # time-constant CCF: firstKey is a commodity string → covers the branch of
    # _filterComponentParametersForInterval that leaves the parameter untouched
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


# ─── Error path tests ─────────────────────────────────────────────────────────


def test_raises_on_single_investment_period():
    """A model of fewer than two investment periods raises ValueError."""
    with pytest.raises(ValueError, match="At least two"):
        rollingHorizonOptimization(
            esM=_minimal_esM(1),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=1,
        )


def test_raises_when_window_not_smaller_than_periods():
    """A window as long as the pathway is perfect foresight and raises ValueError."""
    with pytest.raises(ValueError, match="at least one more"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=4,
        )


@pytest.mark.parametrize("window", [2.0, "2", None])
def test_raises_typeerror_for_non_int_window(window):
    """A non-integer window raises TypeError via utils.isStrictlyPositiveInt.

    The type check runs before the window is compared against the number of investment
    periods, so the error names the actual problem instead of failing inside a
    comparison that a wrongly typed window does not support.
    """
    with pytest.raises(TypeError, match="has to be an integer"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=window,
        )


def test_raises_valueerror_for_non_positive_window():
    """A zero/negative window raises ValueError via utils.isStrictlyPositiveInt."""
    with pytest.raises(ValueError, match="strictly positive"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=0,
        )


def test_the_passed_esm_is_not_modified():
    """The windows are built from a copy; the caller's model keeps its own state.

    rollingHorizonStartYear used to be set on the passed esM so that exportToDict would
    pick it up, which left it behind afterwards and made a later stand-alone optimize of
    the same model report an NPVcontributionRH row it has no rolling horizon for.
    """
    esM = _build_esM()
    before = copy.deepcopy(esM.componentNames)
    assert esM.rollingHorizonStartYear is None

    results = rollingHorizonOptimization(
        esM=esM,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )

    assert esM.rollingHorizonStartYear is None
    assert esM.startYear == 2020
    assert esM.numberOfInvestmentPeriods == 4
    assert esM.componentNames == before
    # the windows still inherit the pathway's start year, which is what
    # NPVcontributionRH is discounted onto
    assert all(sub.rollingHorizonStartYear == 2020 for sub in results.values())


def test_a_user_set_rolling_horizon_start_year_is_kept():
    """An explicitly set rollingHorizonStartYear wins over the pathway's startYear."""
    esM = _minimal_esM_with_source(n_periods=3, rollingHorizonStartYear=2010)
    results = rollingHorizonOptimization(
        esM=esM,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert esM.rollingHorizonStartYear == 2010
    assert all(sub.rollingHorizonStartYear == 2010 for sub in results.values())


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
    """Results are keyed by the first year of each interval (3 intervals for 4 periods, window=2)."""
    assert set(rh_results.keys()) == {2020, 2025, 2030}


def test_sub_esm_start_year(rh_results):
    """_buildIntervalEsm gives each sub-esM the interval's own startYear."""
    assert rh_results[2025].startYear == 2025


def test_sub_esm_number_of_investment_periods(rh_results):
    """_buildIntervalEsm gives each sub-esM numberOfInvestmentPeriods equal to the window size."""
    assert rh_results[2025].numberOfInvestmentPeriods == 2


def test_sub_esm_rolling_horizon_start_year_is_global_start_year(rh_results):
    """Every sub-esM keeps the original overall startYear (2020)
    as rollingHorizonStartYear, independent of its own local startYear.

    This is what NPV reporting (component.py/sourceSink.py/storage.py/transmission.py)
    relies on to discount sub-esM results back to the global start year.
    """
    for year, sub_esM in rh_results.items():
        assert sub_esM.rollingHorizonStartYear == 2020
        assert sub_esM.startYear == year


# ─── Stock logic tests ─────────────────────────────────────────────────────────


def test_stock_from_first_to_second_interval(rh_results):
    """_updateStockCommissioningForInterval, no stock yet: 2020 commissioning stored as stock in [2025,2030]."""
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
    # approx, not equality: the commissioning result is rounded before it is stored as
    # stock, as utils.checkAndSetStock expects for that quantity
    assert stock_2020 == pytest.approx(commis_2020)


def test_stock_accumulates_across_intervals(rh_results):
    """_updateStockCommissioningForInterval, stock exists: 2020 and 2025 commissioning both in [2030,2035] stock."""
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
    assert stock[2020]["PerfectLand"] == pytest.approx(commis_2020)
    assert stock[2025]["PerfectLand"] == pytest.approx(commis_2025)


def test_zero_commissioning_not_added_to_stock(rh_results):
    """Commissioning below stockCommissioningThreshold produces no stock entry in [2025,2030]."""
    stock = (
        rh_results[2025].getComponent("Source_expensive_then_cheap").stockCommissioning
    )
    assert stock is None or 2020 not in stock


def test_outdated_stock_removed(rh_results):
    """_updateStockCommissioningForInterval prunes stock older than technicalLifetime.

    Source_short_lifetime (technicalLifetime=9): 2020 < 2030-9=2021 → cleaned in [2030,2035].
    """
    stock = rh_results[2030].getComponent("Source_short_lifetime").stockCommissioning
    assert stock is None or 2020 not in stock


# ─── Parameter filtering tests ─────────────────────────────────────────────────


def test_operation_params_filtered_to_window(rh_results):
    """_filterComponentParametersForInterval keeps PerOperation params of the window years only.

    [2025,2030] sub-esM opexPerOperation must only contain {2025, 2030}.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    opex = comp_dict["Source"]["Source_cheap_then_expensive"]["opexPerOperation"]
    assert set(opex.keys()) == {2025, 2030}


def test_stock_year_dict_params_include_stock_years(rh_results):
    """_filterComponentParametersForInterval keeps a stock-year parameter for the window
    years plus the stockYears.

    In [2025,2030], 2020 is a stockYear → investPerCapacity keeps key 2020.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    invest = comp_dict["Source"]["Source_cheap_then_expensive"]["investPerCapacity"]
    assert 2020 in invest
    assert 2025 in invest
    assert 2030 in invest
    assert 2035 not in invest


def test_ccf_ip_dependent_filtered_to_window(rh_results):
    """_filterComponentParametersForInterval keeps ip-dependent CCF of the window years only.

    [2025,2030] sub-esM Electrolyzer CCF must only contain {2025, 2030}.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    ccf = comp_dict["Conversion"]["Electrolyzer"]["commodityConversionFactors"]
    assert set(ccf.keys()) == {2025, 2030}


def test_ccf_tuple_keyed_filtered_to_window(rh_results):
    """_filterComponentParametersForInterval keeps tuple (commisYear, opYear) CCF whose opYear is in the window.

    [2025,2030] sub-esM ElectrolyzerTuple CCF must only contain tuples with opYear in {2025, 2030}.
    Pairs with opYear=2035 (e.g. (2025,2035)) and opYear=2020 must be absent.
    """
    _, comp_dict = fn.dictIO.exportToDict(rh_results[2025])
    ccf = comp_dict["Conversion"]["ElectrolyzerTuple"]["commodityConversionFactors"]
    assert len(ccf) > 0
    assert all(isinstance(k, tuple) for k in ccf.keys())
    assert all(op_year in {2025, 2030} for (_, op_year) in ccf.keys())


def test_ccf_time_constant_unchanged_across_windows(rh_results):
    """_filterComponentParametersForInterval leaves a CCF keyed directly by commodity name (no year/tuple
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
    """Pruning also applies when chaining single-period windows:
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
# The individual clustering parameters rollingHorizonOptimization used to carry
# restricted callers to exactly those tsam settings. They are now a single
# timeSeriesAggregationSettings dict passed straight through to
# EnergySystemModel.aggregateTemporally, so any ETHOS.TSAM keyword argument is
# reachable. Settings not given fall back to aggregateTemporally's own defaults
# directly - rolling horizon no longer maintains a default policy of its own.


def test_partial_tsa_settings_override_merges_with_defaults():
    """Passing only one key in timeSeriesAggregationSettings must not reset
    the other tsam settings. n_clusters is left at aggregateTemporally's own
    default (40), which the 2-time-step test system cannot satisfy, proving
    the default is still active alongside the override.
    """
    esM = _build_esM()
    with pytest.raises(ValueError, match="product of the numberOfTypicalPeriods"):
        rollingHorizonOptimization(
            esM=esM,
            scenario_name="test_partial_tsa",
            timeSeriesAggregation=True,
            timeSeriesAggregationSettings={"period_duration": 7860},
            numberOfInvestmentPeriodsForRollingHorizon=2,
        )


@pytest.fixture(scope="module")
def rh_results_tsa_custom():
    """Override the clustering through timeSeriesAggregationSettings to reach
    aggregateTemporally: its default values (40 typical periods of 24 hours)
    are impossible to satisfy for this 2-time-step test system, so a
    successful run here proves the override was applied.
    """
    esM = _build_esM()
    return rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_tsa_custom",
        timeSeriesAggregation=True,
        timeSeriesAggregationSettings={
            "n_clusters": 2,
            "period_duration": 7860,
            "segments": fn.SegmentConfig(n_segments=1),
        },
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )


def test_tsa_settings_passthrough_controls_clustering(rh_results_tsa_custom):
    """The overridden values are the ones actually used for clustering."""
    esM = rh_results_tsa_custom[2020]
    assert len(esM.typicalPeriods) == 2
    assert len(esM.timeStepsPerPeriod) == 1
    assert len(esM.segmentsPerPeriod) == 1


def test_tsa_settings_accept_a_cluster_config_with_its_own_solver():
    """The clustering solver is part of timeSeriesAggregationSettings, not of this
    function's solver argument (which selects the solver of the optimization).
    Passing a ClusterConfig must therefore not collide with anything rolling
    horizon injects itself - it used to pass solver= to aggregateTemporally, which
    the ETHOS.TSAM 4.x interface refuses to combine with a ClusterConfig.
    """
    esM = _build_esM()
    results = rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_tsa_cluster_config",
        timeSeriesAggregation=True,
        timeSeriesAggregationSettings={
            "n_clusters": 2,
            "period_duration": 7860,
            "segments": None,
            "cluster": fn.ClusterConfig(
                method="hierarchical",
                solver=ImplementedSolvers.STANDARD_SOLVER.value,
            ),
        },
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert len(results[2020].typicalPeriods) == 2


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
    fn.xrIO.writeEnergySystemModelToNetCDF(
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
    loaded = fn.xrIO.readNetCDFtoEnergySystemModel(
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


def test_resume_skips_optimize_when_all_intervals_cached(rh_netcdf_cache, monkeypatch):
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


# ─── Energy system model settings a window cannot inherit ─────────────────────
#
# Some settings of the original esM do not survive being cut into windows: their
# meaning refers to the pathway as a whole. They are refused rather than silently
# reinterpreted per window.


def _minimal_esM_with_source(n_periods=3, **esMKwargs):
    esM = fn.EnergySystemModel(
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
        **esMKwargs,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=_ts(1000),
        )
    )
    return esM


def test_stochastic_model_is_refused():
    """In a stochastic model the investment periods are the scenarios of one year,
    not a pathway a window can be moved along.
    """
    esM = _minimal_esM_with_source(stochasticModel=True)
    with pytest.raises(NotImplementedError, match="stochastic"):
        rollingHorizonOptimization(
            esM=esM,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
        )


def test_pathway_balance_limit_is_refused():
    """A pathwayBalanceLimit is a budget for the whole pathway. Every window would
    enforce it again in full, so the run would emit a multiple of the budget.
    """
    pathwayBalanceLimit = pd.DataFrame(
        index=["CO2 limit"], columns=["PerfectLand", "lowerBound"], data=[[100, False]]
    )
    esM = _minimal_esM_with_source(pathwayBalanceLimit=pathwayBalanceLimit)
    with pytest.raises(NotImplementedError, match="pathwayBalanceLimit"):
        rollingHorizonOptimization(
            esM=esM,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
        )


def test_annuity_perpetuity_is_refused():
    """The annuityPerpetuity setting refers to the last investment period of the
    pathway. Per window it would refer to the last year of the window instead.
    """
    esM = _minimal_esM_with_source(annuityPerpetuity=True)
    with pytest.raises(NotImplementedError, match="annuityPerpetuity"):
        rollingHorizonOptimization(
            esM=esM,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
        )


def test_none_of_the_refused_settings_blocks_a_plain_model():
    """The three guards above must not trip on a model that sets none of them."""
    results = rollingHorizonOptimization(
        esM=_minimal_esM_with_source(),
        numberOfInvestmentPeriodsForRollingHorizon=2,
        timeSeriesAggregation=False,
    )
    assert sorted(results) == [2020, 2025]


# ─── Input the rolling horizon has to accept ──────────────────────────────────


def test_already_aggregated_esm_can_be_rolled():
    """Aggregating temporally and then optimizing is the normal FINE workflow, so an
    esM whose time series are already clustered must be accepted. exportToDict adds
    the aggregated* parameters for such a model, which are not constructor arguments
    - dictIO.importFromDict drops them for the same reason.
    """
    esM = _build_esM()
    esM.aggregateTemporally(n_clusters=1, period_duration=7860, segments=None)
    assert esM.isTimeSeriesDataClustered

    results = rollingHorizonOptimization(
        esM=esM,
        scenario_name="test_pre_aggregated",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert sorted(results) == [2020, 2025, 2030]


def test_rolling_horizon_optimization_is_exported_by_the_package():
    """The module is reachable as fn.rollingHorizonOptimization, like the other
    expansion modules (fn.optimizeTSAmultiStage).
    """
    assert fn.rollingHorizonOptimization is rollingHorizonOptimization


# ─── Transmission components ──────────────────────────────────────────────────


def _build_transmission_esM():
    """Two locations connected by a transmission line, with the demand in the
    location that cannot generate, so the line has to be built.
    """
    esM = fn.EnergySystemModel(
        locations={"north", "south"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            locationalEligibility=pd.Series({"north": 1, "south": 0}),
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Line",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=100,
            interestRate=0.02,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                {"north": [0.0, 0.0], "south": [1000.0, 1000.0]}
            ),
        )
    )
    return esM


@pytest.fixture(scope="module")
def rh_results_transmission():
    return rollingHorizonOptimization(
        esM=_build_transmission_esM(),
        scenario_name="test_transmission",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )


def test_transmission_component_summary_is_built(rh_results_transmission):
    """A rolling horizon run of a model with a transmission component must produce
    its optimization summary. The summary spans every location pair, including the
    self-pairs, while a 2-dim component's interestRate is indexed by the connections
    it actually has - so NPVcontributionRH must not look its interest rate up per
    column (which raised a KeyError on every pair the component does not connect).
    """
    summary = rh_results_transmission[2020].getOptimizationSummary(
        "TransmissionModel", ip=2020, outputLevel=0
    )
    assert not summary.empty
    assert "NPVcontributionRH" in summary.index.get_level_values("Property")


def test_transmission_capacity_is_carried_over_as_stock(rh_results_transmission):
    """The line built in the first window must reappear as stock of the second,
    exactly as for a 1-dim component.
    """
    commissioned = (
        rh_results_transmission[2020]
        .getOptimizationSummary("TransmissionModel", ip=2020, outputLevel=0)
        .xs(("Line", "commissioning"), level=("Component", "Property"))
        .max()
        .max()
    )
    assert commissioned > 0
    stock = rh_results_transmission[2025].getComponent("Line").stockCommissioning
    assert stock is not None and 2020 in stock
    assert stock[2020].max() == pytest.approx(commissioned)


# ─── NPVcontributionRH ────────────────────────────────────────────────────────
#
# Every window optimizes its own years, so its NPVcontribution is discounted onto
# its own start year. NPVcontributionRH re-bases that onto the first year of the
# whole pathway (esM.rollingHorizonStartYear), so that the windows' contributions
# are expressed in the same reference year and can be compared or added up.


def _npv_rows(esM, compName, year):
    summary = esM.getOptimizationSummary("SourceSinkModel", ip=year, outputLevel=0)
    npv = summary.xs(
        (compName, "NPVcontribution"), level=("Component", "Property")
    ).iloc[0]
    npvRH = summary.xs(
        (compName, "NPVcontributionRH"), level=("Component", "Property")
    ).iloc[0]
    return npv["PerfectLand"], npvRH["PerfectLand"]


def test_npv_contribution_rh_is_the_contribution_of_the_first_window(rh_results):
    """The first window starts in the pathway's own start year, so there is nothing
    to re-base and both rows are equal.
    """
    npv, npvRH = _npv_rows(rh_results[2020], "Source_cheap_then_expensive", 2020)
    assert npvRH == pytest.approx(npv)


def test_npv_contribution_rh_discounts_later_windows_onto_the_start_year(rh_results):
    """A later window is discounted by its distance to the pathway's start year, at
    the component's own interest rate. The (1 + interestRate) convention factor of
    utils.discountFactor is contained in both rows and cancels.
    """
    esM_2030 = rh_results[2030]
    interestRate = esM_2030.getComponent("Source_cheap_then_expensive").interestRate[
        "PerfectLand"
    ]
    assert esM_2030.rollingHorizonStartYear == 2020
    for year in (2030, 2035):
        npv, npvRH = _npv_rows(esM_2030, "Source_cheap_then_expensive", year)
        assert npvRH == pytest.approx(npv / (1 + interestRate) ** (2030 - 2020))


def test_npv_contribution_rh_is_absent_without_a_rolling_horizon():
    """A stand-alone model does not report the row at all."""
    esM = _minimal_esM_with_source()
    esM.optimize(timeSeriesAggregation=False)
    summary = esM.getOptimizationSummary("SourceSinkModel", ip=2020, outputLevel=0)
    properties = summary.index.get_level_values("Property")
    assert "NPVcontribution" in properties
    assert "NPVcontributionRH" not in properties


# ─── Pass-through settings dicts ──────────────────────────────────────────────
#
# optimize, writeOptimizationOutputToExcel and writeEnergySystemModelToNetCDF are
# reachable through their own settings dict each, so that callers are not limited
# to the arguments rollingHorizonOptimization happens to name itself. The
# arguments it determines itself are refused instead of colliding as duplicate
# keyword arguments deep inside the callee.


@pytest.mark.parametrize(
    "settingsName, settings, reserved",
    [
        ("optimizeSettings", {"solver": "glpk"}, "solver"),
        (
            "optimizeSettings",
            {"timeSeriesAggregation": True},
            "timeSeriesAggregation",
        ),
        ("excelOutputSettings", {"investmentPeriod": 2020}, "investmentPeriod"),
        ("netCDFOutputSettings", {"groupPrefix": "2020"}, "groupPrefix"),
    ],
)
def test_reserved_settings_are_refused(settingsName, settings, reserved):
    esM = _minimal_esM_with_source()
    with pytest.raises(ValueError, match=reserved):
        rollingHorizonOptimization(
            esM=esM,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
            **{settingsName: settings},
        )


def test_optimize_settings_reach_optimize():
    """A setting of EnergySystemModel.optimize that rollingHorizonOptimization does
    not name itself is reachable through optimizeSettings.
    """
    esM = _minimal_esM_with_source()
    results = rollingHorizonOptimization(
        esM=esM,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        timeSeriesAggregation=False,
        optimizeSettings={"includePerformanceSummary": True},
    )
    for esM_window in results.values():
        assert esM_window.performanceSummary is not None


def _written_summary_rows(dir_path, scenario_name):
    loaded = fn.xrIO.readNetCDFtoEnergySystemModel(
        str(_shared_netcdf_path(dir_path, scenario_name)), groupPrefix="2020"
    )
    return len(loaded.getOptimizationSummary("SourceSinkModel", ip=2020))


def test_netcdf_output_settings_reach_the_writer(tmp_path):
    """The optSumOutputLevel argument of writeEnergySystemModelToNetCDF is not named by
    rollingHorizonOptimization itself; it is reachable through netCDFOutputSettings and
    decides what the written file holds. Level 2 drops the summary rows that are zero or
    empty everywhere, level 0 keeps them.
    """
    rowsPerLevel = {}
    for optSumOutputLevel in (0, 2):
        scenario_name = f"netcdf_settings_{optSumOutputLevel}"
        rollingHorizonOptimization(
            esM=_minimal_esM_with_source(),
            scenario_name=scenario_name,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
            writeNetCDFOutput=True,
            resultExportPath=str(tmp_path),
            netCDFOutputSettings={"optSumOutputLevel": optSumOutputLevel},
        )
        rowsPerLevel[optSumOutputLevel] = _written_summary_rows(tmp_path, scenario_name)

    assert rowsPerLevel[2] > 0
    assert rowsPerLevel[2] < rowsPerLevel[0]


# ─── Excel output ─────────────────────────────────────────────────────────────


def _excel_files(dir_path):
    return sorted(path.name for path in dir_path.glob("*.xlsx"))


def test_excel_output_writes_one_file_per_exported_year(tmp_path):
    """Every window except the last exports its first year; the last exports all of
    its years. With 4 investment periods and a window of 2 that is 2020, 2025 and
    then 2030 + 2035.
    """
    rollingHorizonOptimization(
        esM=_build_esM(),
        scenario_name="excel",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        writeExcelOutput=True,
        resultExportPath=str(tmp_path),
    )
    assert _excel_files(tmp_path) == [
        "excel_rollingHorizon_2020.xlsx",
        "excel_rollingHorizon_2025.xlsx",
        "excel_rollingHorizon_2030.xlsx",
        "excel_rollingHorizon_2035.xlsx",
    ]


def test_myopic_excel_output_does_not_overwrite_itself(tmp_path):
    """A myopic window holds a single investment period. The exported year has to be
    part of the file name for those too, or every window would write the same file
    and only the last one would survive.
    """
    rollingHorizonOptimization(
        esM=_build_esM(),
        scenario_name="excel_myopic",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=1,
        writeExcelOutput=True,
        resultExportPath=str(tmp_path),
    )
    assert _excel_files(tmp_path) == [
        "excel_myopic_rollingHorizon_2020.xlsx",
        "excel_myopic_rollingHorizon_2025.xlsx",
        "excel_myopic_rollingHorizon_2030.xlsx",
        "excel_myopic_rollingHorizon_2035.xlsx",
    ]


def test_excel_output_settings_reach_the_writer(tmp_path):
    """The optSumOutputLevel argument of writeOptimizationOutputToExcel is not named by
    rollingHorizonOptimization itself.
    """
    rollingHorizonOptimization(
        esM=_minimal_esM_with_source(),
        scenario_name="excel_settings",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        writeExcelOutput=True,
        resultExportPath=str(tmp_path),
        excelOutputSettings={"optSumOutputLevel": 0, "optValOutputLevel": 0},
    )
    # three investment periods and a window of two: the first window exports 2020, the
    # last one both of its years
    assert _excel_files(tmp_path) == [
        "excel_settings_rollingHorizon_2020.xlsx",
        "excel_settings_rollingHorizon_2025.xlsx",
        "excel_settings_rollingHorizon_2030.xlsx",
    ]


# ─── Stock commissioning handed from one window to the next ───────────────────


def test_stock_commissioning_is_rounded_before_it_is_handed_on():
    """A commissioning result carries the full float64 precision, which the stock
    checks of utils warn about and round to 10 digits themselves. Rounding it here
    keeps a rolling horizon run from warning about its own results.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = rollingHorizonOptimization(
            esM=_build_esM(),
            timeSeriesAggregation=False,
            numberOfInvestmentPeriodsForRollingHorizon=2,
        )
    stockWarnings = [
        str(warning.message)
        for warning in caught
        if "will be rounded to 10 digits" in str(warning.message)
    ]
    assert stockWarnings == []

    stock = results[2025].getComponent("Source_cheap_then_expensive").stockCommissioning
    assert stock[2020]["PerfectLand"] == round(stock[2020]["PerfectLand"], 10)


def test_stock_commissioning_threshold_controls_what_is_carried_over():
    """A window's commissioning is only handed on as stock if it exceeds the
    threshold, which lets the caller decide how much solver noise counts as zero.
    """
    results = rollingHorizonOptimization(
        esM=_build_esM(),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        stockCommissioningThreshold=1e9,
    )
    stock = results[2025].getComponent("Source_cheap_then_expensive").stockCommissioning
    assert stock is None or 2020 not in stock


def test_outdated_stock_is_pruned_without_new_commissioning():
    """Pruning must not depend on the previous window having commissioned anything:
    an entry that has outlived its technical lifetime has to go either way. Here the
    stock of 2015 is given externally and nothing is ever built on top of it.
    """
    esM = _minimal_esM(4)
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=9,
            technicalLifetime=9,
            stockCommissioning={2015: pd.Series({"PerfectLand": 1.0})},
            commissioningFix={year: pd.Series({"PerfectLand": 0.0}) for year in _YEARS},
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=_ts(0),
        )
    )

    results = rollingHorizonOptimization(
        esM=esM,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    # 2015 has outlived its technical lifetime of 9 years by the window starting in
    # 2025 (2015 < 2025 - 9), even though no window commissioned anything
    stock = results[2025].getComponent("Src").stockCommissioning
    assert stock is None or 2015 not in stock


def test_investment_period_parameters_are_filtered_whatever_their_key_order():
    """A parameter given per investment period describes the same years no matter
    which order they were written down in, so the window must be scoped down to its
    own years either way.
    """
    esM = _minimal_esM(3)
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity={year: 1000 for year in reversed([2020, 2025, 2030])},
            interestRate=0.02,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=_ts(1000),
        )
    )

    results = rollingHorizonOptimization(
        esM=esM,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert results[2025].investmentPeriodNames == [2025, 2030]


# â”€â”€â”€ Parameters given per investment period â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# A window only holds its own investment periods, so every parameter given per
# investment period has to be scoped down to them. Which years a window's copy may
# carry differs by parameter: the ones describing capacity that is commissioned once
# and paid for over its lifetime (_STOCK_YEAR_PARAMETERS) are given for the stock
# years too, every other one for the investment periods alone. Getting that wrong is
# invisible until a stock year coincides with an investment period of the pathway,
# which is exactly what the rolling horizon itself produces: from the second window
# on, the stock is keyed by the year a previous window commissioned it in.


def _esM_with_committed_source(sourceKwargs=None, storageKwargs=None):
    """Build a pathway whose components are forced to commission in every window.

    commissioningMin is a plain Series, not a dict per investment period, so that
    forcing the commissioning does not itself introduce the kind of parameter these
    tests are about.
    """
    esM = _minimal_esM(4)
    source = {
        "name": "Src",
        "commodity": "electricity",
        "hasCapacityVariable": True,
        "investPerCapacity": 1000,
        "interestRate": 0.02,
        "economicLifetime": 20,
        "technicalLifetime": 20,
        "commissioningMin": pd.Series({"PerfectLand": 1.0}),
    }
    source.update(sourceKwargs or {})
    esM.add(fn.Source(esM=esM, **source))
    if storageKwargs is not None:
        esM.add(
            fn.Storage(
                esM=esM,
                name="Bat",
                commodity="electricity",
                hasCapacityVariable=True,
                investPerCapacity=100,
                interestRate=0.02,
                economicLifetime=20,
                technicalLifetime=20,
                commissioningMin=pd.Series({"PerfectLand": 1.0}),
                **storageKwargs,
            )
        )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix={year: _ts(1000) for year in _YEARS},
        )
    )
    return esM


@pytest.mark.parametrize(
    "sourceKwargs",
    [
        pytest.param(
            {"operationRateMax": {year: _ts(1.0) for year in _YEARS}},
            id="operationRateMax",
        ),
        pytest.param(
            {"capacityMax": {year: pd.Series({"PerfectLand": 1e6}) for year in _YEARS}},
            id="capacityMax",
        ),
        pytest.param(
            {"commodityCost": {year: 0.05 for year in _YEARS}}, id="commodityCost"
        ),
        pytest.param(
            {"opexPerOperation": {year: 0.5 for year in _YEARS}},
            id="opexPerOperation",
        ),
        pytest.param(
            {"investPerCapacity": {year: 1000 for year in _YEARS}},
            id="investPerCapacity",
        ),
    ],
)
def test_investment_period_parameters_survive_internally_generated_stock(sourceKwargs):
    """A component that both carries a per-investment-period parameter and commissions
    something has stock keyed by a year that is also a key of that parameter, from the
    second window on. Only the parameters that are given per stock year as well may
    keep it; handing it to any other one makes rebuilding the component fail.
    """
    results = rollingHorizonOptimization(
        esM=_esM_with_committed_source(sourceKwargs),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert sorted(results) == [2020, 2025, 2030]
    # the stock that makes this case non-trivial is really there
    assert 2020 in results[2025].getComponent("Src").stockCommissioning


@pytest.mark.parametrize(
    "storageKwargs",
    [
        pytest.param(
            {"opexPerChargeOperation": {year: 0.01 for year in _YEARS}},
            id="opexPerChargeOperation",
        ),
        pytest.param(
            {"chargeOpRateMax": {year: _ts(1.0) for year in _YEARS}},
            id="chargeOpRateMax",
        ),
    ],
)
def test_storage_investment_period_parameters_survive_internally_generated_stock(
    storageKwargs,
):
    """The same for a Storage, whose charge and discharge parameters are named so that
    a rule going by the parameter name alone does not recognize them.
    """
    results = rollingHorizonOptimization(
        esM=_esM_with_committed_source(storageKwargs=storageKwargs),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert 2020 in results[2025].getComponent("Bat").stockCommissioning


def test_stock_year_parameters_keep_their_stock_years():
    """Capacity commissioned before the window is charged with investPerCapacity, so a
    window's copy of it must cover the stock years on top of its own investment periods.
    """
    results = rollingHorizonOptimization(
        esM=_esM_with_committed_source(
            {"investPerCapacity": {year: 1000 for year in _YEARS}}
        ),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert sorted(results[2025].getComponent("Src").investPerCapacity) == [
        2020,
        2025,
        2030,
    ]
    # the operation side of the same component is scoped to the window alone
    assert results[2025].investmentPeriodNames == [2025, 2030]


def test_parameters_that_are_dicts_for_another_reason_are_left_alone():
    """The pwlcfParameters dict is keyed by the cost function, not by year. Filtering it
    as if it were given per investment period leaves an empty dict behind, which silently
    disables the piecewise linear cost function of the rebuilt component.
    """
    esM = _minimal_esM(4)
    pwlcfParameters = {
        "etlParameters": {
            "initCost": 1000,
            "learningRate": 0.18,
            "initCapacity": 10,
            "maxCapacity": 50,
            "noSegments": 4,
        }
    }
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
            pwlcfParameters=pwlcfParameters,
        )
    )
    _, compDict = fn.dictIO.exportToDict(esM)
    compEntry = copy.deepcopy(dict(compDict))["Source"]["Src"]

    _filterComponentParametersForInterval(compEntry, [2025, 2030], [2020], esM)

    assert compEntry["pwlcfParameters"] == pwlcfParameters


def test_empty_commodity_conversion_factors_are_left_alone():
    """An empty commodityConversionFactors dict has no first key to inspect."""
    esM = _minimal_esM(4)
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Conv",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={},
            hasCapacityVariable=True,
            investPerCapacity=1,
            interestRate=0.02,
            economicLifetime=10,
        )
    )
    _, compDict = fn.dictIO.exportToDict(esM)
    compEntry = copy.deepcopy(dict(compDict))["Conversion"]["Conv"]

    _filterComponentParametersForInterval(compEntry, [2025, 2030], [2020], esM)

    assert compEntry["commodityConversionFactors"] == {}


def test_endogenous_technological_learning_is_refused():
    """A learning curve accumulates over the pathway, while every window is built and
    solved on its own, so it would restart in each of them.
    """
    esM = _minimal_esM_with_source(n_periods=3)
    esM.add(
        fn.Source(
            esM=esM,
            name="Learning",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
            pwlcfParameters={
                "etlParameters": {
                    "initCost": 1000,
                    "learningRate": 0.18,
                    "initCapacity": 10,
                    "maxCapacity": 50,
                    "noSegments": 4,
                }
            },
        )
    )
    with pytest.raises(NotImplementedError, match="pwlcfParameters"):
        rollingHorizonOptimization(
            esM=esM,
            timeSeriesAggregation=False,
            numberOfInvestmentPeriodsForRollingHorizon=2,
        )


# â”€â”€â”€ Costs across the windows â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _pathway_years(results):
    """Return the years of the pathway, and which window is responsible for each of them.

    The windows overlap, so a year is reported by every window that spans it. Exactly
    one of them owns it: every window its own first year, and the last window all of
    the years it spans. This is the selection the Excel output writes, and the one that
    adds up to the pathway without counting an overlap twice.
    """
    lastWindow = max(results)
    return [
        (startYear, year)
        for startYear in sorted(results)
        for year in (
            results[startYear].investmentPeriodNames
            if startYear == lastWindow
            else [startYear]
        )
    ]


def _fixed_commissioning_esM(n_periods=4):
    """Build a pathway whose commissioning is fixed, so that a rolling horizon and a
    perfect foresight run of it differ in nothing but how the costs are booked.
    """
    esM = _minimal_esM(n_periods)
    years = esM.investmentPeriodNames
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            hasCapacityVariable=True,
            commissioningFix={
                year: pd.Series({"PerfectLand": 1.0 if year == years[0] else 0.0})
                for year in years
            },
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
            technicalLifetime=20,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Slack",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerOperation=1,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix={year: _ts(1000) for year in years},
        )
    )
    return esM


def _npv(esM, year, compName, propertyName):
    summary = esM.getOptimizationSummary("SourceSinkModel", ip=year, outputLevel=0)
    row = summary.xs((compName, propertyName), level=("Component", "Property"))
    return float(row.iloc[0]["PerfectLand"])


@pytest.mark.parametrize("window", [1, 2, 3])
def test_chain_books_the_same_net_present_value_as_perfect_foresight(window):
    """No cost is lost or counted twice between the windows.

    A window charges an annuity for the investment periods it spans, and the capacity it
    commissions keeps being charged in the following windows, where it arrives as stock.
    Summed over the years the windows own, and re-based onto the pathway's start year by
    NPVcontributionRH, that has to reproduce what perfect foresight books for the same
    commissioning decisions - which commissioningFix pins down here.
    """
    perfectForesight = _fixed_commissioning_esM()
    perfectForesight.optimize(timeSeriesAggregation=False)
    expected = sum(
        _npv(perfectForesight, year, "Src", "NPVcontribution")
        for year in perfectForesight.investmentPeriodNames
    )

    results = rollingHorizonOptimization(
        esM=_fixed_commissioning_esM(),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=window,
    )
    booked = sum(
        _npv(results[startYear], year, "Src", "NPVcontributionRH")
        for startYear, year in _pathway_years(results)
    )

    assert booked == pytest.approx(expected, rel=1e-6)


# â”€â”€â”€ Excel output â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_excel_export_accepts_a_numpy_investment_period(tmp_path):
    """A year read off a pandas index is a numpy integer, which is just as valid a year
    as an int here.
    """
    esM = _minimal_esM_with_source(n_periods=2)
    esM.optimize(timeSeriesAggregation=False)
    year = pd.Index(esM.investmentPeriodNames)[0]
    assert not isinstance(year, int)

    writeOptimizationOutputToExcel(
        esM,
        outputFileName=str(tmp_path / "numpyYear"),
        investmentPeriod=year,
    )

    assert [path.name for path in tmp_path.glob("*.xlsx")] == ["numpyYear_2020.xlsx"]


def test_excel_export_refuses_a_boolean_investment_period():
    """A bool is an int, but not a year."""
    esM = _minimal_esM_with_source(n_periods=2)
    with pytest.raises(ValueError, match="must be type int"):
        writeOptimizationOutputToExcel(
            esM, outputFileName="unused", investmentPeriod=True
        )


def test_every_netcdf_writer_argument_is_reachable(tmp_path, monkeypatch):
    """Settings are passed on unchanged, so an argument of the writer that
    rollingHorizonOptimization does not name itself reaches it - including
    includeShadowPrices and shadowPriceConstraintStr, which it cannot demonstrate end to
    end because writeEnergySystemModelToNetCDF fails on them for any energy system model
    (independently of the rolling horizon). The call is recorded rather than executed, so
    that this stays a statement about the pass-through and not about the writer.
    """
    seen = []

    def _recordingWriter(esM, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(
        "fine.expansionModules.rollingHorizon.writeEnergySystemModelToNetCDF",
        _recordingWriter,
    )

    rollingHorizonOptimization(
        esM=_minimal_esM_with_source(),
        scenario_name="passthrough",
        numberOfInvestmentPeriodsForRollingHorizon=2,
        timeSeriesAggregation=False,
        writeNetCDFOutput=True,
        resultExportPath=str(tmp_path),
        netCDFOutputSettings={
            "optSumOutputLevel": 1,
            "includeShadowPrices": True,
            "shadowPriceConstraintStr": "commodityBalanceConstraint",
        },
    )

    assert seen, "the writer was never called"
    for kwargs in seen:
        assert kwargs["optSumOutputLevel"] == 1
        assert kwargs["includeShadowPrices"] is True
        assert kwargs["shadowPriceConstraintStr"] == "commodityBalanceConstraint"
        # the three the function determines itself are still its own
        assert kwargs["overwriteExisting"] is False
        assert kwargs["groupPrefix"] in ("2020", "2025")


# â”€â”€â”€ The module's assumptions about FINE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _parametersKeyedByStockYears(esM, compName):
    """Return the parameters whose processed counterpart is keyed by the stock years.

    A component builds the per-investment-period parameters that are charged for
    capacity commissioned before the first modeled year over
    ``processedStockYears + esM.investmentPeriods``, and every other one over the
    investment periods alone. On a component that has stock, the two are therefore told
    apart by the keys of the processed counterpart.
    """
    component = esM.getComponent(compName)
    keyedByStockYears = set()
    for name in inspect.getfullargspec(type(component).__init__).args:
        processed = getattr(component, f"processed{name[:1].upper()}{name[1:]}", None)
        if isinstance(processed, dict) and set(processed) > set(esM.investmentPeriods):
            keyedByStockYears.add(name)
    return keyedByStockYears


def _esM_with_source_stock():
    esM = _minimal_esM(3)
    esM.add(
        fn.Source(
            esM=esM,
            name="WithStock",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
            technicalLifetime=20,
            stockCommissioning={2015: pd.Series({"PerfectLand": 1.0})},
        )
    )
    return esM


def _esM_with_transmission_stock():
    """Build a model whose Transmission has stock.

    A Transmission keeps its investment parameters under preprocessed* as well, so it is
    worth checking separately that the processed* ones still carry the signal.
    """
    esM = fn.EnergySystemModel(
        locations={"PerfectLand", "OtherLand"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )
    esM.add(
        fn.Transmission(
            esM=esM,
            name="WithStock",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=50,
            interestRate=0.02,
            economicLifetime=20,
            technicalLifetime=20,
            stockCommissioning={
                2015: pd.Series(
                    {"PerfectLand_OtherLand": 1.0, "OtherLand_PerfectLand": 1.0}
                )
            },
        )
    )
    return esM


@pytest.mark.parametrize(
    "esMWithStock",
    [
        pytest.param(_esM_with_source_stock, id="Source"),
        pytest.param(_esM_with_transmission_stock, id="Transmission"),
    ],
)
def test_stock_year_parameters_still_match_what_fine_builds(esMWithStock):
    """_STOCK_YEAR_PARAMETERS is written out rather than derived, so it can go stale.

    The list cannot be derived where it is used - the components it filters come from
    the original pathway model, which normally has no stock, and without stock the
    runtime signal is silent for every parameter (see the test below). It can be checked
    here though, against a component that does have stock. This fails if a parameter is
    added to or removed from the ``processedStockYears + esM.investmentPeriods`` calls
    in component.py or transmission.py.
    """
    esM = esMWithStock()

    keyedByStockYears = _parametersKeyedByStockYears(esM, "WithStock")

    assert keyedByStockYears, (
        "the runtime signal is silent - the component has no stock"
    )
    assert keyedByStockYears == set(_STOCK_YEAR_PARAMETERS)


def test_the_runtime_signal_is_silent_without_stock():
    """Why _STOCK_YEAR_PARAMETERS cannot be derived where it is used.

    The filter runs against the original pathway model's components, which normally
    carry no stock. Deriving the set from them would answer "no stock years" for every
    parameter and strip the years the previous windows commissioned in.
    """
    esM = _minimal_esM_with_source(n_periods=3)

    assert _parametersKeyedByStockYears(esM, "Src") == set()


def test_non_constructor_parameters_are_dropped_from_a_clustered_model():
    """Non-constructor keys are dropped by the constructor's signature, not by name.

    exportToDict adds the aggregated time series of a clustered model on top of the
    constructor arguments, and a component cannot be rebuilt with them. They are
    recognized by the constructor's signature rather than by their name, so that
    anything else exportToDict may add later is dropped as well.
    """
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=8,
        hoursPerTimeStep=1095,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Src",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame({"PerfectLand": np.linspace(0.1, 0.9, 8)}),
            investPerCapacity=1000,
            interestRate=0.02,
            economicLifetime=20,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame({"PerfectLand": np.full(8, 100.0)}),
        )
    )
    esM.aggregateTemporally(n_clusters=2, period_duration=4 * 1095)
    _, compDict = fn.dictIO.exportToDict(esM)
    assert any(name.startswith("aggregated") for name in compDict["Source"]["Src"])

    # the first interval, so that no previous window's results are looked up
    built = _buildIntervalComponentDict(
        compDict,
        [2020, 2025],
        [[2020, 2025], [2025, 2030]],
        5,
        {},
        esM,
        {
            classname: {comp: None for comp in compDict[classname]}
            for classname in compDict
        },
        1e-5,
    )

    for classname, comps in built.items():
        constructorArguments = set(
            inspect.getfullargspec(getattr(fn, classname).__init__).args
        )
        for compName, compEntry in comps.items():
            leftOver = set(compEntry) - constructorArguments
            assert not leftOver, f"{classname} '{compName}' kept {sorted(leftOver)}"
