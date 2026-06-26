import pytest
import fine as fn
from fine.expansionModules.rollingHorizon import rollingHorizonOptimization
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


def _build_esM():
    """Construct an esM whose components exercise all rolling horizon code paths.

    Source_cheap_then_expensive : stock accumulation + non-PerOperation dict param filtering
    Source_expensive_then_cheap : zero-commissioning guard (line 96 False branch)
    Source_short_lifetime       : outdated stock cleanup (lines 113-132); dedicated heat commodity
    Electrolyzer                : ip-dependent CCF (line 162)
    FuelCell                    : time-constant CCF (line 196 else:pass)
    EDemand / H2Demand          : electricity and hydrogen sinks
    HeatDemand                  : growing demand forces new commissioning every period
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
    esM.add(fn.Source(
        esM=esM,
        name="Source_cheap_then_expensive",
        commodity="electricity",
        hasCapacityVariable=True,
        investPerCapacity={2020: 1000, 2025: 900, 2030: 800, 2035: 700},
        interestRate=0.02,
        opexPerOperation={2020: 1, 2025: 1, 2030: 1, 2035: 100},
        economicLifetime=15,
        technicalLifetime=15,
    ))

    # opex=100 in 2020-2030 → optimizer commissions 0 → covers line 96 False branch
    esM.add(fn.Source(
        esM=esM,
        name="Source_expensive_then_cheap",
        commodity="electricity",
        hasCapacityVariable=True,
        investPerCapacity=1e3,
        interestRate=0.02,
        opexPerOperation={2020: 100, 2025: 100, 2030: 100, 2035: 1},
        economicLifetime=15,
    ))

    # Dedicated heat commodity forces commissioning in every period via growing HeatDemand.
    # technicalLifetime=9 → cleanup condition 2020 < 2030-9=2021 fires in [2030,2035].
    # Covers lines 113-132 (non-empty outdatedStockYears path).
    esM.add(fn.Source(
        esM=esM,
        name="Source_short_lifetime",
        commodity="heat",
        hasCapacityVariable=True,
        investPerCapacity=1e3,
        interestRate=0.02,
        opexPerOperation={2020: 1, 2025: 1, 2030: 1, 2035: 1},
        economicLifetime=9,
        technicalLifetime=9,
    ))

    # ip-dependent CCF: firstKey is a year → covers line 162 branch
    esM.add(fn.Conversion(
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
    ))

    # tuple-keyed CCF: firstKey is (commisYear, opYear) → covers lines 173-195 branch.
    # Exactly 9 valid pairs for technicalLifetime=15 across [2020,2025,2030,2035].
    # Varying efficiency per commissioning year makes FINE set isCommisDepending=True.
    esM.add(fn.Conversion(
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
    ))

    # time-constant CCF: firstKey is a commodity string → covers line 196 else:pass branch
    esM.add(fn.Conversion(
        esM=esM,
        name="FuelCell",
        physicalUnit=r"kW$_{H_{2},LHV}$",
        commodityConversionFactors={"hydrogen": -1, "electricity": 0.5},
        hasCapacityVariable=True,
        investPerCapacity=300,
        interestRate=0.02,
        economicLifetime=15,
    ))

    esM.add(fn.Sink(
        esM=esM,
        name="EDemand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix={
            2020: _ts(2190),
            2025: _ts(4380),
            2030: _ts(6570),
            2035: _ts(8760),
        },
    ))

    esM.add(fn.Sink(
        esM=esM,
        name="H2Demand",
        commodity="hydrogen",
        hasCapacityVariable=False,
        operationRateFix={year: _ts(100) for year in _YEARS},
    ))

    # Growing demand forces new heat capacity in every period, guaranteeing commissioning.
    esM.add(fn.Sink(
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
    ))

    return esM


@pytest.fixture(scope="module")
def rh_results(tmp_path_factory):
    esM = _build_esM()
    tmp = tmp_path_factory.mktemp("rh")
    return rollingHorizonOptimization(
        esM=esM,
        resultExportPath=str(tmp),
        scenario_name="test",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        numberOfTimeStepsPerPeriod=1,
        numberOfSegments=1,
        numberOfTypicalPeriods=1,
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

def test_raises_on_single_investment_period(tmp_path):
    """Line 26: numberOfInvestmentPeriods < 2 raises ValueError."""
    with pytest.raises(ValueError, match="At least two"):
        rollingHorizonOptimization(
            esM=_minimal_esM(1),
            resultExportPath=str(tmp_path),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=1,
        )


def test_raises_when_window_not_smaller_than_periods(tmp_path):
    """Line 28: window >= numberOfInvestmentPeriods raises ValueError."""
    with pytest.raises(ValueError, match="at least one more"):
        rollingHorizonOptimization(
            esM=_minimal_esM(4),
            resultExportPath=str(tmp_path),
            scenario_name="err",
            numberOfInvestmentPeriodsForRollingHorizon=4,
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
    stock = rh_results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning
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
    stock = rh_results[2030].getComponent("Source_cheap_then_expensive").stockCommissioning
    assert stock[2020]["PerfectLand"] == commis_2020
    assert stock[2025]["PerfectLand"] == commis_2025


def test_zero_commissioning_not_added_to_stock(rh_results):
    """Line 96 False branch: zero commissioning in 2020 produces no stock entry in [2025,2030]."""
    stock = rh_results[2025].getComponent("Source_expensive_then_cheap").stockCommissioning
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
