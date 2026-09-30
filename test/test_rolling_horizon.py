"""Tests for fine.expansionModules.rollingHorizon.

The behaviour of a rolling horizon shows in more than one window, so the runs are solved
once per module and shared: `rollingResults` (the pathway with a window of two),
`myopicResults` (the same pathway with a window of one) and `cachedRun` (the netCDF cache
the resume tests build on). Everything that does not need a solved pathway - argument
validation, parameter filtering, cache validation - is tested against the module's own
helpers instead.
"""

import copy
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import fine as fn
from fine.expansionModules.rollingHorizon import (
    _STOCK_YEAR_PARAMETERS,
    _buildIntervalComponentDict,
    _cachedGroupExists,
    _cachedIntervalChainMismatches,
    _cachedIntervalConfigMismatches,
    _filterComponentParametersForInterval,
    _stockCommissioningDiffers,
    rollingHorizonOptimization,
)
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from fine.utils import ImplementedSolvers
from test.conftest import build_perfectForesight_test_esM

# Investment periods of conftest's perfectForesight_test_esM.
PATHWAY_YEARS = [2020, 2025, 2030, 2035, 2040]

# Scenario name the shared netCDF file of the cached run is named after.
CACHE_NAME = "rollingHorizon_cache"

PWLCF_PARAMETERS = {
    "etlParameters": {
        "initCost": 1000,
        "learningRate": 0.18,
        "initCapacity": 10,
        "maxCapacity": 50,
        "noSegments": 4,
    }
}


# --- Models ------------------------------------------------------------------------


def pathwayEsM():
    """Build the transformation pathway of conftest, extended by what a rolling horizon
    has to carry across its windows.

    perfectForesight_test_esM already brings two locations, five investment periods, a
    Source with a per-location profile and a demand given per investment period. Added
    here, one component per code path that the base model leaves untouched:

    Electrolyzer : commodityConversionFactors given per investment period, plus one
                   parameter that is given per stock year as well (investPerCapacity) and
                   one that is not (opexPerOperation)
    H2Demand     : the demand that makes the electrolyzer run
    Line         : a 2-dim component, whose stock is indexed by connections rather than by
                   locations
    Backup       : priced out, so that it never commissions anything and never becomes
                   stock
    """
    esM = build_perfectForesight_test_esM()
    esM.verboseLogLevel = 0
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                year: {"electricity": -1, "hydrogen": 0.6 + 0.01 * index}
                for index, year in enumerate(PATHWAY_YEARS)
            },
            hasCapacityVariable=True,
            investPerCapacity={
                year: 500 - 10 * index for index, year in enumerate(PATHWAY_YEARS)
            },
            opexPerOperation={year: 0.01 for year in PATHWAY_YEARS},
            interestRate=0.02,
            economicLifetime=15,
            technicalLifetime=15,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                {"PerfectLand": [500.0, 500.0], "ForesightLand": [0.0, 0.0]}
            ),
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
            technicalLifetime=20,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Backup",
            commodity="electricity",
            hasCapacityVariable=True,
            investPerCapacity=1e9,
            interestRate=0.02,
            economicLifetime=20,
        )
    )
    return esM


def simplePathwayEsM(
    numberOfInvestmentPeriods=3, sourceKwargs=None, demand=1000.0, **esMKwargs
):
    """Build a one location pathway of a single source and a fixed demand.

    Used wherever the test is about the rolling horizon itself rather than about the model
    it is applied to - argument validation, output files, pass-through settings - so that
    those do not pay for solving the full pathway above.
    """
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=numberOfInvestmentPeriods,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=0,
        **esMKwargs,
    )
    source = {
        "name": "Src",
        "commodity": "electricity",
        "hasCapacityVariable": True,
        "investPerCapacity": 1000,
        "interestRate": 0.02,
        "economicLifetime": 20,
        "technicalLifetime": 20,
    }
    source.update(sourceKwargs or {})
    esM.add(fn.Source(esM=esM, **source))
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame({"PerfectLand": [demand, demand]}),
        )
    )
    return esM


def fixedCommissioningEsM(numberOfInvestmentPeriods=4):
    """Build a pathway whose commissioning is pinned, so that a rolling horizon and a
    perfect foresight run of it differ in nothing but how the costs are booked.

    The slack source keeps every window feasible without being able to invest, so the only
    capacity in the model is the one commissioningFix prescribes.
    """
    years = [2020 + 5 * step for step in range(numberOfInvestmentPeriods)]
    esM = simplePathwayEsM(
        numberOfInvestmentPeriods,
        sourceKwargs={
            "commissioningFix": {
                year: pd.Series({"PerfectLand": 1.0 if year == years[0] else 0.0})
                for year in years
            }
        },
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
    return esM


def co2PathwayEsM():
    """Build a pathway whose CO2 budget tightens from one investment period to the next.

    A cheap emitting gas plant is preferred over expensive wind for as long as the budget
    allows it. The budget is a balanceLimit per investment period, a setting of the energy
    system model itself, which the windows have to be scoped down to just like a
    component's parameters.
    """
    reductionTargets = {2020: 0.25, 2025: 0.5, 2030: 1.0}
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
        balanceLimit={
            year: pd.DataFrame(
                index=["CO2 limit"],
                columns=["Total", "lowerBound"],
                data=[[-100 * (1 - target), True]],
            )
            for year, target in reductionTargets.items()
        },
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame({"PerfectLand": [0.5, 0.2]}),
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
            operationRateFix=pd.DataFrame({"PerfectLand": [6.0, 4.0]}),
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
    return esM


# --- Shared runs and result accessors ----------------------------------------------


@pytest.fixture(scope="module")
def rollingResults():
    """Roll the pathway with a window of two."""
    return rollingHorizonOptimization(
        esM=pathwayEsM(),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )


@pytest.fixture(scope="module")
def cachedRun(tmp_path_factory):
    """Roll a pathway into a shared netCDF file, and return it with its directory.

    The netCDF round trip is what the resume tests are about, not the model behind it, so
    this uses the simple pathway: writing and reading back the full pathway of every window
    costs more than everything the resume tests assert.
    """
    cacheDir = tmp_path_factory.mktemp("rollingHorizonCache")
    results = rollingHorizonOptimization(
        esM=simplePathwayEsM(4),
        scenario_name=CACHE_NAME,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        writeNetCDFOutput=True,
        resultExportPath=str(cacheDir),
    )
    return results, cacheDir


@pytest.fixture(scope="module")
def myopicResults():
    """Roll the pathway with a window of one, i.e. myopic foresight.

    Every investment period is optimized on its own, which chains the stock through four
    handoffs instead of three.
    """
    return rollingHorizonOptimization(
        esM=pathwayEsM(),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=1,
    )


def summaryValue(esM, modelName, compName, propertyName, year):
    """Return one property of one component of a solved model, per location."""
    summary = esM.getOptimizationSummary(modelName, ip=year, outputLevel=0)
    row = summary.xs((compName, propertyName), level=("Component", "Property"))
    return row.iloc[0].astype(float)


def stockOf(results, year, compName):
    return results[year].getComponent(compName).stockCommissioning


def yearsOwnedByEachWindow(results):
    """Return which window is responsible for which year of the pathway.

    The windows overlap, so a year is reported by every window that spans it. Exactly one
    of them owns it: every window its own first year, and the last window all of the years
    it spans. This is the selection the Excel output writes, and the one that adds up to
    the pathway without counting an overlap twice.
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


def pathwayNetPresentValue(results, compName):
    """Return what the windows book for a component over the whole pathway.

    Read from the optimization summary rather than from the objective, so that it can also
    be read off a window that was loaded from the cache and therefore has no solved pyomo
    model of its own.
    """
    return sum(
        summaryValue(
            results[startYear], "SourceSinkModel", compName, "NPVcontributionRH", year
        ).sum()
        for startYear, year in yearsOwnedByEachWindow(results)
    )


# --- Arguments and models a rolling horizon refuses --------------------------------


@pytest.mark.parametrize(
    "numberOfInvestmentPeriods, window, error, message",
    [
        (1, 1, ValueError, "At least two"),
        (4, 4, ValueError, "at least one more"),
        (4, 5, ValueError, "at least one more"),
        (4, 0, ValueError, "strictly positive"),
        (4, 2.0, TypeError, "has to be an integer"),
        (4, "2", TypeError, "has to be an integer"),
        (4, None, TypeError, "has to be an integer"),
    ],
)
def test_an_impossible_window_is_refused(
    numberOfInvestmentPeriods, window, error, message
):
    """A window has to be a positive integer and leave at least one investment period of
    the pathway outside itself; a window spanning the whole pathway is perfect foresight.

    The type check runs first, so that a wrongly typed window is reported as such instead
    of as a comparison it does not support.
    """
    with pytest.raises(error, match=message):
        rollingHorizonOptimization(
            esM=simplePathwayEsM(numberOfInvestmentPeriods),
            numberOfInvestmentPeriodsForRollingHorizon=window,
        )


@pytest.mark.parametrize("option", ["writeExcelOutput", "writeNetCDFOutput", "resume"])
@pytest.mark.parametrize(
    "given, missing",
    [
        ({"scenario_name": "name"}, "resultExportPath"),
        ({"resultExportPath": "some/path"}, "scenario_name"),
    ],
)
def test_writing_results_requires_a_path_and_a_name(option, given, missing):
    """Every option that writes files needs somewhere to write them and something to name
    them after; resume needs both because it reads back what such a run wrote.
    """
    with pytest.raises(ValueError, match=missing):
        rollingHorizonOptimization(
            esM=simplePathwayEsM(4),
            numberOfInvestmentPeriodsForRollingHorizon=2,
            **{option: True},
            **given,
        )


@pytest.mark.parametrize(
    "settingsName, settings, reserved",
    [
        ("optimizeSettings", {"solver": "glpk"}, "solver"),
        ("optimizeSettings", {"timeSeriesAggregation": True}, "timeSeriesAggregation"),
        ("excelOutputSettings", {"investmentPeriod": 2020}, "investmentPeriod"),
        ("netCDFOutputSettings", {"groupPrefix": "2020"}, "groupPrefix"),
    ],
)
def test_settings_the_rolling_horizon_determines_itself_are_refused(
    settingsName, settings, reserved
):
    """The pass-through settings dicts must not carry an argument the rolling horizon
    determines itself, which would otherwise collide as a duplicate keyword argument deep
    inside the callee.
    """
    with pytest.raises(ValueError, match=reserved):
        rollingHorizonOptimization(
            esM=simplePathwayEsM(),
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
            **{settingsName: settings},
        )


@pytest.mark.parametrize(
    "esMKwargs, sourceKwargs, message",
    [
        pytest.param({"stochasticModel": True}, {}, "stochastic", id="stochasticModel"),
        pytest.param(
            {
                "pathwayBalanceLimit": pd.DataFrame(
                    index=["CO2 limit"],
                    columns=["PerfectLand", "lowerBound"],
                    data=[[100, False]],
                )
            },
            {},
            "pathwayBalanceLimit",
            id="pathwayBalanceLimit",
        ),
        pytest.param(
            {"annuityPerpetuity": True}, {}, "annuityPerpetuity", id="annuityPerpetuity"
        ),
        pytest.param(
            {},
            {"pwlcfParameters": PWLCF_PARAMETERS},
            "pwlcfParameters",
            id="pwlcfParameters",
        ),
    ],
)
def test_pathway_settings_a_window_cannot_inherit_are_refused(
    esMKwargs, sourceKwargs, message
):
    """Settings whose meaning refers to the pathway as a whole do not survive being cut
    into windows: a stochastic model's investment periods are scenarios rather than a
    pathway, a pathway budget would be enforced again in full by every window,
    annuityPerpetuity would refer to a window's last year instead of the pathway's, and a
    learning curve would restart in every window. They are refused rather than silently
    reinterpreted.
    """
    with pytest.raises(NotImplementedError, match=message):
        rollingHorizonOptimization(
            esM=simplePathwayEsM(3, sourceKwargs=sourceKwargs, **esMKwargs),
            numberOfInvestmentPeriodsForRollingHorizon=2,
            timeSeriesAggregation=False,
        )


# --- The windows the pathway is cut into -------------------------------------------


def test_windows_move_along_the_pathway_one_investment_period_at_a_time(rollingResults):
    """Five investment periods and a window of two give four windows, each starting one
    investment period after the previous one and spanning the window size.

    Every window keeps the first year of the whole pathway as rollingHorizonStartYear,
    independent of its own start year; that is the shared reference year NPVcontributionRH
    is discounted onto.
    """
    assert sorted(rollingResults) == [2020, 2025, 2030, 2035]
    for startYear, esM in rollingResults.items():
        assert esM.startYear == startYear
        assert esM.numberOfInvestmentPeriods == 2
        assert esM.investmentPeriodNames == [startYear, startYear + 5]
        assert esM.rollingHorizonStartYear == 2020


def test_the_passed_esM_is_not_modified():
    """The windows are built from an exported copy, so the caller's model comes back
    unchanged and can be optimized on its own afterwards - in particular it does not pick
    up the rollingHorizonStartYear the windows are given.
    """
    esM = simplePathwayEsM(3)
    componentsBefore = copy.deepcopy(esM.componentNames)

    results = rollingHorizonOptimization(
        esM=esM,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )

    assert esM.rollingHorizonStartYear is None
    assert esM.startYear == 2020
    assert esM.numberOfInvestmentPeriods == 3
    assert esM.componentNames == componentsBefore
    assert all(window.rollingHorizonStartYear == 2020 for window in results.values())


def test_a_user_set_rolling_horizon_start_year_is_kept():
    """An explicitly set rollingHorizonStartYear wins over the pathway's own start year."""
    results = rollingHorizonOptimization(
        esM=simplePathwayEsM(3, rollingHorizonStartYear=2010),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert all(window.rollingHorizonStartYear == 2010 for window in results.values())


# --- Stock handed from one window to the next --------------------------------------


def test_what_a_window_commissions_is_the_stock_of_the_next(rollingResults):
    """The capacity a window builds has to reappear as the next window's stock, for a
    1-dim component per location and for a 2-dim one per connection.

    It is rounded on the way, to the number of digits utils.checkAndSetStock rounds a stock
    to itself.
    """
    commissioned = summaryValue(
        rollingResults[2020], "SourceSinkModel", "PV", "commissioning", 2020
    )
    stock = stockOf(rollingResults, 2025, "PV")[2020]
    assert stock["PerfectLand"] == pytest.approx(commissioned["PerfectLand"])
    assert (stock == stock.round(10)).all()

    lineCommissioned = (
        rollingResults[2020]
        .getOptimizationSummary("TransmissionModel", ip=2020, outputLevel=0)
        .xs(("Line", "commissioning"), level=("Component", "Property"))
        .max()
        .max()
    )
    assert lineCommissioned > 0
    lineStock = stockOf(rollingResults, 2025, "Line")[2020]
    assert sorted(lineStock.index) == [
        "ForesightLand_PerfectLand",
        "PerfectLand_ForesightLand",
    ]
    assert lineStock.max() == pytest.approx(lineCommissioned)


def test_stock_accumulates_and_outdated_entries_are_dropped(rollingResults):
    """A window's stock holds every commissioning of the chain so far that is still within
    the component's technical lifetime, and nothing else.

    The line (technical lifetime 20 years) keeps what it was given in 2020 alongside what
    it built later, while the PV (technical lifetime 10 years, i.e. two investment periods)
    has its 2020 entry pruned by the window starting in 2035.
    """
    assert sorted(stockOf(rollingResults, 2025, "Line")) == [2020]
    assert {2020, 2030}.issubset(stockOf(rollingResults, 2035, "Line"))

    assert 2020 in stockOf(rollingResults, 2030, "PV")
    assert 2020 not in stockOf(rollingResults, 2035, "PV")


def test_a_component_that_builds_nothing_never_becomes_stock(rollingResults):
    """Commissioning below stockCommissioningThreshold is not carried over at all, so the
    priced-out backup source stays without stock in every window.
    """
    assert all(
        stockOf(rollingResults, year, "Backup") is None for year in rollingResults
    )


def test_outdated_stock_is_dropped_without_new_commissioning():
    """Pruning must not depend on the previous window having built something: an entry that
    has outlived its technical lifetime has to go either way. Here the stock of 2015 is
    given externally and nothing is ever built on top of it.
    """
    results = rollingHorizonOptimization(
        esM=simplePathwayEsM(
            4,
            demand=0.0,
            sourceKwargs={
                "economicLifetime": 9,
                "technicalLifetime": 9,
                "stockCommissioning": {2015: pd.Series({"PerfectLand": 1.0})},
                "commissioningFix": {
                    year: pd.Series({"PerfectLand": 0.0})
                    for year in [2020, 2025, 2030, 2035]
                },
            },
        ),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    # 2015 has outlived its technical lifetime of 9 years by the window starting in 2025
    assert 2015 not in (results[2025].getComponent("Src").stockCommissioning or {})


def test_a_run_does_not_warn_about_the_precision_of_its_own_results():
    """The stock checks of utils warn about a value that carries more digits than they
    round to. A commissioning result carries the full float64 precision, so a run that
    hands it on unrounded warns about itself.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rollingHorizonOptimization(
            esM=simplePathwayEsM(3),
            timeSeriesAggregation=False,
            numberOfInvestmentPeriodsForRollingHorizon=2,
        )
    assert not [
        str(warning.message)
        for warning in caught
        if "will be rounded to 10 digits" in str(warning.message)
    ]


# --- Parameters scoped down to a window --------------------------------------------


def test_parameters_are_scoped_to_the_window_and_its_stock_years(rollingResults):
    """A window only spans its own investment periods, so every parameter given per
    investment period is scoped down to them - plus, for the ones that are given per stock
    year as well, the years the previous windows commissioned in.

    In the window [2030, 2035], 2020 is such a stock year: the electrolyzer is still being
    paid for at the investPerCapacity of the year it was built in, while its operation
    side describes the window alone.
    """
    _, compDict = fn.dictIO.exportToDict(rollingResults[2030])
    electrolyzer = compDict["Conversion"]["Electrolyzer"]

    assert 2020 in stockOf(rollingResults, 2030, "Electrolyzer")
    assert sorted(electrolyzer["investPerCapacity"]) == [2020, 2030, 2035]
    assert sorted(electrolyzer["opexPerOperation"]) == [2030, 2035]
    assert sorted(electrolyzer["commodityConversionFactors"]) == [2030, 2035]


@pytest.mark.parametrize(
    "parameterName, keepsTheStockYear",
    [
        ("investPerCapacity", True),
        ("investIfBuilt", True),
        ("opexPerCapacity", True),
        ("opexIfBuilt", True),
        ("QPcostScale", True),
        ("opexPerOperation", False),
        ("opexPerChargeOperation", False),
        ("commodityCost", False),
        ("capacityMax", False),
        ("operationRateMax", False),
    ],
)
def test_only_parameters_paid_per_stock_year_may_keep_a_stock_year(
    parameterName, keepsTheStockYear
):
    """From the second window on, the stock is keyed by the year a previous window
    commissioned it in, which is also a key of every parameter given per investment period.
    Only the parameters describing capacity that is commissioned once and paid for over its
    lifetime are validated against the stock years as well; handing a stock year to any
    other one makes rebuilding the component fail.
    """
    esM = simplePathwayEsM(len(PATHWAY_YEARS))
    compEntry = {parameterName: {year: 1.0 for year in PATHWAY_YEARS}}

    _filterComponentParametersForInterval(compEntry, [2030, 2035], [2020], esM)

    expected = [2030, 2035] + ([2020] if keepsTheStockYear else [])
    assert sorted(compEntry[parameterName]) == sorted(expected)


def test_investment_period_parameters_are_filtered_whatever_their_key_order():
    """A parameter given per investment period describes the same years no matter which
    order they were written down in.
    """
    esM = simplePathwayEsM(len(PATHWAY_YEARS))
    compEntry = {"commodityCost": {year: 1.0 for year in reversed(PATHWAY_YEARS)}}

    _filterComponentParametersForInterval(compEntry, [2030, 2035], [], esM)

    assert sorted(compEntry["commodityCost"]) == [2030, 2035]


@pytest.mark.parametrize(
    "commodityConversionFactors, expected",
    [
        pytest.param(
            {year: {"electricity": -1} for year in PATHWAY_YEARS},
            {2030: {"electricity": -1}, 2035: {"electricity": -1}},
            id="perInvestmentPeriod",
        ),
        pytest.param(
            {
                (2020, 2030): {"electricity": -1},
                (2030, 2035): {"electricity": -2},
                (2025, 2035): {"electricity": -3},
                (2030, 2040): {"electricity": -4},
            },
            {(2020, 2030): {"electricity": -1}, (2030, 2035): {"electricity": -2}},
            id="perCommissioningAndOperationYear",
        ),
        pytest.param(
            {"hydrogen": -1, "electricity": 0.5},
            {"hydrogen": -1, "electricity": 0.5},
            id="timeConstant",
        ),
        pytest.param({}, {}, id="empty"),
    ],
)
def test_commodity_conversion_factors_are_filtered_by_what_they_are_keyed_by(
    commodityConversionFactors, expected
):
    """A conversion factor is keyed by the investment period, by a (commissioning year,
    operation year) pair, or directly by the commodity.

    Only the operation year has to lie in the window; a commissioning year outside it is
    kept if the component holds stock from it (2020 here) and dropped otherwise (2025). A
    factor that does not depend on time at all has no year to filter by and is passed
    through, as is an empty one, which has no first key to look at.
    """
    esM = simplePathwayEsM(len(PATHWAY_YEARS))
    compEntry = {
        "commodityConversionFactors": copy.deepcopy(commodityConversionFactors)
    }

    _filterComponentParametersForInterval(compEntry, [2030, 2035], [2020], esM)

    assert compEntry["commodityConversionFactors"] == expected


def test_a_dict_that_is_not_given_per_investment_period_is_left_alone():
    """The pwlcfParameters dict is keyed by the cost function, not by year. Filtering it as
    if it were given per investment period leaves an empty dict behind, which silently
    disables the piecewise linear cost function of the rebuilt component.
    """
    esM = simplePathwayEsM(len(PATHWAY_YEARS))
    compEntry = {"pwlcfParameters": copy.deepcopy(PWLCF_PARAMETERS)}

    _filterComponentParametersForInterval(compEntry, [2030, 2035], [2020], esM)

    assert compEntry["pwlcfParameters"] == PWLCF_PARAMETERS


def test_parameters_that_are_not_constructor_arguments_are_dropped():
    """A clustered model exports its aggregated time series on top of the constructor
    arguments, and a component cannot be rebuilt with them. They are recognized by the
    constructor's own signature rather than by name, so that anything else an export may
    carry is dropped as well.
    """
    esM = pathwayEsM()
    esM.aggregateTemporally(n_clusters=1, period_duration=4380, segments=None)
    _, compDict = fn.dictIO.exportToDict(esM)
    assert any(name.startswith("aggregated") for name in compDict["Source"]["PV"])

    # the first window, so that no previous one's results are looked up
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

    for classname, components in built.items():
        constructorArguments = set(
            inspect.getfullargspec(getattr(fn, classname).__init__).args
        )
        for compName, compEntry in components.items():
            leftOver = set(compEntry) - constructorArguments
            assert not leftOver, f"{classname} '{compName}' kept {sorted(leftOver)}"


def parametersKeyedByStockYears(esM, compName):
    """Return the parameters whose processed counterpart is keyed by the stock years.

    A component builds the parameters that are charged for capacity commissioned before the
    first modeled year over the stock years and the investment periods, and every other one
    over the investment periods alone. On a component that has stock, the keys of the
    processed counterpart therefore tell the two apart.
    """
    component = esM.getComponent(compName)
    keyedByStockYears = set()
    for name in inspect.getfullargspec(type(component).__init__).args:
        processed = getattr(component, f"processed{name[:1].upper()}{name[1:]}", None)
        if isinstance(processed, dict) and set(processed) > set(esM.investmentPeriods):
            keyedByStockYears.add(name)
    return keyedByStockYears


def esMWithTransmissionStock():
    """Build a model whose Transmission holds stock.

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


def esMWithSourceStock():
    return simplePathwayEsM(
        3,
        sourceKwargs={"stockCommissioning": {2015: pd.Series({"PerfectLand": 1.0})}},
    )


@pytest.mark.parametrize(
    "buildEsM, compName",
    [
        pytest.param(esMWithSourceStock, "Src", id="Source"),
        pytest.param(esMWithTransmissionStock, "WithStock", id="Transmission"),
    ],
)
def test_the_stock_year_parameters_still_match_what_fine_builds(buildEsM, compName):
    """_STOCK_YEAR_PARAMETERS is written out rather than derived, so it can go stale.

    It cannot be derived where it is used, because the filter runs against the components
    of the pathway model, which normally carry no stock - and without stock a component
    keys all of its parameters alike, which is asserted here first. A component that does
    have stock can be asked, and this fails once FINE charges another parameter per stock
    year.
    """
    assert parametersKeyedByStockYears(simplePathwayEsM(3), "Src") == set()

    keyedByStockYears = parametersKeyedByStockYears(buildEsM(), compName)

    assert keyedByStockYears, "the component has no stock, so it cannot be asked"
    assert keyedByStockYears == set(_STOCK_YEAR_PARAMETERS)


# --- Numerical results --------------------------------------------------------------


def test_the_windows_reproduce_their_objective_values(rollingResults):
    """Pin what every window of the pathway costs, so that a change to how a window is
    built, scoped or handed its stock shows up as a changed number and not only as a
    changed structure.
    """
    expected = {
        2020: 4663.182150280922,
        2025: 4665.563680198432,
        2030: 4665.486671470777,
        2035: 4651.507422856613,
    }
    for startYear, objective in expected.items():
        np.testing.assert_allclose(
            rollingResults[startYear].pyM.Obj(), objective, rtol=1e-5
        )


@pytest.fixture(scope="module")
def perfectForesightNetPresentValue():
    """Return the net present value perfect foresight books for the pinned commissioning."""
    esM = fixedCommissioningEsM()
    esM.optimize(timeSeriesAggregation=False)
    return sum(
        summaryValue(esM, "SourceSinkModel", "Src", "NPVcontribution", year)[
            "PerfectLand"
        ]
        for year in esM.investmentPeriodNames
    )


@pytest.mark.parametrize("window", [1, 2, 3])
def test_the_windows_book_the_same_net_present_value_as_perfect_foresight(
    window, perfectForesightNetPresentValue
):
    """No cost is lost or counted twice between the windows.

    A window charges an annuity for the investment periods it spans, and the capacity it
    commissions keeps being charged in the following windows, where it arrives as stock.
    Summed over the years the windows own, and re-based onto the pathway's start year by
    NPVcontributionRH, that has to reproduce what perfect foresight books for the same
    commissioning decisions - which commissioningFix pins down here.
    """
    results = rollingHorizonOptimization(
        esM=fixedCommissioningEsM(),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=window,
    )
    booked = sum(
        summaryValue(
            results[startYear], "SourceSinkModel", "Src", "NPVcontributionRH", year
        )["PerfectLand"]
        for startYear, year in yearsOwnedByEachWindow(results)
    )
    assert booked == pytest.approx(perfectForesightNetPresentValue, rel=1e-6)


def test_the_net_present_value_of_a_window_is_rebased_onto_the_pathway(rollingResults):
    """Every window discounts onto its own start year, so its NPVcontribution cannot be
    compared across windows. NPVcontributionRH is the same contribution discounted onto the
    first year of the whole pathway instead, at the component's own interest rate. The
    (1 + interestRate) convention factor of utils.discountFactor is in both rows and
    cancels.
    """
    interestRate = rollingResults[2030].getComponent("PV").interestRate["PerfectLand"]

    for year in (2030, 2035):
        npv = summaryValue(
            rollingResults[2030], "SourceSinkModel", "PV", "NPVcontribution", year
        )
        npvRebased = summaryValue(
            rollingResults[2030], "SourceSinkModel", "PV", "NPVcontributionRH", year
        )
        assert npvRebased["PerfectLand"] == pytest.approx(
            npv["PerfectLand"] / (1 + interestRate) ** (2030 - 2020)
        )

    # the first window already starts in the pathway's own start year
    npv = summaryValue(
        rollingResults[2020], "SourceSinkModel", "PV", "NPVcontribution", 2020
    )
    npvRebased = summaryValue(
        rollingResults[2020], "SourceSinkModel", "PV", "NPVcontributionRH", 2020
    )
    assert npvRebased["PerfectLand"] == pytest.approx(npv["PerfectLand"])


def test_a_model_without_a_rolling_horizon_does_not_report_the_rebased_row():
    """A stand-alone model has no shared reference year, so it reports NPVcontribution
    alone.
    """
    esM = simplePathwayEsM(3)
    esM.optimize(timeSeriesAggregation=False)
    properties = esM.getOptimizationSummary(
        "SourceSinkModel", ip=2020, outputLevel=0
    ).index.get_level_values("Property")
    assert "NPVcontribution" in properties
    assert "NPVcontributionRH" not in properties


def test_a_transmission_component_is_reported_like_any_other(rollingResults):
    """A 2-dim component's summary spans every location pair, including the ones it does
    not connect, while its interest rate is indexed by the connections it actually has. The
    rebased net present value has to be reported for it all the same.
    """
    summary = rollingResults[2020].getOptimizationSummary(
        "TransmissionModel", ip=2020, outputLevel=0
    )
    assert not summary.empty
    assert "NPVcontributionRH" in summary.index.get_level_values("Property")


# --- Myopic foresight ---------------------------------------------------------------


def test_a_window_of_one_optimizes_every_investment_period_on_its_own(myopicResults):
    """A window of one is myopic foresight: every investment period becomes its own window.

    The stock has to accumulate across all of the handoffs between them, not just hold what
    the immediately preceding window built.
    """
    assert sorted(myopicResults) == PATHWAY_YEARS
    assert all(esM.numberOfInvestmentPeriods == 1 for esM in myopicResults.values())

    lineStock = myopicResults[2040].getComponent("Line").stockCommissioning
    assert len(lineStock) > 1
    assert min(lineStock) < 2035


def test_a_myopic_co2_reduction_pathway_tightens_as_its_balance_limit_does():
    """A balanceLimit is a setting of the energy system model given per investment period,
    so every window has to be scoped down to it just like to a component's parameters.

    A loose target leaves the cheap emitting plant alone; a 100% reduction target leaves no
    budget at all, so emissions have to be zero and the demand has to be met by wind
    instead.
    """
    results = rollingHorizonOptimization(
        esM=co2PathwayEsM(),
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=1,
    )

    def emissions(year):
        return summaryValue(
            results[year], "SourceSinkModel", "CO2 to environment", "operation", year
        ).sum()

    assert emissions(2020) > 0
    assert emissions(2030) == pytest.approx(0)
    assert (
        summaryValue(results[2030], "SourceSinkModel", "Wind", "capacity", 2030).sum()
        > 0
    )


# --- Time series aggregation --------------------------------------------------------


def test_time_series_aggregation_settings_reach_the_clustering():
    """The clustering parameters are one dict passed straight through to
    aggregateTemporally, so that any keyword argument it takes is reachable. Its defaults
    are impossible to satisfy for this two time step model, so a run that succeeds at all
    already shows that the given values were the ones used.
    """
    results = rollingHorizonOptimization(
        esM=simplePathwayEsM(3),
        timeSeriesAggregation=True,
        timeSeriesAggregationSettings={
            "n_clusters": 2,
            "period_duration": 4380,
            "segments": fn.SegmentConfig(n_segments=1),
        },
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    esM = results[2020]
    assert len(esM.typicalPeriods) == 2
    assert len(esM.timeStepsPerPeriod) == 1
    assert len(esM.segmentsPerPeriod) == 1


def test_partial_time_series_aggregation_settings_keep_the_remaining_defaults():
    """Settings that are not given fall back to aggregateTemporally's own defaults rather
    than to defaults of the rolling horizon. Here n_clusters is left to it, and its default
    is more than this model has time steps.
    """
    with pytest.raises(ValueError, match="product of the numberOfTypicalPeriods"):
        rollingHorizonOptimization(
            esM=simplePathwayEsM(3),
            timeSeriesAggregation=True,
            timeSeriesAggregationSettings={"period_duration": 4380},
            numberOfInvestmentPeriodsForRollingHorizon=2,
        )


def test_a_cluster_config_brings_its_own_solver():
    """A clustering method that needs a solver brings it along in its ClusterConfig. That is
    where the clustering solver belongs, not in this function's solver argument, which
    selects the solver of the optimization - the two must not collide.
    """
    results = rollingHorizonOptimization(
        esM=simplePathwayEsM(3),
        timeSeriesAggregation=True,
        timeSeriesAggregationSettings={
            "n_clusters": 2,
            "period_duration": 4380,
            "segments": None,
            "cluster": fn.ClusterConfig(
                method="hierarchical",
                solver=ImplementedSolvers.STANDARD_SOLVER.value,
            ),
        },
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert len(results[2020].typicalPeriods) == 2


def test_an_already_aggregated_model_can_be_rolled():
    """Aggregating temporally and then optimizing is the normal FINE workflow, so a model
    whose time series are already clustered has to be accepted as the pathway to roll.
    """
    esM = simplePathwayEsM(3)
    esM.aggregateTemporally(n_clusters=1, period_duration=4380, segments=None)
    assert esM.isTimeSeriesDataClustered

    results = rollingHorizonOptimization(
        esM=esM,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )
    assert sorted(results) == [2020, 2025]


# --- netCDF output and resuming an interrupted run ----------------------------------


def sharedNetCDFPath(dirPath, scenario_name=CACHE_NAME):
    return dirPath / f"{scenario_name}_rollingHorizon.nc"


def writeCachedGroup(esM, dirPath, startYear, scenario_name=CACHE_NAME):
    """Write a single window's model into its own group of the shared netCDF file, without
    touching any other group already there. Lets a test build a specific (partial, stale)
    cache from already solved windows, without solving them again.
    """
    fn.xrIO.writeEnergySystemModelToNetCDF(
        esM,
        outputFilePath=str(sharedNetCDFPath(dirPath, scenario_name)),
        overwriteExisting=False,
        groupPrefix=str(startYear),
    )


def trackOptimizeCalls(monkeypatch):
    """Record the start year of every model that is actually optimized, without changing
    what optimize does.
    """
    calls = []
    originalOptimize = fn.EnergySystemModel.optimize

    def trackingOptimize(self, *args, **kwargs):
        calls.append(self.startYear)
        return originalOptimize(self, *args, **kwargs)

    monkeypatch.setattr(fn.EnergySystemModel, "optimize", trackingOptimize)
    return calls


def srcCommissioning(results, year):
    return summaryValue(results[year], "SourceSinkModel", "Src", "commissioning", year)[
        "PerfectLand"
    ]


def test_the_windows_share_one_netcdf_file_with_one_group_each(cachedRun):
    """Writing netCDF output produces a single file named after the scenario, holding one
    group per window keyed by its start year - mirroring the single file output of perfect
    foresight instead of writing one file per window. A group read back reproduces the
    window it was written from.
    """
    results, cacheDir = cachedRun
    netCDFPath = sharedNetCDFPath(cacheDir)
    assert netCDFPath.is_file()
    assert all(_cachedGroupExists(netCDFPath, str(year)) for year in results)

    loaded = fn.xrIO.readNetCDFtoEnergySystemModel(str(netCDFPath), groupPrefix="2020")
    assert summaryValue(loaded, "SourceSinkModel", "Src", "commissioning", 2020)[
        "PerfectLand"
    ] == pytest.approx(srcCommissioning(results, 2020))


def test_resuming_a_finished_run_solves_nothing(cachedRun, monkeypatch):
    """The point of resuming is that no window is solved twice: with every group in place,
    each window is loaded from the cache instead of being rebuilt and re-solved.
    """
    results, cacheDir = cachedRun
    optimizeCalls = trackOptimizeCalls(monkeypatch)

    resumed = rollingHorizonOptimization(
        esM=simplePathwayEsM(4),
        scenario_name=CACHE_NAME,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        resume=True,
        resultExportPath=str(cacheDir),
    )

    assert optimizeCalls == []
    assert sorted(resumed) == sorted(results)
    for year in resumed:
        assert srcCommissioning(resumed, year) == pytest.approx(
            srcCommissioning(results, year)
        )


def test_resuming_solves_only_the_windows_that_are_missing(
    cachedRun, tmp_path, monkeypatch
):
    """With only the first window's group in place, as if the run had been interrupted right
    after it, resuming loads that one and solves the rest. The result has to match an
    uninterrupted run, which shows that reloading rather than re-solving a window leaves the
    stock bookkeeping untouched.
    """
    results, _ = cachedRun
    writeCachedGroup(results[2020], tmp_path, 2020)
    optimizeCalls = trackOptimizeCalls(monkeypatch)

    resumed = rollingHorizonOptimization(
        esM=simplePathwayEsM(4),
        scenario_name=CACHE_NAME,
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        resume=True,
        resultExportPath=str(tmp_path),
    )

    assert optimizeCalls == [2025, 2030]
    assert pathwayNetPresentValue(resumed, "Src") == pytest.approx(
        pathwayNetPresentValue(results, "Src")
    )


def test_resuming_refuses_a_cache_built_for_another_window_size(cachedRun, tmp_path):
    """A cached window that was built for another configuration is a user error - calling
    with a different numberOfInvestmentPeriodsForRollingHorizon than the interrupted run
    used - and fails loudly instead of silently producing an inconsistent result.
    """
    results, _ = cachedRun
    writeCachedGroup(results[2020], tmp_path, 2020)

    with pytest.raises(ValueError, match="does not match this call's configuration"):
        rollingHorizonOptimization(
            esM=simplePathwayEsM(4),
            scenario_name=CACHE_NAME,
            timeSeriesAggregation=False,
            numberOfInvestmentPeriodsForRollingHorizon=1,
            resume=True,
            resultExportPath=str(tmp_path),
        )


def test_resuming_discards_a_stale_cache_and_solves_it_again(
    cachedRun, tmp_path, monkeypatch
):
    """A cached window whose accumulated stock does not match what was just recomputed for
    it no longer belongs to the chain being computed - an earlier window was solved
    differently in the meantime. Unlike a configuration mismatch this can legitimately
    happen, so it is discarded with a warning and solved again.

    Every later window is solved again as well, even though its own group is untouched: a
    group surviving downstream of a regenerated predecessor was necessarily built from a
    different chain. The result therefore matches an uninterrupted run again.
    """
    results, cacheDir = cachedRun
    writeCachedGroup(results[2020], tmp_path, 2020)
    stale = fn.xrIO.readNetCDFtoEnergySystemModel(
        str(sharedNetCDFPath(cacheDir)), groupPrefix="2025"
    )
    staleComponent = stale.getComponent("Src")
    staleComponent.stockCommissioning = {
        year: series + 1.0 for year, series in staleComponent.stockCommissioning.items()
    }
    writeCachedGroup(stale, tmp_path, 2025)
    writeCachedGroup(results[2030], tmp_path, 2030)
    optimizeCalls = trackOptimizeCalls(monkeypatch)

    with pytest.warns(UserWarning, match="stale"):
        resumed = rollingHorizonOptimization(
            esM=simplePathwayEsM(4),
            scenario_name=CACHE_NAME,
            timeSeriesAggregation=False,
            numberOfInvestmentPeriodsForRollingHorizon=2,
            resume=True,
            resultExportPath=str(tmp_path),
        )

    assert optimizeCalls == [2025, 2030]
    assert pathwayNetPresentValue(resumed, "Src") == pytest.approx(
        pathwayNetPresentValue(results, "Src")
    )


def test_a_run_that_does_not_resume_starts_from_a_clean_cache_file(tmp_path):
    """A fresh run must not mix its groups with the ones of an earlier, unrelated run, so
    it clears the file it is about to write - loudly, since that is the moment a caller who
    meant to resume finds out.
    """

    def freshRun():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = rollingHorizonOptimization(
                esM=simplePathwayEsM(3),
                scenario_name="fresh",
                timeSeriesAggregation=False,
                numberOfInvestmentPeriodsForRollingHorizon=2,
                writeNetCDFOutput=True,
                resultExportPath=str(tmp_path),
            )
        return results, [
            warning for warning in caught if "is deleted" in str(warning.message)
        ]

    _, firstRunWarnings = freshRun()
    results, secondRunWarnings = freshRun()

    assert firstRunWarnings == []
    assert len(secondRunWarnings) == 1
    assert sorted(results) == [2020, 2025]
    assert all(
        _cachedGroupExists(sharedNetCDFPath(tmp_path, "fresh"), str(year))
        for year in results
    )


# --- Cache validation, without solving anything -------------------------------------


class CachedEsm:
    """Stand in for a cached window, holding only what the configuration check reads."""

    def __init__(self, startYear, numberOfInvestmentPeriods):
        self.startYear = startYear
        self.numberOfInvestmentPeriods = numberOfInvestmentPeriods


@pytest.mark.parametrize(
    "cached, years, window, expected",
    [
        pytest.param((2020, 2), [2025, 2030], 2, "startYear", id="differentStartYear"),
        pytest.param(
            (2020, 2),
            [2020, 2025],
            1,
            "numberOfInvestmentPeriods",
            id="differentWindow",
        ),
        pytest.param((2020, 2), [2020, 2025], 2, None, id="matching"),
    ],
)
def test_a_cached_window_is_checked_against_this_call(cached, years, window, expected):
    """The hard check resume relies on: the cached window's own configuration against what
    this call asks for.
    """
    reasons = _cachedIntervalConfigMismatches(
        CachedEsm(*cached),
        rollingHorizonYears=years,
        numberOfInvestmentPeriodsForRollingHorizon=window,
    )
    if expected is None:
        assert reasons == []
    else:
        assert any(expected in reason for reason in reasons)


@pytest.mark.parametrize(
    "fresh, cached, differs",
    [
        pytest.param(None, None, False, id="neitherHasStock"),
        pytest.param(None, {2020: 1.0}, True, id="onlyOneHasStock"),
        pytest.param({2020: 1.0}, None, True, id="onlyTheOtherHasStock"),
        pytest.param({2020: 1.0000001}, {2020: 1.0000002}, False, id="withinTolerance"),
        pytest.param({2020: 1.0}, {2020: 1.1}, True, id="beyondTolerance"),
        pytest.param({2020: 1.0}, {2025: 1.0}, True, id="differentYears"),
    ],
)
def test_two_stocks_are_compared_across_solver_runs(fresh, cached, differs):
    """The soft check resume relies on. It compares the results of two separate solver runs,
    so it absorbs solver noise rather than floating point noise.
    """

    def asStock(stock):
        if stock is None:
            return None
        return {
            year: pd.Series({"PerfectLand": value}) for year, value in stock.items()
        }

    assert _stockCommissioningDiffers(asStock(fresh), asStock(cached)) is differs


@pytest.mark.parametrize(
    "difference, expected",
    [
        pytest.param("componentSet", "component set", id="differentComponents"),
        pytest.param("stock", "stockCommissioning", id="differentStock"),
        pytest.param(None, None, id="matching"),
    ],
)
def test_a_cached_window_is_checked_against_the_chain_it_belongs_to(
    difference, expected
):
    """The cached window has to have been built from the same components and the same
    accumulated stock as what was just recomputed for it.
    """
    esM = esMWithSourceStock()
    _, cachedCompDict = fn.dictIO.exportToDict(esM)
    freshCompDict = copy.deepcopy(cachedCompDict)
    if difference == "componentSet":
        freshCompDict["Source"].pop("Src")
    elif difference == "stock":
        freshCompDict["Source"]["Src"]["stockCommissioning"] = {
            2015: pd.Series({"PerfectLand": 999.0})
        }

    reasons = _cachedIntervalChainMismatches(esM, freshCompDict)

    if expected is None:
        assert reasons == []
    else:
        assert any(expected in reason for reason in reasons)


# --- Excel output and pass-through settings -----------------------------------------


@pytest.mark.parametrize("window", [2, 1], ids=["window2", "myopic"])
def test_excel_output_writes_one_file_per_owned_year(tmp_path, window):
    """Every window except the last exports its own first year, the last one every year it
    spans - together the years that add up to the pathway, whatever the window size. Each
    of them is named after the year it holds, so that no window overwrites another.

    optSumOutputLevel is not an argument of this function; it is reachable through
    excelOutputSettings.
    """
    rollingHorizonOptimization(
        esM=simplePathwayEsM(4),
        scenario_name="excel",
        timeSeriesAggregation=False,
        numberOfInvestmentPeriodsForRollingHorizon=window,
        writeExcelOutput=True,
        resultExportPath=str(tmp_path),
        excelOutputSettings={"optSumOutputLevel": 0, "optValOutputLevel": 0},
    )
    assert sorted(path.name for path in tmp_path.glob("*.xlsx")) == [
        f"excel_rollingHorizon_{year}.xlsx" for year in (2020, 2025, 2030, 2035)
    ]


def test_optimize_settings_reach_optimize():
    """A setting of EnergySystemModel.optimize that the rolling horizon does not name itself
    is reachable through optimizeSettings.
    """
    results = rollingHorizonOptimization(
        esM=simplePathwayEsM(3),
        numberOfInvestmentPeriodsForRollingHorizon=2,
        timeSeriesAggregation=False,
        optimizeSettings={"includePerformanceSummary": True},
    )
    assert all(window.performanceSummary is not None for window in results.values())


def test_every_netcdf_writer_argument_is_reachable(tmp_path, monkeypatch):
    """Settings are passed on unchanged, so every argument of the writer that the rolling
    horizon does not name itself reaches it, while the three it determines itself stay its
    own. The writer is recorded rather than run, so that this says something about the
    pass-through and nothing about what the writer makes of the arguments.
    """
    seen = []

    def recordingWriter(_esM, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(
        "fine.expansionModules.rollingHorizon.writeEnergySystemModelToNetCDF",
        recordingWriter,
    )

    rollingHorizonOptimization(
        esM=simplePathwayEsM(3),
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


def test_the_excel_writer_accepts_a_numpy_year(tmp_path):
    """A year read off a pandas index is a numpy integer, which is just as valid a year as
    an int - and the form a window hands its years on in.
    """
    esM = simplePathwayEsM(2)
    esM.optimize(timeSeriesAggregation=False)

    writeOptimizationOutputToExcel(
        esM,
        outputFileName=str(tmp_path / "numpyYear"),
        investmentPeriod=pd.Index(esM.investmentPeriodNames)[0],
    )

    assert [path.name for path in tmp_path.glob("*.xlsx")] == ["numpyYear_2020.xlsx"]


def test_the_excel_writer_refuses_a_boolean_year():
    """A bool is an int, but not a year."""
    esM = simplePathwayEsM(2)
    with pytest.raises(ValueError, match="must be type int"):
        writeOptimizationOutputToExcel(
            esM, outputFileName="unused", investmentPeriod=True
        )


def test_the_rolling_horizon_is_exported_by_the_package():
    """The module is reachable as fn.rollingHorizonOptimization, like the other expansion
    modules.
    """
    assert fn.rollingHorizonOptimization is rollingHorizonOptimization
