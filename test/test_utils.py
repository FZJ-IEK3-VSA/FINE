import pandas as pd
from fine import utils
import fine as fn
import numpy as np
import pytest
import re

from fine.utils import ImplementedSolvers
from fine.utils import checkCallableConversionFactor


def test_checkSimultaneousChargeDischarge():
    """Test a minimal example, with two regions and 10 days, where simultaneous charge and discharge occurs."""
    locations = {"Region1", "Region2"}
    commodityUnitDict = {"electricity": r"MW$_{el}$"}
    commodities = {"electricity"}
    ndays = 10
    nhours = 24 * ndays
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
    )
    # Create synthetic daily demand profile
    dailyProfileSimple = [
        0.6,
        0.6,
        0.6,
        0.6,
        0.6,
        0.7,
        0.9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0.9,
        0.8,
    ]
    demand = pd.DataFrame(
        [[u * 40, u * 60] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    # Add storage 'Batteries'
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 1000, 0
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000
    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=chargeEfficiency,
            cyclicLifetime=cyclicLifetime,
            dischargeEfficiency=dischargeEfficiency,
            selfDischarge=selfDischarge,
            chargeRate=chargeRate,
            dischargeRate=dischargeRate,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )
    # Create synthetic profile for PV and add PV with fixed operationRate. Therefore, it cannot be curtailed.
    # To achieve a curtailment, the system 'burns' energy by charging and discharging the storage simultaneously.
    dailyProfileSimple = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.05,
        0.15,
        0.2,
        0.4,
        0.8,
        0.7,
        0.4,
        0.2,
        0.15,
        0.05,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    operationRateFix = pd.DataFrame(
        [[u, u] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    )
    capacityMax = pd.Series([10000, 10000], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 100, 10
    interestRate, economicLifetime = 0.08, 25
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=operationRateFix,
            capacityFix=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    with pytest.warns(UserWarning, match="Charge and discharge at the same time"):
        esM.optimize(
            timeSeriesAggregation=False,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )
    # Get the charge and discharge time series of the Batteries and use the check in the utils.
    tsCharge = esM.componentModelingDict[
        "StorageModel"
    ].chargeOperationVariablesOptimum.loc["Batteries"]
    tsDischarge = esM.componentModelingDict[
        "StorageModel"
    ].dischargeOperationVariablesOptimum.loc["Batteries"]
    simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
        tsCharge, tsDischarge
    )

    assert simultaneousChargeDischarge, (
        "Check for simultaneous charge & discharge should have returned True"
    )


def test_functionality_checkSimultaneousChargeDischarge():
    """Simple functionality test for utils.checkSimultaneousChargeDischarge."""
    # Define charge and discharge time series for one region
    tsCharge = pd.DataFrame(columns=["Region1"])
    tsCharge["Region1"] = 3 * [1] + 1 * [0]
    tsDischarge = pd.DataFrame(columns=["Region1"])
    tsDischarge["Region1"] = 2 * [0] + 2 * [1]
    simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
        tsCharge, tsDischarge
    )

    assert simultaneousChargeDischarge, (
        "Check for simultaneous charge & discharge should have returned True"
    )


def test_check_and_set_cost_parameter_for_part_load_conversion_factors():
    """Test cost parameter validation used by part-load conversion factor functions."""
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # Test with valid integer data (1dim)
    assert utils.checkAndSetCostParameter(esM, "testParam", 10, "1dim", None).equals(
        pd.Series([10.0], index=esM.locations)
    )

    # Test with valid series data (1dim)
    valid_series_1dim = pd.Series([10], index=esM.locations)
    assert utils.checkAndSetCostParameter(
        esM, "testParam", valid_series_1dim, "1dim", None
    ).equals(valid_series_1dim.astype(float))

    # Test with NaN in integer data (1dim)
    with pytest.raises(ValueError):
        assert utils.checkAndSetCostParameter(
            esM, "testParam", np.nan, "1dim", None
        ).equals(pd.Series([np.nan], index=esM.locations))

    # Test with NaN in series data (2dim)
    with pytest.raises(ValueError):
        invalid_series_with_nan = pd.Series([10, np.nan], index=["loc1", "loc2"])
        assert utils.checkAndSetCostParameter(
            esM, "testParam", invalid_series_with_nan, "2dim", None
        ).equals(invalid_series_with_nan, index=esM.locations)


def positive_factor(x):
    """Return a positive value in the interval [0, 1]."""
    return 0.5 + x


def zero_factor(_):
    """Return zero."""
    return 0


def negative_factor(_):
    """Return a negative value."""
    return -1


def crossing_factor(x):
    """Return values crossing zero within the interval [0, 1]."""
    return x - 0.5


@pytest.mark.parametrize(
    "conversion_factor, should_raise",
    [
        (positive_factor, False),
        (zero_factor, True),
        (negative_factor, True),
        (crossing_factor, True),
    ],
)
def test_checkCallableConversionFactor(conversion_factor, should_raise):
    """Test checkCallableConversionFactor for valid and invalid callables.

    The function should accept conversion factors that are strictly positive
    over the entire part-load range [0, 1] and raise a ValueError if the
    conversion factor becomes zero or negative at least once.
    """
    if should_raise:
        with pytest.raises(
            ValueError,
            match=(
                "The callable part load conversion factor is smaller or equal "
                "to 0 at least once within \\[0,1\\]."
            ),
        ):
            checkCallableConversionFactor(conversion_factor)
    else:
        checkCallableConversionFactor(conversion_factor)


def capacityDevelopmentKwargs(capacityFix, stock=None, lifetime=2.0, floor=True):
    """Build the processed parameters from {ip: value | {region: value} | None}."""

    def toSeries(values):
        if values is None:
            return None
        return {
            ip: None
            if value is None
            else pd.Series(value if isinstance(value, dict) else {"R": value})
            for ip, value in values.items()
        }

    capacityFix = toSeries(capacityFix)
    regions = next(v for v in capacityFix.values() if v is not None).index
    return dict(
        investmentPeriods=list(capacityFix.keys()),
        capacityMax=toSeries({ip: None for ip in capacityFix}),
        capacityFix=capacityFix,
        stockCommissioning=toSeries(stock),
        technicalLifetime=pd.Series(lifetime, index=regions),
        floorTechnicalLifetime=floor,
    )


@pytest.mark.parametrize(
    "capacityFix, stock, lifetime, floor, issueRegions",
    [
        # missing values for a single region and for a whole ip
        ({0: {"D": 5, "G": 10}, 1: {"D": None, "G": 15}, 2: None}, None, 2, True, []),
        (
            {0: {"D": 5, "G": 10}, 1: {"D": None, "G": 15}, 2: None},
            {-2: {"D": 0, "G": 0}, -1: {"D": 2, "G": 5}},
            2,
            True,
            [],
        ),
        # D: -3 from IP1 to IP2 exceeds the 1 commissioned in IP0
        (
            {0: {"D": 6, "G": 10}, 1: {"D": 3, "G": 5}, 2: {"D": 0, "G": 5}},
            {-2: {"D": 10, "G": 20}, -1: {"D": 5, "G": 10}},
            2,
            True,
            ["D"],
        ),
        # capacity can decrease to 0 in IP2, when the IP0 commissioning is decommissioned
        ({0: 10, 1: None, 2: None, 3: 0}, None, 2, True, []),
        # the stock is decommissioned until IP1
        ({0: None, 1: None, 2: 0}, {-2: 0, -1: 10}, 2, True, []),
        # requires a commissioning of 10 in IP1 instead of IP2
        ({0: 10, 1: None, 2: 10, 3: 0}, None, 2, True, []),
        # the IP0 commissioning is still active in IP1
        ({0: 10, 1: 0, 2: None}, None, 2, True, ["R"]),
        # floating point errors must not cause an infeasibility
        ({0: 4.88, 1: 8.87, 2: 6.24, 3: 2.25}, None, 2, True, []),
        ({0: 6.38, 1: 6.38}, {-3: 0, -2: 2.19, -1: 4.19}, 3, True, []),
        # a lifetime of 1.5 ip is floored to 1 or ceiled to 2
        ({0: 10, 1: 0}, None, 1.5, True, []),
        ({0: 10, 1: 0}, None, 1.5, False, ["R"]),
    ],
)
def test_checkCapacityDevelopmentWithStock(
    capacityFix, stock, lifetime, floor, issueRegions
):
    kwargs = capacityDevelopmentKwargs(capacityFix, stock, lifetime, floor)
    if not issueRegions:
        utils.checkCapacityDevelopmentWithStock(**kwargs)
    else:
        with pytest.raises(ValueError, match=re.escape(f"regions {issueRegions}")):
            utils.checkCapacityDevelopmentWithStock(**kwargs)


def test_checkCapacityDevelopmentWithStock_precedingMissingValue():
    expectedMessage = "A capacityFix value given for R is preceded by a missing value."
    with pytest.warns(UserWarning, match=re.escape(expectedMessage)):
        utils.checkCapacityDevelopmentWithStock(
            **capacityDevelopmentKwargs({0: None, 1: 4, 2: 5})
        )


def sourceEsM(stochasticModel=False, **sourceKwargs):
    esM = fn.EnergySystemModel(
        locations={"R"},
        commodities={"el"},
        numberOfTimeSteps=2,
        commodityUnitsDict={"el": "kW"},
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        startYear=2023,
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        lengthUnit="km",
        verboseLogLevel=2,
        stochasticModel=stochasticModel,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="el",
            hasCapacityVariable=True,
            technicalLifetime=10,
            **sourceKwargs,
        )
    )
    return esM


def test_capacityFixWithMissingInvestmentPeriods():
    # issue 839: capacityFix given as dict with None for some investment periods
    esM = sourceEsM(capacityFix={2023: 0.0, 2028: None, 2033: None})
    esM.optimize(
        solver=ImplementedSolvers.STANDARD_SOLVER.value, timeSeriesAggregation=False
    )
    assert esM.solverSpecs["terminationCondition"] == "optimal"


def test_capacityFixDecreasingInStochasticModel():
    # the investment periods of stochastic models are scenarios, so a decreasing
    # capacityFix does not conflict with the technical lifetime
    capacityFix = {2023: 10.0, 2028: 0.0, 2033: None}
    with pytest.raises(ValueError, match="Decreasing capacity fix"):
        sourceEsM(capacityFix=capacityFix)
    sourceEsM(capacityFix=capacityFix, stochasticModel=True)


def test_stockExceedsCapacityMaxInStochasticModel():
    # the stock of the first investment period applies to all scenarios of
    # stochastic models, while it is decommissioned until 2028 otherwise
    kwargs = dict(
        capacityMax={2023: 10.0, 2028: 5.0, 2033: None},
        stockCommissioning={2018: 10.0},
    )
    sourceEsM(**kwargs)
    with pytest.raises(ValueError, match="Mismatch between stock capacity"):
        sourceEsM(stochasticModel=True, **kwargs)
