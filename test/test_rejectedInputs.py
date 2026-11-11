"""Tests that component constructors reject invalid input (FINE 3 refactor, PR 2).

Each case reaches one check function in fine/utils.py through a component
constructor, so a check lost during later refactors makes its test fail. The
tests assert the exception type raised today and may break on purpose once
InputValidationError is introduced.
"""

import numpy as np
import pandas as pd
import pytest

import fine as fn

PATHWAY = dict(numberOfInvestmentPeriods=2, investmentPeriodInterval=5, startYear=2020)
ESM_VARIANTS = {
    "basic": {},
    "pathway": PATHWAY,
    "perpetuity": dict(PATHWAY, annuityPerpetuity=True),
}

VALID = {
    fn.Source: dict(name="src", commodity="electricity", hasCapacityVariable=True),
    fn.Conversion: dict(
        name="Conv",
        physicalUnit="GW",
        commodityConversionFactors={"electricity": 1, "hydrogen": -0.7},
    ),
    fn.Storage: dict(name="s", commodity="electricity"),
    fn.Transmission: dict(name="Tr1", commodity="electricity"),
    fn.ConversionDynamic: dict(
        name="dyn",
        physicalUnit="GW",
        commodityConversionFactors={"electricity": -1, "hydrogen": 1},
        hasCapacityVariable=True,
    ),
    fn.ConversionPartLoad: dict(
        name="pl",
        physicalUnit="GW",
        commodityConversionFactors={"electricity": -1, "hydrogen": 1},
        commodityConversionFactorsPartLoad={
            "electricity": -1,
            "hydrogen": lambda _: 0.5,
        },
        hasCapacityVariable=True,
    ),
}


def buildEsM(variant):
    return fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={"electricity": "GW", "hydrogen": "GW"},
        numberOfTimeSteps=4,
        hoursPerTimeStep=2190,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        **ESM_VARIANTS[variant],
    )


# Rows: (check function, class or tuple of classes, ESM variant, overrides, exception,
# match). A row with several classes is one case per class.
# fn.Sink only forwards to Source.__init__, so all Source/Sink cases use fn.Source.
# sortTimeSeriesColumns is not covered: it has no raise of its own, only a builtin
# sort TypeError whose message is not specific to it.
# fmt: off
ROWS = [
    ("checkCommodities", (fn.Source, fn.Storage, fn.Transmission),
        "basic", {"commodity": "gas"}, ValueError, "Commodity does not match"),
    ("setLocationalEligibility", (fn.Source, fn.Conversion, fn.Storage),
        "basic", {"locationalEligibility": pd.Series({"loc1": 1})},
        ValueError, "match the esM locations"),
    # checkAndSetInvestmentPeriodParameters only forwards to checkInvestmentPeriodParameters,
    # which raises; the 'capacityPerPlantUnit' name pins the call path.
    ("checkAndSetInvestmentPeriodParameters", fn.Source,
        "basic", {"capacityPerPlantUnit": {2020: 1}}, ValueError, "'capacityPerPlantUnit'"),
    ("checkDesignVariableModelingParameters", (fn.Source, fn.Conversion),
        "basic", {"capacityVariableDomain": "integer"}, ValueError, "'continuous' or 'discrete'"),
    ("isStrictlyPositiveNumber", (fn.Source, fn.Conversion, fn.Transmission),
        "basic", {"capacityPerPlantUnit": 0}, ValueError, "has to be strictly positive"),
    ("isPositiveNumber", (fn.Source, fn.Conversion, fn.Transmission),
        "basic", {"tsaWeight": -1}, ValueError, "has to be positive"),
    ("isString", (fn.Source, fn.Storage),
        "basic", {"sharedPotentialID": 1}, TypeError, "argument has to be a string"),
    ("checkAndSetCostParameter", (fn.Source, fn.Conversion),
        "basic", {"economicLifetime": -1}, ValueError, "Economic parameters"),
    ("checkAndSetInterestRate", (fn.Source, fn.Conversion, fn.Storage, fn.Transmission),
        "perpetuity", {"interestRate": 0}, ValueError, "interest rate of 0"),
    ("checkEconomicAndTechnicalLifetime", fn.Source,
        "basic", {"economicLifetime": 10, "technicalLifetime": 5}, ValueError, "Economic Lifetime must"),
    ("checkFlooringParameter", (fn.Source, fn.Conversion, fn.Storage, fn.Transmission),
        "basic", {"floorTechnicalLifetime": "yes"}, ValueError, "floorTechnicalLifetime must"),
    ("checkStockYears", fn.Source,
        "basic", {"stockCommissioning": {5: pd.Series({"loc1": 1, "loc2": 1})}},
        ValueError, "Stock years"),
    ("checkAndSetStock", fn.Source,
        "basic", {"stockCommissioning": {-1: pd.Series({"loc1": -1, "loc2": 0})}},
        ValueError, "negative or NaN"),
    ("checkCapacityDevelopmentWithStock", fn.Source,
        "pathway", {"capacityFix": {2020: 5, 2025: 1}, "economicLifetime": 20},
        ValueError, "Decreasing capacity fix"),
    ("checkAndSetInvestmentPeriodCostParameter", fn.Source,
        "basic", {"investPerCapacity": "high"}, TypeError, "series or a dictionary"),
    ("checkAndSetBounds", (fn.Source, fn.Conversion),
        "basic", {"capacityMin": pd.Series({"loc1": -1, "loc2": 0})},
        ValueError, "capacityMin values for"),
    ("checkAndSetFullLoadHoursParameter", (fn.Source, fn.Conversion),
        "basic", {"yearlyFullLoadHoursMin": -1}, ValueError, "Full load hours"),
    ("checkLocationSpecficDesignInputParams", (fn.Source, fn.Conversion, fn.Storage),
        "basic", {"isBuiltFix": pd.Series({"loc1": 1, "loc2": 1})},
        ValueError, "Fixed design decisions"),
    ("checkAndSetBalanceLimitID", fn.Source,
        "basic", {"balanceLimitID": 1}, ValueError, "string or None"),
    ("checkAndSetInvestmentPeriodTimeSeries", fn.Source,
        "basic", {"operationRateMax": "full"}, TypeError, "does not match required type"),
    ("processCommodityCost", fn.Source,
        "basic", {"commodityCost": "cheap"}, TypeError, "unsupported type"),
    ("checkAndSetPartLoadMin", fn.Source,
        "basic", {"partLoadMin": "half"}, TypeError, "Wrong datatype for partLoadMin"),
    ("checkCommodities", fn.Conversion,
        "basic", {"commodityConversionFactors": {"electricity": 1, "gas": -1}},
        ValueError, "Commodity does not match"),
    ("checkCommodityUnits", fn.Conversion,
        "basic", {"physicalUnit": "MW"}, ValueError, "Commodity unit does not"),
    ("checkConversionFactorProperties", fn.Conversion,
        "basic", {"commodityConversionFactors": [1]}, ValueError, "Factor must be a dict"),
    ("checkAndSetCommodityConversionFactor", fn.Conversion,
        "basic", {"commodityConversionFactors": {"electricity": 1, "hydrogen": float("nan")}},
        ValueError, "NaN found at key"),
    ("checkEmissionFactors", fn.Conversion,
        "basic", {"emissionFactors": {"electricity": {"hydrogen": 1}}},
        NotImplementedError, "Emission factors can only"),
    ("checkAndSetFlowShares", fn.Conversion,
        "basic", {"flowShares": {"min": {"electricity": 0.5}}},
        NotImplementedError, "Flow shares can only"),
    ("checkAndSetInvestmentPeriodParameters", fn.Conversion,
        "basic", {"capacityPerPlantUnit": {5: 1}}, ValueError, "'capacityPerPlantUnit'"),
    ("isString", fn.Conversion,
        "basic", {"linkedConversionCapacityID": 1}, TypeError, "argument has to be a string"),
    ("checkEconomicAndTechnicalLifetime", (fn.Conversion, fn.Storage, fn.Transmission),
        "basic", {"economicLifetime": 20, "technicalLifetime": 10},
        ValueError, "Economic Lifetime must"),
    ("checkStockYears", fn.Conversion,
        "basic", {"stockCommissioning": {5: 1}}, ValueError, "Stock years"),
    ("checkAndSetStock", fn.Conversion,
        "basic", {"stockCommissioning": {-1: 1}}, ValueError, "location of the stock"),
    ("checkCapacityDevelopmentWithStock", fn.Conversion,
        "pathway", {
            "economicLifetime": 20,
            "capacityMax": {2020: pd.Series({"loc1": 10, "loc2": 10}), 2025: pd.Series({"loc1": 1, "loc2": 1})},
            "stockCommissioning": {2015: pd.Series({"loc1": 5, "loc2": 5})},
        },
        ValueError, "Mismatch between stock"),
    ("checkAndSetInvestmentPeriodCostParameter", fn.Conversion,
        "basic", {"investPerCapacity": "abc"}, TypeError, "series or a dictionary"),
    ("checkAndSetInvestmentPeriodTimeSeries", (fn.Conversion, fn.Transmission),
        "basic", {"operationRateMax": "abc"}, TypeError, "does not match required type"),
    ("checkRampRates", fn.Conversion,
        "basic", {"rampUpMax": np.int64(-1)}, TypeError, "positive float or int"),
    ("checkAndSetPartLoadMin", fn.Conversion,
        "basic", {"partLoadMin": 1.5, "bigM": 100}, ValueError, "float in the intervall"),
    ("isInRange", fn.Storage,
        "basic", {"chargeEfficiency": 1.5}, ValueError, "has to be in the range"),
    ("isStrictlyPositiveNumber", fn.Storage,
        "basic", {"chargeRate": 0}, ValueError, "has to be strictly positive"),
    ("isPositiveNumber", fn.Storage,
        "basic", {"dischargeTsaWeight": -1}, ValueError, "has to be positive"),
    ("checkAndSetInvestmentPeriodParameters", fn.Storage,
        "basic", {"capacityPerPlantUnit": {1: 1, 2: 1}}, ValueError, "'capacityPerPlantUnit'"),
    ("checkDesignVariableModelingParameters", (fn.Storage, fn.Transmission),
        "basic", {"capacityVariableDomain": "foo"}, ValueError, "'continuous' or 'discrete'"),
    ("checkAndSetCostParameter", fn.Storage,
        "basic", {"economicLifetime": "a"}, TypeError, "Economic parameters"),
    ("checkStockYears", fn.Storage,
        "pathway", {"stockCommissioning": {2020: pd.Series({"loc1": 1, "loc2": 1})}},
        ValueError, "Stock years"),
    ("checkAndSetStock", fn.Storage,
        "pathway", {"stockCommissioning": {2015: 1}}, ValueError, "location of the stock"),
    ("checkCapacityDevelopmentWithStock", fn.Storage,
        "pathway", {"capacityFix": {2020: 2, 2025: 1}}, ValueError, "Decreasing capacity fix"),
    ("checkAndSetInvestmentPeriodCostParameter", fn.Storage,
        "basic", {"investPerCapacity": "a"}, TypeError, "series or a dictionary"),
    ("checkAndSetBounds", fn.Storage,
        "basic", {"capacityMax": pd.Series({"loc1": -1, "loc2": -1})},
        ValueError, "capacityMax values for"),
    ("checkAndSetInvestmentPeriodTimeSeries", fn.Storage,
        "basic", {"stateOfChargeMin": "a"}, TypeError, "does not match required type"),
    ("checkAndSetPartLoadMin", fn.Storage,
        "basic", {"partLoadMin": "a"}, TypeError, "Wrong datatype for partLoadMin"),
    ("checkAndSetDistances", fn.Transmission,
        "basic", {"distances": pd.Series({"loc1_loc2": -1.0, "loc2_loc1": -1.0})},
        ValueError, "Distance values"),
    ("checkAndSetTransmissionLosses", fn.Transmission,
        "basic", {"losses": 2}, ValueError, "Losses have to be"),
    ("checkCapacityOrCommissioningTransmission", fn.Transmission,
        "basic", {"capacityMax": "abc"}, ValueError, "dataframe, series, dict"),
    ("preprocess2dimInvestmentPeriodData", fn.Transmission,
        "basic", {"investPerCapacity": "abc"}, TypeError, "dataframe or a dictionary"),
    ("setLocationalEligibility", fn.Transmission,
        "basic", {"locationalEligibility": pd.Series({"loc1_loc2": 2, "loc2_loc1": 2})},
        ValueError, "either True or False"),
    ("checkAndSetInvestmentPeriodParameters", fn.Transmission,
        "basic", {"capacityPerPlantUnit": {0: 1, 1: 1}}, ValueError, "'capacityPerPlantUnit'"),
    ("isString", fn.Transmission,
        "basic", {"sharedPotentialID": 5, "capacityMax": 5}, TypeError, "argument has to be a string"),
    ("checkAndSetCostParameter", fn.Transmission,
        "basic", {"economicLifetime": float("nan")}, ValueError, "Economic parameters"),
    ("checkStockYears", fn.Transmission,
        "basic", {"stockCommissioning": {0: pd.Series({"loc1_loc2": 1.0, "loc2_loc1": 1.0})}},
        ValueError, "Stock years"),
    ("checkAndSetStock", fn.Transmission,
        "basic", {"stockCommissioning": {-1: 5}}, ValueError, "location of the stock"),
    ("checkCapacityDevelopmentWithStock", fn.Transmission,
        "pathway", {"capacityFix": {2020: 10, 2025: 5}, "economicLifetime": 20},
        ValueError, "Decreasing capacity fix"),
    ("checkAndSetInvestmentPeriodCostParameter", fn.Transmission,
        "basic", {"QPcostScale": "abc"}, TypeError, "series or a dictionary"),
    ("checkAndSetBounds", fn.Transmission,
        "basic", {"capacityMin": 5, "capacityMax": 3}, ValueError, "capacityMin values >"),
    ("checkLocationSpecficDesignInputParams", fn.Transmission,
        "basic", {"isBuiltFix": 1}, ValueError, "Fixed design decisions"),
    ("checkAndSetPartLoadMin", fn.Transmission,
        "basic", {"partLoadMin": 2.0, "bigM": 100}, ValueError, "float in the intervall"),
    # Only check calls written directly in fine/subclasses/*.py. LinearOptimalPowerFlow
    # (lopf.py) only calls preprocess2dimData, which cannot raise.
    ("checkConversionDynamicSpecficDesignInputParams", fn.ConversionDynamic,
        "basic", {"downTimeMin": 0, "bigM": 100}, ValueError, "integer in the intervall"),
    ("checkAndSetMaintenanceParameters", fn.ConversionDynamic,
        "basic", {"maintenanceTime": "x", "maintenanceOccurrences": 1},
        TypeError, "maintenanceTime must be"),
    ("checkNumberOfConversionFactors", fn.ConversionPartLoad,
        "basic", {"commodityConversionFactorsPartLoad": {"electricity": -1, "hydrogen": 1, "heat": 1}},
        ValueError, "are overwritten by"),
    ("checkCommodities", fn.ConversionPartLoad,
        "basic", {"commodityConversionFactorsPartLoad": {"electricity": -1, "heat": lambda _: 0.5}},
        ValueError, "Commodity does not match"),
]
# fmt: on

CASES = [
    pytest.param(
        cls,
        variant,
        overrides,
        exception,
        match,
        id=f"{'SourceSink' if cls is fn.Source else cls.__name__}-{function}",
    )
    for function, classes, variant, overrides, exception, match in ROWS
    for cls in (classes if isinstance(classes, tuple) else (classes,))
]


@pytest.mark.parametrize("cls, variant, overrides, exception, match", CASES)
def test_rejectedInput(cls, variant, overrides, exception, match):
    esM = buildEsM(variant)
    with pytest.raises(exception, match=match):
        cls(esM=esM, **{**VALID[cls], **overrides})
