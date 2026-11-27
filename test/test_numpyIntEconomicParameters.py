"""Economic parameters given as numpy integers are accepted like Python ints (#908)."""

import numpy as np
import pandas as pd
import pytest

import fine as fn

CONVERSION = dict(
    name="Conv",
    physicalUnit="GW",
    commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
    hasCapacityVariable=True,
)
SOURCE = dict(name="src", commodity="electricity", hasCapacityVariable=True)
TRANSMISSION = dict(name="Tr", commodity="electricity", hasCapacityVariable=True)

CASES = [
    (fn.Conversion, CONVERSION, "economicLifetime", 20, np.int64(20)),
    (fn.Conversion, CONVERSION, "technicalLifetime", 20, np.int64(20)),
    (fn.Conversion, CONVERSION, "interestRate", 0, np.int64(0)),
    (fn.Conversion, CONVERSION, "investPerCapacity", 1, np.int64(1)),
    (fn.Conversion, CONVERSION, "investPerCapacity", 1, {2020: np.int64(1)}),
    (fn.Transmission, TRANSMISSION, "investPerCapacity", 1, np.int64(1)),
    (fn.Transmission, TRANSMISSION, "investPerCapacity", 1, {2020: np.int64(1)}),
    (fn.Source, SOURCE, "commodityCost", 1, np.int64(1)),
    (fn.Source, SOURCE, "commodityCost", 1, {2020: np.int64(1)}),
]

ATTRIBUTES = {
    "economicLifetime": "economicLifetime",
    "technicalLifetime": "technicalLifetime",
    "interestRate": "interestRate",
    "investPerCapacity": "processedInvestPerCapacity",
    "commodityCost": "fullCommodityCost",
}


def buildEsM():
    return fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={"electricity": "GW", "hydrogen": "GW"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        startYear=2020,
    )


@pytest.mark.parametrize("cls, kwargs, param, intValue, numpyValue", CASES)
def test_numpyIntEconomicParameter(cls, kwargs, param, intValue, numpyValue):
    attr = ATTRIBUTES[param]
    expected = getattr(cls(esM=buildEsM(), **kwargs, **{param: intValue}), attr)
    actual = getattr(cls(esM=buildEsM(), **kwargs, **{param: numpyValue}), attr)
    if isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for ip in expected:
            pd.testing.assert_frame_equal(
                pd.DataFrame(actual[ip]), pd.DataFrame(expected[ip])
            )
    else:
        pd.testing.assert_series_equal(actual, expected)
