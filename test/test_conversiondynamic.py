#!/usr/bin/env python
# coding: utf-8

# # Workflow for a multi-regional energy system
#
import FINE as fn
import pandas as pd

import pytest


def test_ConversionDynamicNeedsCapacity():
    esM = fn.EnergySystemModel(
        locations={
            "example_region1",
        },
        commodities={"electricity", "methane"},
        commodityUnitsDict={"electricity": r"GW$_{el}$", "methane": r"GW$_{th}$"},
        verboseLogLevel=2,
    )

    with pytest.raises(ValueError, match=r".*hasCapacityVariable.*"):
        fn.ConversionDynamic(
            esM=esM,
            name="restricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
            partLoadMin=0.3,
            bigM=100,
            rampDownMax=0.5,
            investPerCapacity=0.5,
            opexPerCapacity=0.021,
            opexPerOperation=1,
            interestRate=0.08,
            economicLifetime=33,
            hasCapacityVariable=False,
        )


def test_ConversionDynamicNeedsHigherOperationRate():
    numberOfTimeSteps = 4
    locations = {"ElectrolyzerLocation", "IndustryLocation"}
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity", "methane"},
        commodityUnitsDict={"electricity": r"GW$_{el}$", "methane": r"GW$_{th}$"},
        numberOfTimeSteps=numberOfTimeSteps,
        verboseLogLevel=2,
    )

    operationRateMax = pd.DataFrame(
        [
            [
                0.2,
                0.4,
                1.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
        index=list(locations),
        columns=range(0, numberOfTimeSteps),
    ).T

    with pytest.raises(ValueError, match=r".*operationRateMax.*"):
        fn.ConversionDynamic(
            esM=esM,
            name="restricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
            partLoadMin=0.3,
            bigM=100,
            rampDownMax=0.5,
            operationRateMax=operationRateMax,
            investPerCapacity=0.5,
            opexPerCapacity=0.021,
            opexPerOperation=1,
            interestRate=0.08,
            economicLifetime=33,
        )


def test_ConversionDynamicHasHigherOperationRate():
    numberOfTimeSteps = 4
    locations = {"ElectrolyzerLocation", "IndustryLocation"}
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity", "methane"},
        commodityUnitsDict={"electricity": r"GW$_{el}$", "methane": r"GW$_{th}$"},
        numberOfTimeSteps=numberOfTimeSteps,
        verboseLogLevel=2,
    )

    operationRateMax = pd.DataFrame(
        [
            [
                0.0,
                0.4,
                1.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ],
        index=list(locations),
        columns=range(0, numberOfTimeSteps),
    ).T

    fn.ConversionDynamic(
        esM=esM,
        name="restricted",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": 1, "methane": -1 / 0.625},
        partLoadMin=0.3,
        bigM=100,
        rampDownMax=0.5,
        operationRateMax=operationRateMax,
        investPerCapacity=0.5,
        opexPerCapacity=0.021,
        opexPerOperation=1,
        interestRate=0.08,
        economicLifetime=33,
    )


if __name__ == "__main__":
    test_ConversionDynamicNeedsCapacity()
    test_ConversionDynamicNeedsHigherOperationRate()
    test_ConversionDynamicHasHigherOperationRate()
