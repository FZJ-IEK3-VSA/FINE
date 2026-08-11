#!/usr/bin/env python

# # Workflow for a multi-regional energy system
#
import fine as fn
import pandas as pd

import pytest


NUMBER_OF_HOURS = 10
ENERGY_FLOW = 10


def _create_system(
    numberOfTimeSteps,
    hoursPerTimeStep,
):
    """Create energy system with several components.

    Heat purchase (grid) -------------------------------------------->
                                                                        Heat Demand
    Methane purchase -----> Methane boiler (Conversion Dynamic) ----->

    """
    esM = fn.EnergySystemModel(
        locations={
            "region1",
        },
        numberOfTimeSteps=int(numberOfTimeSteps),
        hoursPerTimeStep=hoursPerTimeStep,
        commodities={"electricity", "methane", "heat"},
        commodityUnitsDict={"electricity": "kW", "methane": "kW", "heat": "kW"},
        verboseLogLevel=2,
    )

    esM.add(
        fn.ConversionDynamic(
            esM=esM,
            name="Methane heater",
            physicalUnit="kW",
            commodityConversionFactors={
                "methane": -1,
                "heat": 1,
            },
            hasCapacityVariable=True,
            capacityFix=ENERGY_FLOW,
        )
    )

    # add heat source from grid
    esM.add(
        fn.Source(
            esM=esM,
            name="Heating Grid",
            commodity="heat",
            hasCapacityVariable=False,
            commodityCost=1,  # some value higher than methan
        )
    )

    # add methane purchase
    esM.add(
        fn.Source(
            esM=esM,
            name="Methane purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCost=0.05,
        )
    )
    # add methane purchase
    esM.add(
        fn.Sink(
            esM=esM,
            name="Heat demand",
            commodity="heat",
            operationRateFix=pd.Series(data=[ENERGY_FLOW] * int(numberOfTimeSteps))
            * hoursPerTimeStep,  # multiplication to get from power to energy
            hasCapacityVariable=False,
        )
    )
    return esM


@pytest.mark.parametrize(
    ["hoursPerTimeStep", "partLoadMin"],
    [
        (0.25, None),
        (1, None),
        (1, 0.5),
        (0.25, 0.5),
    ],
)
def test_downTimeMin(hoursPerTimeStep, partLoadMin, useTemporalCyclicConstraints=True):
    """Test the downtime of the system.

    The heat production with a methane heater is cheaper and therefore preferred.

    The operationRateFix will be set to 0 in the middle of the considered time-steps for a single time step.
    Therefore, the methane heater must be shut off but has to keep the downtime min.
    """
    esM = _create_system(
        numberOfTimeSteps=NUMBER_OF_HOURS / hoursPerTimeStep,
        hoursPerTimeStep=hoursPerTimeStep,
    )

    # Update Demand: Single time step to 0 so the heater must shut off
    opRateFix = esM.getComponent("Heat demand").operationRateFix.copy()
    opRateFix[4] = 0
    esM.updateComponent("Heat demand", {"operationRateFix": opRateFix})

    # Update ConversionDynamic: Use downTimeMin
    downTimeMin = 3
    esM.updateComponent(
        "Methane heater",
        {
            "downTimeMin": downTimeMin,
            "bigM": 10000,
            "partLoadMin": partLoadMin,
            "useTemporalCyclicConstraints": useTemporalCyclicConstraints,
        },
    )

    # optimize
    esM.optimize()

    # Check results: operation of the methane heater must be 3h*10kW less
    expectedOperation = ENERGY_FLOW * (NUMBER_OF_HOURS - downTimeMin)
    heater_operation = (
        esM.getOptimizationSummary("ConversionDynamicModel")
        .loc["Methane heater", "operation", "[kW*h]"]
        .loc["region1"]
    )

    assert expectedOperation == heater_operation


@pytest.mark.parametrize("hoursPerTimeStep", [0.25, 1])
def test_minimumDowntimeRequired(hoursPerTimeStep):
    """A profitable converter is required to be offline for downTimeMin hours."""
    esM = _create_system(
        numberOfTimeSteps=NUMBER_OF_HOURS / hoursPerTimeStep,
        hoursPerTimeStep=hoursPerTimeStep,
    )
    esM.updateComponent(
        "Methane heater",
        {
            "downTimeMin": 3,
            "minimumDowntimeRequired": True,
            "bigM": 1000,
        },
    )

    esM.optimize()

    heater_operation = (
        esM.getOptimizationSummary("ConversionDynamicModel")
        .loc["Methane heater", "operation", "[kW*h]"]
        .loc["region1"]
    )
    assert heater_operation == ENERGY_FLOW * (NUMBER_OF_HOURS - 3)


@pytest.mark.parametrize(
    "updates, error, message",
    [
        (
            {"minimumDowntimeRequired": 1},
            TypeError,
            "minimumDowntimeRequired must be a boolean",
        ),
        (
            {"minimumDowntimeRequired": True},
            ValueError,
            "downTimeMin needs to be specified",
        ),
        (
            {
                "minimumDowntimeRequired": True,
                "downTimeMin": 2,
                "bigM": 100,
                "useTemporalCyclicConstraints": False,
            },
            ValueError,
            "useTemporalCyclicConstraints=True",
        ),
    ],
)
def test_minimumDowntimeRequired_validation(updates, error, message):
    esM = _create_system(numberOfTimeSteps=NUMBER_OF_HOURS, hoursPerTimeStep=1)
    with pytest.raises(error, match=message):
        esM.updateComponent("Methane heater", updates)


@pytest.mark.parametrize(
    ["hoursPerTimeStep", "partLoadMin"],
    [(0.25, None), (1, None), (0.25, 0.5), (1, 0.5)],
)
def test_upTimeMin(hoursPerTimeStep, partLoadMin, useTemporalCyclicConstraints=True):
    """Test the uptime of the system.

    The heat production with a methane heater is cheaper and therefore preferred.

    The operationRateFix will be set fluctuation so the uptime cannot be kept.
    Only the first time steps allow for an operation of the heater.
    """
    esM = _create_system(
        numberOfTimeSteps=NUMBER_OF_HOURS / hoursPerTimeStep,
        hoursPerTimeStep=hoursPerTimeStep,
    )

    # Update ConversionDynamic: Use downTimeMin
    upTimeMin = 3
    esM.updateComponent(
        "Methane heater",
        {
            "upTimeMin": upTimeMin,
            "bigM": 1000,
            "partLoadMin": partLoadMin,
            "useTemporalCyclicConstraints": useTemporalCyclicConstraints,
        },
    )

    # Update Demand: Single time step to 0 so the heater must shut off
    opRateFix = esM.getComponent("Heat demand").operationRateFix.copy()
    numberOfTimeSteps = len(opRateFix)
    opRateFix[range(1, numberOfTimeSteps, 2)] = 0
    opRateFix[numberOfTimeSteps - 1] = 0
    opRateFix[
        range(
            0,
            int(upTimeMin / hoursPerTimeStep),
        )
    ] = ENERGY_FLOW
    opRateFix[int(upTimeMin / hoursPerTimeStep)] = 0
    esM.updateComponent("Heat demand", {"operationRateFix": opRateFix})

    # optimize
    esM.optimize()

    # Check results: operation of the methane heater must be 3h*10kW less
    expectedOperation = ENERGY_FLOW * upTimeMin
    heater_operation = (
        esM.getOptimizationSummary("ConversionDynamicModel")
        .loc["Methane heater", "operation", "[kW*h]"]
        .loc["region1"]
    )
    assert expectedOperation == heater_operation


@pytest.mark.parametrize(
    ["hoursPerTimeStep", "partLoadMin", "rampUpMax"],
    [
        (0.25, None, 0.5),
        (0.25, None, 1),
        (1, None, 0.5),
        (1, None, 1),
        (0.25, 0.5, 0.5),
        (0.25, 0.5, 1),
        (1, 0.5, 0.5),
        (1, 0.5, 1),
    ],
)
def test_rampUpMax(
    hoursPerTimeStep, partLoadMin, rampUpMax, useTemporalCyclicConstraints=False
):
    """Test the maximum ramping up of the component methane heater.

    The heat production with a methane heater is cheaper and therefore preferred.

    The operationRateFix of the demand will be set as a step function.
    Therefore, the component starts ramping as the demand starts, however, cannot fully fulfill the demand.
    """
    esM = _create_system(
        numberOfTimeSteps=NUMBER_OF_HOURS / hoursPerTimeStep,
        hoursPerTimeStep=hoursPerTimeStep,
    )

    # Update ConversionDynamic: Use downTimeMin
    esM.updateComponent(
        "Methane heater",
        {
            "rampUpMax": rampUpMax,
            "bigM": 10000,
            "partLoadMin": partLoadMin,
            "useTemporalCyclicConstraints": useTemporalCyclicConstraints,
        },
    )

    # Update Demand: Step function for the heat demand
    numberOfTimeStepsWithoutDemand = 4
    opRateFix = esM.getComponent("Heat demand").operationRateFix.copy()
    opRateFix[range(0, numberOfTimeStepsWithoutDemand)] = 0
    esM.updateComponent("Heat demand", {"operationRateFix": opRateFix})

    # optimize
    esM.optimize()

    # Check results
    if partLoadMin is not None and rampUpMax < partLoadMin:
        expectedOperation = 0
        expectedTimeStepOperation = 0
    else:
        # total operation
        rampingTimeSteps = 1 / (rampUpMax) - 1
        operationRamping = 0.5 * ENERGY_FLOW * rampingTimeSteps * hoursPerTimeStep
        timeStepsFullLoad = (
            NUMBER_OF_HOURS / hoursPerTimeStep
            - rampingTimeSteps
            - numberOfTimeStepsWithoutDemand
        )
        operationFullLoad = ENERGY_FLOW * timeStepsFullLoad * hoursPerTimeStep
        expectedOperation = operationRamping + operationFullLoad

        # specific time step operation of first ramping step
        expectedTimeStepOperation = rampUpMax * ENERGY_FLOW * hoursPerTimeStep

    heater_operation = (
        esM.getOptimizationSummary("ConversionDynamicModel")
        .loc["Methane heater", "operation", "[kW*h]"]
        .loc["region1"]
    )
    timeSeries = esM.componentModelingDict[
        "ConversionDynamicModel"
    ].operationVariablesOptimum
    specificTimeStepOperation = timeSeries.loc["Methane heater", "region1"][
        numberOfTimeStepsWithoutDemand
    ]

    assert expectedOperation == heater_operation
    assert expectedTimeStepOperation == specificTimeStepOperation


@pytest.mark.parametrize(
    ["hoursPerTimeStep", "partLoadMin", "rampDownMax"],
    [
        (0.25, None, 0.5),
        (0.25, None, 1),
        (1, None, 0.5),
        (1, None, 1),
        (0.25, 0.5, 0.5),
        (0.25, 0.5, 1),
        (1, 0.5, 0.5),
        (1, 0.5, 1),
    ],
)
def test_rampDownMax(hoursPerTimeStep, partLoadMin, rampDownMax):
    """Test the maximum ramping down of the component methane heater.

    The heat production with a methane heater is cheaper and therefore preferred.

    The operationRateFix of the demand will be set as a step function.
    Therefore, the component starts ramping down before the demand stops, as it cannot dump the produced heat.
    """
    numberOfTimeSteps = int(NUMBER_OF_HOURS / hoursPerTimeStep)
    esM = _create_system(
        numberOfTimeSteps=numberOfTimeSteps,
        hoursPerTimeStep=hoursPerTimeStep,
    )

    # Update ConversionDynamic: Use rampDownMax
    esM.updateComponent(
        "Methane heater",
        {"rampDownMax": rampDownMax, "bigM": 1000, "partLoadMin": partLoadMin},
    )

    # Update Demand: Step function for the heat demand
    numberOfTimeStepsWithoutDemand = 5
    opRateFix = esM.getComponent("Heat demand").operationRateFix.copy()
    opRateFix[
        range(numberOfTimeSteps - numberOfTimeStepsWithoutDemand, numberOfTimeSteps)
    ] = 0
    esM.updateComponent("Heat demand", {"operationRateFix": opRateFix})

    # optimize
    esM.optimize()

    # Check results
    if partLoadMin is not None and rampDownMax < partLoadMin:
        expectedOperation = 0
        expectedTimeStepOperation = 0
    else:
        rampingTimeSteps = 1 / (rampDownMax) - 1
        operationRamping = 0.5 * ENERGY_FLOW * rampingTimeSteps * hoursPerTimeStep
        timeStepsFullLoad = (
            numberOfTimeSteps - rampingTimeSteps - numberOfTimeStepsWithoutDemand
        )
        operationFullLoad = ENERGY_FLOW * timeStepsFullLoad * hoursPerTimeStep
        expectedOperation = operationRamping + operationFullLoad

        # specific time step operation of first ramping step
        expectedTimeStepOperation = rampDownMax * ENERGY_FLOW * hoursPerTimeStep

    heater_operation = (
        esM.getOptimizationSummary("ConversionDynamicModel")
        .loc["Methane heater", "operation", "[kW*h]"]
        .loc["region1"]
    )

    timeSeries = esM.componentModelingDict[
        "ConversionDynamicModel"
    ].operationVariablesOptimum
    specificTimeStepOperation = timeSeries.loc["Methane heater", "region1"].iloc[
        -numberOfTimeStepsWithoutDemand - 1
    ]

    assert expectedOperation == heater_operation
    assert expectedTimeStepOperation == specificTimeStepOperation


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


def test_ConversionDynamicCommissioningDependent():
    esM = fn.EnergySystemModel(
        locations={
            "example_region1",
        },
        commodities={"electricity", "methane"},
        commodityUnitsDict={"electricity": r"GW$_{el}$", "methane": r"GW$_{th}$"},
        verboseLogLevel=2,
        numberOfInvestmentPeriods=2,
    )
    error_msg = "Currently commissioning-depending constraints are not possible"
    with pytest.raises(ValueError, match=error_msg):
        fn.ConversionDynamic(
            esM=esM,
            name="restricted",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={
                (0, 0): {"electricity": 1, "methane": -1 / 0.6},
                (0, 1): {"electricity": 1, "methane": -1 / 0.65},
                (1, 1): {"electricity": 1, "methane": -1 / 0.7},
            },
            partLoadMin=0.3,
            bigM=100,
            rampDownMax=0.5,
        )
