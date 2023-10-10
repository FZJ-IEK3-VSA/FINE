import FINE as fn
import numpy as np
import pandas as pd
import pytest

np.random.seed(
    42
)  # Sets a "seed" to produce the same random input data in each model run


def test_CO2ReductionTargets():
    locations = {"regionN", "regionS"}
    commodityUnitDict = {
        "electricity": r"GW$_{el}$",
        "naturalGas": r"GW$_{CH_{4},LHV}$",
        "CO2": r"Mio. t$_{CO_2}$/h",
    }
    commodities = {"electricity", "naturalGas", "CO2"}
    numberOfTimeSteps, hoursPerTimeStep = 8760, 1
    costUnit, lengthUnit = "1e6 Euro", "km"
    CO2_reductionTarget = 0.8

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit=costUnit,
        lengthUnit=lengthUnit,
        verboseLogLevel=0,
    )

    # Add Source Components
    ## Wind turbines
    name, commodity = "Wind turbines", "electricity"
    hasCapacityVariable = True
    operationRateMax = pd.DataFrame(
        [[np.random.beta(a=2, b=7.5), np.random.beta(a=2, b=9)] for t in range(8760)],
        index=range(8760),
        columns=["regionN", "regionS"],
    ).round(6)
    capacityMax = pd.Series([400, 200], index=["regionN", "regionS"])
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20

    esM.add(
        fn.Source(
            esM=esM,
            name=name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
            operationRateMax=operationRateMax,
            capacityMax=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    ## PV
    name, commodity = "PV", "electricity"
    hasCapacityVariable = True
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
    operationRateMax = pd.DataFrame(
        [[u, u] for day in range(365) for u in dailyProfileSimple],
        index=range(8760),
        columns=["regionN", "regionS"],
    )
    capacityMax = pd.Series([100, 100], index=["regionN", "regionS"])
    investPerCapacity, opexPerCapacity = 800, 800 * 0.02
    interestRate, economicLifetime = 0.08, 25

    esM.add(
        fn.Source(
            esM=esM,
            name=name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
            operationRateMax=operationRateMax,
            capacityMax=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Natural Gas
    name, commodity = "Natural gas import", "naturalGas"
    hasCapacityVariable = False
    commodityCost = 0.03

    esM.add(
        fn.Source(
            esM=esM,
            name=name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
            commodityCost=commodityCost,
        )
    )

    # Add Conversion components
    # Gas power plants
    name, physicalUnit = "Gas power plants", r"GW$_{el}$"
    commodityConversionFactors = {
        "electricity": 1,
        "naturalGas": -1 / 0.63,
        "CO2": 201 * 1e-6 / 0.63,
    }
    hasCapacityVariable = True
    investPerCapacity, opexPerCapacity = 650, 650 * 0.03
    interestRate, economicLifetime = 0.08, 30

    esM.add(
        fn.Conversion(
            esM=esM,
            name=name,
            physicalUnit=physicalUnit,
            commodityConversionFactors=commodityConversionFactors,
            hasCapacityVariable=hasCapacityVariable,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Storage Components
    ## Batteries
    name, commodity = "Batteries", "electricity"
    hasCapacityVariable = True
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 150, 150 * 0.01
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000

    esM.add(
        fn.Storage(
            esM=esM,
            name=name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
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

    # Transmission Components
    ## AC cables
    name, commodity = "AC cables", "electricity"
    hasCapacityVariable = True
    capacityFix = pd.DataFrame(
        [[0, 30], [30, 0]], columns=["regionN", "regionS"], index=["regionN", "regionS"]
    )
    distances = pd.DataFrame(
        [[0, 400], [400, 0]],
        columns=["regionN", "regionS"],
        index=["regionN", "regionS"],
    )
    losses = 0.0001

    esM.add(
        fn.Transmission(
            esM=esM,
            name=name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
            capacityFix=capacityFix,
            distances=distances,
            losses=losses,
        )
    )

    # Sink Components
    ## Electricity Demand
    name, commodity = (
        "Electricity demand",
        "electricity",
    )
    hasCapacityVariable = False
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
    operationRateFix = pd.DataFrame(
        [
            [(u + 0.1 * np.random.rand()) * 25, (u + 0.1 * np.random.rand()) * 40]
            for day in range(365)
            for u in dailyProfileSimple
        ],
        index=range(8760),
        columns=["regionN", "regionS"],
    ).round(2)

    esM.add(
        fn.Sink(
            esM=esM,
            name=name,
            commodity=commodity,
            hasCapacityVariable=hasCapacityVariable,
            operationRateFix=operationRateFix,
        )
    )

    # CO2 to environment
    name, commodity = (
        "CO2 to environment",
        "CO2",
    )
    hasCapacityVariable = False
    commodityLimitID, yearlyLimit = "CO2 limit", 366 * (1 - CO2_reductionTarget)

    if yearlyLimit > 0:
        esM.add(
            fn.Sink(
                esM=esM,
                name=name,
                commodity=commodity,
                hasCapacityVariable=hasCapacityVariable,
                commodityLimitID=commodityLimitID,
                yearlyLimit=yearlyLimit,
            )
        )

    # Optimize the system with simple myopic approach
    results = fn.optimizeSimpleMyopic(
        esM,
        startYear=2020,
        nbOfSteps=2,
        nbOfRepresentedYears=5,
        CO2Reference=366,
        CO2ReductionTargets=[25, 50, 100],
        saveResults=False,
        trackESMs=True,
        numberOfTypicalPeriods=3,
        solver="glpk",
    )

    assert (
        results["ESM_2025"]
        .getOptimizationSummary("SourceSinkModel")
        .loc["CO2 to environment"]
        .loc["operation", "[Mio. t$_{CO_2}$/h*h/a]"]
        .sum()
        < 183
    )

    assert (
        results["ESM_2030"]
        .getOptimizationSummary("SourceSinkModel")
        .loc["CO2 to environment"]
        .loc["operation", "[Mio. t$_{CO_2}$/h*h/a]"]
        .sum()
        == 0
    )


@pytest.mark.skip()
def test_exceededLifetime():
    # load a minimal test system
    """Returns minimal instance of esM"""

    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
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
        lengthUnit="km",
        verboseLogLevel=2,
    )

    ### Buy electricity at the electricity market
    costs = pd.DataFrame(
        [
            np.array(
                [
                    0.05,
                    0.0,
                    0.1,
                    0.051,
                ]
            )
        ],
        index=["OneLocation"],
    ).T
    revenues = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                ]
            )
        ],
        index=["OneLocation"],
    ).T
    maxpurchase = (
        pd.DataFrame(
            [
                np.array(
                    [
                        1e6,
                        1e6,
                        1e6,
                        1e6,
                    ]
                )
            ],
            index=["OneLocation"],
        ).T
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
    )  # eur/kWh

    ### Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### Hydrogen filled somewhere
    esM.add(
        fn.Storage(
            esM=esM,
            name="Pressure tank",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ### Industry site
    demand = (
        pd.DataFrame(
            [
                np.array(
                    [
                        6e3,
                        6e3,
                        6e3,
                        6e3,
                    ]
                )
            ],
            index=["OneLocation"],
        ).T
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

    # Set the technical lifetime of the electrolyzers to 7 years.
    setattr(
        esM.componentModelingDict["ConversionModel"].componentsDict["Electrolyzers"],
        "technicalLifetime",
        pd.Series([7], index=["OneLocation"]),
    )

    results = fn.optimizeSimpleMyopic(
        esM,
        startYear=2020,
        endYear=2030,
        nbOfRepresentedYears=5,
        timeSeriesAggregation=False,
        solver="glpk",
        saveResults=False,
        trackESMs=True,
    )

    # Check if electrolyzers which are installed in 2020 are not included in the system of 2030 due to the exceeded lifetime
    assert "Electrolyzers_stock_2020" not in results["ESM_2030"].componentNames.keys()
