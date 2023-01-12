import pytest
import sys
import os

import numpy as np
import pandas as pd

import FINE as fn

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "Multi-regional_Energy_System_Workflow",
    )
)
from getData import getData


@pytest.fixture
def minimal_test_esM(scope="session"):
    """Returns minimal instance of esM"""

    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation", "IndustryLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
        lowerBound=False,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

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
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
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
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
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
                ),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ],
            index=["ElectrolyzerLocation", "IndustryLocation"],
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

    ### Hydrogen pipelines
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines",
            commodity="hydrogen",
            hasCapacityVariable=True,
            investPerCapacity=0.177,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    ### Industry site
    demand = (
        pd.DataFrame(
            [
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                np.array(
                    [
                        6e3,
                        6e3,
                        6e3,
                        6e3,
                    ]
                ),
            ],
            index=["ElectrolyzerLocation", "IndustryLocation"],
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

    return esM


@pytest.fixture
def single_node_test_esM():
    """Returns minimal instance of esM with one node"""

    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"Location"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
        lowerBound=False,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    ### Buy electricity at the electricity market
    costs = pd.Series(
        np.array(
            [
                0.05,
                0.0,
                0.1,
                0.051,
            ]
        )
    )
    revenues = pd.Series(
        np.array(
            [
                0.0,
                0.01,
                0.0,
                0.0,
            ]
        )
    )
    maxpurchase = pd.Series(
        np.array(
            [
                1e6,
                1e6,
                1e6,
                1e6,
            ]
        )
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
    demand = pd.Series(
        np.array(
            [
                6e3,
                6e3,
                6e3,
                6e3,
            ]
        )
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

    return esM


@pytest.fixture(scope="session")
def esM_init():

    locations = {
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
        "cluster_4",
        "cluster_5",
        "cluster_6",
        "cluster_7",
    }
    commodityUnitDict = {
        "electricity": r"GW$_{el}$",
        "methane": r"GW$_{CH_{4},LHV}$",
        "biogas": r"GW$_{biogas,LHV}$",
        "CO2": r"Mio. t$_{CO_2}$/h",
        "hydrogen": r"GW$_{H_{2},LHV}$",
    }
    commodities = {"electricity", "hydrogen", "methane", "biogas", "CO2"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=8760,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    return esM


@pytest.fixture(scope="session")
def multi_node_test_esM_init(esM_init):
    data = getData()

    # 2. Create an energy system model instance
    esM = esM_init

    CO2_reductionTarget = 1

    # 3. Add commodity sources to the energy system model
    ## 3.1. Electricity sources
    ### Wind onshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (onshore), operationRateMax"],
            capacityMax=data["Wind (onshore), capacityMax"],
            investPerCapacity=1.1,
            opexPerCapacity=1.1 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    data["Wind (onshore), operationRateMax"].sum()

    ### Wind offshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (offshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (offshore), operationRateMax"],
            capacityMax=data["Wind (offshore), capacityMax"],
            investPerCapacity=2.3,
            opexPerCapacity=2.3 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    data["Wind (offshore), operationRateMax"].sum()

    ### PV

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["PV, operationRateMax"],
            capacityMax=data["PV, capacityMax"],
            investPerCapacity=0.65,
            opexPerCapacity=0.65 * 0.02,
            interestRate=0.08,
            economicLifetime=25,
        )
    )

    data["PV, operationRateMax"].sum()

    ### Exisisting run-of-river hydroelectricity plants

    esM.add(
        fn.Source(
            esM=esM,
            name="Existing run-of-river plants",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=data["Existing run-of-river plants, operationRateFix"],
            tsaWeight=0.01,
            capacityFix=data["Existing run-of-river plants, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.208,
        )
    )

    ## 3.2. Methane (natural gas and biogas)
    ### Natural gas
    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Natural Gas, commodityCostTimeSeries"],
        )
    )

    ### Biogas
    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas purchase",
            commodity="biogas",
            operationRateMax=data["Biogas, operationRateMax"],
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Biogas, commodityCostTimeSeries"],
        )
    )

    ## 3.3 CO2
    ### CO2

    esM.add(
        fn.Source(
            esM=esM,
            name="CO2 from enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # 4. Add conversion components to the energy system model

    ### Combined cycle gas turbine plants

    esM.add(
        fn.Conversion(
            esM=esM,
            name="CCGT plants (methane)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "methane": -1 / 0.625,
                "CO2": 201 * 1e-6 / 0.625,
            },
            hasCapacityVariable=True,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### New combined cycle gas turbine plants for biogas

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (biogas)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.635},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### New combined cycly gas turbines for hydrogen

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (hydrogen)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### Electrolyzers

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electroylzers",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### rSOC

    capexRSOC = 1.5

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOEC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOFC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # 5. Add commodity storages to the energy system model
    ## 5.1. Electricity storage
    ### Lithium ion batteries

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.95,
            cyclicLifetime=10000,
            dischargeEfficiency=0.95,
            selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
        )
    )

    ## 5.2. Hydrogen storage
    ### Hydrogen filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (hydrogen)",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=133,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (hydrogen), capacityMax"],
            investPerCapacity=0.00011,
            opexPerCapacity=0.00057,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## 5.3. Methane storage
    ### Methane filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (biogas)",
            commodity="biogas",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=443,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (methane), capacityMax"],
            investPerCapacity=0.00004,
            opexPerCapacity=0.00001,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## 5.4 Pumped hydro storage
    ### Pumped hydro storage

    esM.add(
        fn.Storage(
            esM=esM,
            name="Pumped hydro storage",
            commodity="electricity",
            chargeEfficiency=0.88,
            dischargeEfficiency=0.88,
            hasCapacityVariable=True,
            selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
            chargeRate=0.16,
            dischargeRate=0.12,
            capacityFix=data["Pumped hydro storage, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.000153,
        )
    )

    # 6. Add commodity transmission components to the energy system model
    ## 6.1. Electricity transmission
    ### AC cables

    esM.add(
        fn.LinearOptimalPowerFlow(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=data["AC cables, capacityFix"],
            reactances=data["AC cables, reactances"],
        )
    )

    ### DC cables

    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=data["DC cables, losses"],
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
        )
    )

    ## 6.2 Methane transmission
    ### Methane pipeline

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (biogas)",
            commodity="biogas",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000037,
            investIfBuilt=0.000314,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    ## 6.3 Hydrogen transmission
    ### Hydrogen pipelines

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (hydrogen)",
            commodity="hydrogen",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000177,
            investIfBuilt=0.00033,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    # 7. Add commodity sinks to the energy system model
    ## 7.1. Electricity sinks
    ### Electricity demand

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=data["Electricity demand, operationRateFix"],
        )
    )

    ## 7.2. Hydrogen sinks
    ### Fuel cell electric vehicle (FCEV) demand

    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
        )
    )

    ## 7.3. CO2 sinks
    ### CO2 exiting the system's boundary

    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    return esM


@pytest.fixture(scope="session")
def test_esM_for_spagat(esM_init):
    """
    Simpler version of multi_node_test_esM_init.
    Makes spagat tests faster.
    """
    data = getData()

    # Create an energy system model instance
    esM = esM_init

    # onshore wind
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (onshore), operationRateMax"],
            capacityMax=data["Wind (onshore), capacityMax"],
            investPerCapacity=1.1,
            opexPerCapacity=1.1 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    # CO2 from environment
    CO2_reductionTarget = 1
    esM.add(
        fn.Source(
            esM=esM,
            name="CO2 from enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electroylzers",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # Pumped hydro storage
    esM.add(
        fn.Storage(
            esM=esM,
            name="Pumped hydro storage",
            commodity="electricity",
            chargeEfficiency=0.88,
            dischargeEfficiency=0.88,
            hasCapacityVariable=True,
            selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
            chargeRate=0.16,
            dischargeRate=0.12,
            capacityFix=data["Pumped hydro storage, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.000153,
        )
    )

    # DC cables
    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=data["DC cables, losses"],
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
        )
    )

    # Hydrogen sinks
    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
        )
    )

    return esM


@pytest.fixture(scope="session")
def multi_node_test_esM_optimized(esM_init):
    data = getData(esM_init)

    # 2. Create an energy system model instance
    esM = esM_init

    CO2_reductionTarget = 1

    # 3. Add commodity sources to the energy system model
    ## 3.1. Electricity sources
    ### Wind onshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (onshore), operationRateMax"],
            capacityMax=data["Wind (onshore), capacityMax"],
            investPerCapacity=1.1,
            opexPerCapacity=1.1 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    data["Wind (onshore), operationRateMax"].sum()

    ### Wind offshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (offshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (offshore), operationRateMax"],
            capacityMax=data["Wind (offshore), capacityMax"],
            investPerCapacity=2.3,
            opexPerCapacity=2.3 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    data["Wind (offshore), operationRateMax"].sum()

    ### PV

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["PV, operationRateMax"],
            capacityMax=data["PV, capacityMax"],
            investPerCapacity=0.65,
            opexPerCapacity=0.65 * 0.02,
            interestRate=0.08,
            economicLifetime=25,
        )
    )

    data["PV, operationRateMax"].sum()

    ### Exisisting run-of-river hydroelectricity plants

    esM.add(
        fn.Source(
            esM=esM,
            name="Existing run-of-river plants",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=data["Existing run-of-river plants, operationRateFix"],
            tsaWeight=0.01,
            capacityFix=data["Existing run-of-river plants, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.208,
        )
    )

    ## 3.2. Methane (natural gas and biogas)
    ### Natural gas
    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Natural Gas, commodityCostTimeSeries"],
        )
    )

    ### Biogas
    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas purchase",
            commodity="biogas",
            operationRateMax=data["Biogas, operationRateMax"],
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Biogas, commodityCostTimeSeries"],
        )
    )

    ## 3.3 CO2
    ### CO2

    esM.add(
        fn.Source(
            esM=esM,
            name="CO2 from enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # 4. Add conversion components to the energy system model

    ### Combined cycle gas turbine plants

    esM.add(
        fn.Conversion(
            esM=esM,
            name="CCGT plants (methane)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "methane": -1 / 0.625,
                "CO2": 201 * 1e-6 / 0.625,
            },
            hasCapacityVariable=True,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### New combined cycle gas turbine plants for biogas

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (biogas)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.635},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### New combined cycly gas turbines for hydrogen

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (hydrogen)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### Electrolyzers

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electroylzers",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### rSOC

    capexRSOC = 1.5

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOEC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOFC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # 5. Add commodity storages to the energy system model
    ## 5.1. Electricity storage
    ### Lithium ion batteries

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.95,
            cyclicLifetime=10000,
            dischargeEfficiency=0.95,
            selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
        )
    )

    ## 5.2. Hydrogen storage
    ### Hydrogen filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (hydrogen)",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=133,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (hydrogen), capacityMax"],
            investPerCapacity=0.00011,
            opexPerCapacity=0.00057,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## 5.3. Methane storage
    ### Methane filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (biogas)",
            commodity="biogas",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=443,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (methane), capacityMax"],
            investPerCapacity=0.00004,
            opexPerCapacity=0.00001,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## 5.4 Pumped hydro storage
    ### Pumped hydro storage

    esM.add(
        fn.Storage(
            esM=esM,
            name="Pumped hydro storage",
            commodity="electricity",
            chargeEfficiency=0.88,
            dischargeEfficiency=0.88,
            hasCapacityVariable=True,
            selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
            chargeRate=0.16,
            dischargeRate=0.12,
            capacityFix=data["Pumped hydro storage, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.000153,
        )
    )

    # 6. Add commodity transmission components to the energy system model
    ## 6.1. Electricity transmission
    ### AC cables

    esM.add(
        fn.LinearOptimalPowerFlow(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=data["AC cables, capacityFix"],
            reactances=data["AC cables, reactances"],
        )
    )

    ### DC cables

    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=data["DC cables, losses"],
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
        )
    )

    ## 6.2 Methane transmission
    ### Methane pipeline

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (biogas)",
            commodity="biogas",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000037,
            investIfBuilt=0.000314,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    ## 6.3 Hydrogen transmission
    ### Hydrogen pipelines

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (hydrogen)",
            commodity="hydrogen",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000177,
            investIfBuilt=0.00033,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    # 7. Add commodity sinks to the energy system model
    ## 7.1. Electricity sinks
    ### Electricity demand

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=data["Electricity demand, operationRateFix"],
        )
    )

    ## 7.2. Hydrogen sinks
    ### Fuel cell electric vehicle (FCEV) demand

    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
        )
    )

    ## 7.3. CO2 sinks
    ### CO2 exiting the system's boundary

    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # 8. Optimize energy system model

    esM.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    return esM


@pytest.fixture
def multi_node_test_esM_init(scope="session"):
    data = getData()

    # 2. Create an energy system model instance
    locations = {
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
        "cluster_4",
        "cluster_5",
        "cluster_6",
        "cluster_7",
    }
    commodityUnitDict = {
        "electricity": r"GW$_{el}$",
        "methane": r"GW$_{CH_{4},LHV}$",
        "biogas": r"GW$_{biogas,LHV}$",
        "CO2": r"Mio. t$_{CO_2}$/h",
        "hydrogen": r"GW$_{H_{2},LHV}$",
    }
    commodities = {"electricity", "hydrogen", "methane", "biogas", "CO2"}
    numberOfTimeSteps = 8760
    hoursPerTimeStep = 1

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=8760,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    CO2_reductionTarget = 1

    # 3. Add commodity sources to the energy system model
    ## 3.1. Electricity sources
    ### Wind onshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (onshore), operationRateMax"],
            capacityMax=data["Wind (onshore), capacityMax"],
            investPerCapacity=1.1,
            opexPerCapacity=1.1 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
            opexPerOperation=0.008,
        )
    )

    data["Wind (onshore), operationRateMax"].sum()

    ### Wind offshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (offshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (offshore), operationRateMax"],
            capacityMax=data["Wind (offshore), capacityMax"],
            investPerCapacity=2.3,
            opexPerCapacity=2.3 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
            opexPerOperation=0.005,
        )
    )

    data["Wind (offshore), operationRateMax"].sum()

    ### PV

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["PV, operationRateMax"],
            capacityMax=data["PV, capacityMax"],
            investPerCapacity=0.65,
            opexPerCapacity=0.65 * 0.02,
            interestRate=0.08,
            economicLifetime=25,
            opexPerOperation=0.01,
        )
    )

    data["PV, operationRateMax"].sum()

    ### Exisisting run-of-river hydroelectricity plants

    esM.add(
        fn.Source(
            esM=esM,
            name="Existing run-of-river plants",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=data["Existing run-of-river plants, operationRateFix"],
            tsaWeight=0.01,
            capacityFix=data["Existing run-of-river plants, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.208,
            opexPerOperation=0.005,
        )
    )

    ## 3.2. Methane (natural gas and biogas)
    ### Natural gas
    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Natural Gas, commodityCostTimeSeries"],
        )
    )

    ### Biogas
    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas purchase",
            commodity="biogas",
            operationRateMax=data["Biogas, operationRateMax"],
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Biogas, commodityCostTimeSeries"],
        )
    )

    ## 3.3 CO2
    ### CO2

    esM.add(
        fn.Source(
            esM=esM,
            name="CO2 from enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # 4. Add conversion components to the energy system model

    ### Combined cycle gas turbine plants

    esM.add(
        fn.Conversion(
            esM=esM,
            name="CCGT plants (methane)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "methane": -1 / 0.625,
                "CO2": 201 * 1e-6 / 0.625,
            },
            hasCapacityVariable=True,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
            opexPerOperation=0.005,
        )
    )

    ### New combined cycle gas turbine plants for biogas

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (biogas)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.635},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
            opexPerOperation=0.01,
        )
    )

    ### New combined cycly gas turbines for hydrogen

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (hydrogen)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
            opexPerOperation=0.01,
        )
    )

    ### Electrolyzers

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electroylzers",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            opexPerOperation=0.01,
        )
    )

    ### rSOC

    capexRSOC = 1.5

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOEC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
            opexPerOperation=0.01,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOFC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
            opexPerOperation=0.01,
        )
    )

    # 5. Add commodity storages to the energy system model
    ## 5.1. Electricity storage
    ### Lithium ion batteries

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.95,
            cyclicLifetime=10000,
            dischargeEfficiency=0.95,
            selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
            opexPerChargeOperation=0.0001,
        )
    )

    ## 5.2. Hydrogen storage
    ### Hydrogen filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (hydrogen)",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=133,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (hydrogen), capacityMax"],
            investPerCapacity=0.00011,
            opexPerCapacity=0.00057,
            interestRate=0.08,
            economicLifetime=30,
            opexPerChargeOperation=0.0001,
        )
    )

    ## 5.3. Methane storage
    ### Methane filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (biogas)",
            commodity="biogas",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=443,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (methane), capacityMax"],
            investPerCapacity=0.00004,
            opexPerCapacity=0.00001,
            interestRate=0.08,
            economicLifetime=30,
            opexPerChargeOperation=0.0001,
        )
    )

    ## 5.4 Pumped hydro storage
    ### Pumped hydro storage

    esM.add(
        fn.Storage(
            esM=esM,
            name="Pumped hydro storage",
            commodity="electricity",
            chargeEfficiency=0.88,
            dischargeEfficiency=0.88,
            hasCapacityVariable=True,
            selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
            chargeRate=0.16,
            dischargeRate=0.12,
            capacityFix=data["Pumped hydro storage, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.000153,
            opexPerChargeOperation=0.0001,
        )
    )

    # 6. Add commodity transmission components to the energy system model
    ## 6.1. Electricity transmission
    ### AC cables

    esM.add(
        fn.LinearOptimalPowerFlow(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=data["AC cables, capacityFix"],
            reactances=data["AC cables, reactances"],
            opexPerOperation=0.01,
        )
    )

    ### DC cables

    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=data["DC cables, losses"],
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
            opexPerOperation=0.01,
        )
    )

    ## 6.2 Methane transmission
    ### Methane pipeline

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (biogas)",
            commodity="biogas",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000037,
            investIfBuilt=0.000314,
            interestRate=0.08,
            economicLifetime=40,
            opexPerOperation=0.01,
        )
    )

    ## 6.3 Hydrogen transmission
    ### Hydrogen pipelines

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (hydrogen)",
            commodity="hydrogen",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000177,
            investIfBuilt=0.00033,
            interestRate=0.08,
            economicLifetime=40,
            opexPerOperation=0.01,
        )
    )

    # 7. Add commodity sinks to the energy system model
    ## 7.1. Electricity sinks
    ### Electricity demand

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=data["Electricity demand, operationRateFix"],
        )
    )

    ## 7.2. Hydrogen sinks
    ### Fuel cell electric vehicle (FCEV) demand

    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
        )
    )

    ## 7.3. CO2 sinks
    ### CO2 exiting the system's boundary

    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    return esM


@pytest.fixture
def multi_node_test_esM_optimized(scope="session"):
    cwd = os.getcwd()
    data = getData()

    # 2. Create an energy system model instance
    locations = {
        "cluster_0",
        "cluster_1",
        "cluster_2",
        "cluster_3",
        "cluster_4",
        "cluster_5",
        "cluster_6",
        "cluster_7",
    }
    commodityUnitDict = {
        "electricity": r"GW$_{el}$",
        "methane": r"GW$_{CH_{4},LHV}$",
        "biogas": r"GW$_{biogas,LHV}$",
        "CO2": r"Mio. t$_{CO_2}$/h",
        "hydrogen": r"GW$_{H_{2},LHV}$",
    }
    commodities = {"electricity", "hydrogen", "methane", "biogas", "CO2"}
    numberOfTimeSteps = 8760
    hoursPerTimeStep = 1

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=8760,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
        balanceLimit=None,
        lowerBound=False,
    )

    CO2_reductionTarget = 1

    # 3. Add commodity sources to the energy system model
    ## 3.1. Electricity sources
    ### Wind onshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (onshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (onshore), operationRateMax"],
            capacityMax=data["Wind (onshore), capacityMax"],
            investPerCapacity=1.1,
            opexPerCapacity=1.1 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    data["Wind (onshore), operationRateMax"].sum()

    ### Wind offshore

    esM.add(
        fn.Source(
            esM=esM,
            name="Wind (offshore)",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["Wind (offshore), operationRateMax"],
            capacityMax=data["Wind (offshore), capacityMax"],
            investPerCapacity=2.3,
            opexPerCapacity=2.3 * 0.02,
            interestRate=0.08,
            economicLifetime=20,
        )
    )

    data["Wind (offshore), operationRateMax"].sum()

    ### PV

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=data["PV, operationRateMax"],
            capacityMax=data["PV, capacityMax"],
            investPerCapacity=0.65,
            opexPerCapacity=0.65 * 0.02,
            interestRate=0.08,
            economicLifetime=25,
        )
    )

    data["PV, operationRateMax"].sum()

    ### Exisisting run-of-river hydroelectricity plants

    esM.add(
        fn.Source(
            esM=esM,
            name="Existing run-of-river plants",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=data["Existing run-of-river plants, operationRateFix"],
            tsaWeight=0.01,
            capacityFix=data["Existing run-of-river plants, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.208,
        )
    )

    ## 3.2. Methane (natural gas and biogas)
    ### Natural gas
    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Natural Gas, commodityCostTimeSeries"],
        )
    )

    ### Biogas
    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas purchase",
            commodity="biogas",
            operationRateMax=data["Biogas, operationRateMax"],
            hasCapacityVariable=False,
            commodityCostTimeSeries=data["Biogas, commodityCostTimeSeries"],
        )
    )

    ## 3.3 CO2
    ### CO2

    esM.add(
        fn.Source(
            esM=esM,
            name="CO2 from enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # 4. Add conversion components to the energy system model

    ### Combined cycle gas turbine plants

    esM.add(
        fn.Conversion(
            esM=esM,
            name="CCGT plants (methane)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "methane": -1 / 0.625,
                "CO2": 201 * 1e-6 / 0.625,
            },
            hasCapacityVariable=True,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### New combined cycle gas turbine plants for biogas

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (biogas)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.635},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### New combined cycly gas turbines for hydrogen

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (hydrogen)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    ### Electrolyzers

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electroylzers",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### rSOC

    capexRSOC = 1.5

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOEC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="rSOFC",
            physicalUnit=r"GW$_{el}$",
            linkedConversionCapacityID="rSOC",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.6},
            hasCapacityVariable=True,
            investPerCapacity=capexRSOC / 2,
            opexPerCapacity=capexRSOC * 0.02 / 2,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    # 5. Add commodity storages to the energy system model
    ## 5.1. Electricity storage
    ### Lithium ion batteries

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=0.95,
            cyclicLifetime=10000,
            dischargeEfficiency=0.95,
            selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
        )
    )

    ## 5.2. Hydrogen storage
    ### Hydrogen filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (hydrogen)",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=133,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (hydrogen), capacityMax"],
            investPerCapacity=0.00011,
            opexPerCapacity=0.00057,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## 5.3. Methane storage
    ### Methane filled salt caverns

    esM.add(
        fn.Storage(
            esM=esM,
            name="Salt caverns (biogas)",
            commodity="biogas",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityPerPlantUnit=443,
            chargeRate=1 / 470.37,
            dischargeRate=1 / 470.37,
            sharedPotentialID="Existing salt caverns",
            stateOfChargeMin=0.33,
            stateOfChargeMax=1,
            capacityMax=data["Salt caverns (methane), capacityMax"],
            investPerCapacity=0.00004,
            opexPerCapacity=0.00001,
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## 5.4 Pumped hydro storage
    ### Pumped hydro storage

    esM.add(
        fn.Storage(
            esM=esM,
            name="Pumped hydro storage",
            commodity="electricity",
            chargeEfficiency=0.88,
            dischargeEfficiency=0.88,
            hasCapacityVariable=True,
            selfDischarge=1 - (1 - 0.00375) ** (1 / (30 * 24)),
            chargeRate=0.16,
            dischargeRate=0.12,
            capacityFix=data["Pumped hydro storage, capacityFix"],
            investPerCapacity=0,
            opexPerCapacity=0.000153,
        )
    )

    # 6. Add commodity transmission components to the energy system model
    ## 6.1. Electricity transmission
    ### AC cables

    esM.add(
        fn.LinearOptimalPowerFlow(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=data["AC cables, capacityFix"],
            reactances=data["AC cables, reactances"],
        )
    )

    ### DC cables

    esM.add(
        fn.Transmission(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            losses=data["DC cables, losses"],
            distances=data["DC cables, distances"],
            hasCapacityVariable=True,
            capacityFix=data["DC cables, capacityFix"],
        )
    )

    ## 6.2 Methane transmission
    ### Methane pipeline

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (biogas)",
            commodity="biogas",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000037,
            investIfBuilt=0.000314,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    ## 6.3 Hydrogen transmission
    ### Hydrogen pipelines

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (hydrogen)",
            commodity="hydrogen",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=False,
            bigM=300,
            locationalEligibility=data["Pipelines, eligibility"],
            capacityMax=data["Pipelines, eligibility"] * 15,
            sharedPotentialID="pipelines",
            investPerCapacity=0.000177,
            investIfBuilt=0.00033,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    # 7. Add commodity sinks to the energy system model
    ## 7.1. Electricity sinks
    ### Electricity demand

    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=data["Electricity demand, operationRateFix"],
        )
    )

    ## 7.2. Hydrogen sinks
    ### Fuel cell electric vehicle (FCEV) demand

    FCEV_penetration = 0.5
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=data["Hydrogen demand, operationRateFix"]
            * FCEV_penetration,
        )
    )

    ## 7.3. CO2 sinks
    ### CO2 exiting the system's boundary

    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            commodityLimitID="CO2 limit",
            yearlyLimit=366 * (1 - CO2_reductionTarget),
        )
    )

    # 8. Optimize energy system model

    esM.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    return esM


@pytest.fixture
def dsm_test_esM(scope="session"):
    """
    Generate a simple energy system model with one node, two fixed generators and one load time series
    for testing demand side management functionality.
    """
    # load without dsm
    now = pd.Timestamp.now().round("h")
    number_of_time_steps = 28
    # t_index = pd.date_range(now, now + pd.DateOffset(hours=number_of_timeSteps - 1), freq='h')
    t_index = range(number_of_time_steps)
    load_without_dsm = pd.Series([80.0] * number_of_time_steps, index=t_index)

    timestep_up = 10
    timestep_down = 20
    load_without_dsm[timestep_up:timestep_down] += 40.0

    time_shift = 3
    cheap_capacity = 100.0
    expensive_capacity = 20.0

    # set up energy model
    esM = fn.EnergySystemModel(
        locations={"location"},
        commodities={"electricity"},
        numberOfTimeSteps=number_of_time_steps,
        commodityUnitsDict={"electricity": r"MW$_{el}$"},
        hoursPerTimeStep=1,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="cheap",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(cheap_capacity, index=t_index),
            opexPerOperation=25,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="expensive",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(expensive_capacity, index=t_index),
            opexPerOperation=50,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="back-up",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.Series(1000, index=t_index),
            opexPerOperation=1000,
        )
    )

    return esM, load_without_dsm, timestep_up, timestep_down, time_shift, cheap_capacity


@pytest.fixture
def balanceLimitConstraint_test_esM():
    # 0) Preprocess energy system model
    locations = {"Region1", "Region2"}
    commodityUnitDict = {"electricity": r"MW$_{el}$", "heat": r"MW$_{th}$"}
    commodities = {"electricity", "heat"}
    ndays = 30
    nhours = 24 * ndays

    # Electricity Demand
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
        [
            [(u + 0.1 * np.random.rand()) * 40, (u + 0.1 * np.random.rand()) * 60]
            for day in range(ndays)
            for u in dailyProfileSimple
        ],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)
    heat_demand = pd.DataFrame(
        [
            [(u + 0.1 * np.random.rand()) * 10, (u + 0.1 * np.random.rand()) * 20]
            for day in range(ndays)
            for u in dailyProfileSimple
        ],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)
    # 1) Define balanceLimit constraint in relation to demand in regions
    balanceLimit = pd.DataFrame(columns=["Region1", "Region2"], index=["el", "heat"])
    perNetAutarky = 0.75
    perNetAutarky_h = 1
    balanceLimit.loc["el"] = (1 - perNetAutarky) * demand.sum()
    balanceLimit.loc["heat"] = (1 - perNetAutarky_h) * heat_demand.sum()

    # 2) Initialize esM with two regions
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=balanceLimit,
    )

    # 3) Components are added: 'Electricity demand', 'Heat demand', 'Electricity purchase', 'Heat purchase',
    # 'Heat pump', 'Wind turbines', 'PV', 'Batteries', 'AC cables', 'Heat pipes'

    # Define Electricity demand and added to Energysystem
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    # Define Heat demand and added to Energysystem
    esM.add(
        fn.Sink(
            esM=esM,
            name="Heat demand",
            commodity="heat",
            hasCapacityVariable=False,
            operationRateFix=heat_demand,
        )
    )

    # Define Cheap purchase 'Electricity purchase' and 'Heat purchase', which incentives to purchase,
    # but is limited because of balanceLimit
    # added to Energysystem
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity purchase",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.001,
            balanceLimitID="el",
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Heat purchase",
            commodity="heat",
            hasCapacityVariable=False,
            commodityCost=0.001,
            balanceLimitID="heat",
        )
    )
    # Define heatpump and added to Energysystem
    esM.add(
        fn.Conversion(
            esM=esM,
            name="heatpump",
            physicalUnit=r"MW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "heat": 2.5,
            },
            hasCapacityVariable=True,
            capacityMax=1e6,
            investPerCapacity=0.95,
            opexPerCapacity=0.95 * 0.01,
            interestRate=0.08,
            economicLifetime=33,
        )
    )
    # Define Wind turbines and added to Energysystem
    operationRateMax = pd.DataFrame(
        [[np.random.beta(a=2, b=7.5), np.random.beta(a=2, b=9)] for t in range(nhours)],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(6)
    capacityMax = pd.Series([400, 200], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=operationRateMax,
            capacityMax=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Define PV and added to Energysystem
    operationRateMax = pd.DataFrame(
        [[u, u] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    )
    capacityMax = pd.Series([100, 100], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 800, 800 * 0.02
    interestRate, economicLifetime = 0.08, 25
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=operationRateMax,
            capacityMax=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Define Batteries and added to Energysystem
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

    # Transmission Components
    # Define AC Cables and added to Energysystem
    capacityFix = pd.DataFrame(
        [[0, 30], [30, 0]], columns=["Region1", "Region2"], index=["Region1", "Region2"]
    )
    distances = pd.DataFrame(
        [[0, 400], [400, 0]],
        columns=["Region1", "Region2"],
        index=["Region1", "Region2"],
    )
    losses = 0.0001
    esM.add(
        fn.Transmission(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=capacityFix,
            distances=distances,
            losses=losses,
            balanceLimitID="el",
        )
    )

    # Define Heat pipes and added to Energysystem
    capacityFix = pd.DataFrame(
        [[0, 30], [30, 0]], columns=["Region1", "Region2"], index=["Region1", "Region2"]
    )
    distances = pd.DataFrame(
        [[0, 400], [400, 0]],
        columns=["Region1", "Region2"],
        index=["Region1", "Region2"],
    )
    losses = 0.0001
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Heat pipes",
            commodity="heat",
            hasCapacityVariable=True,
            capacityFix=capacityFix,
            distances=distances,
            losses=losses,
            balanceLimitID="heat",
        )
    )

    return esM, losses, distances, balanceLimit


@pytest.fixture
def perfectForesight_test_esM(scope="session"):

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand", "ForesightLand"},
        commodities={"electricity", "hydrogen"},
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=5,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    PVoperationRateMax = pd.DataFrame(
        [
            np.array(
                [
                    0.5,
                    0.25,
                ]
            ),
            np.array(
                [
                    0.25,
                    0.5,
                ]
            ),
        ],
        index=["PerfectLand", "ForesightLand"],
    ).T

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            capacityMax=4e6,
            investPerCapacity=1e3,
            opexPerCapacity=1,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=10,
        )
    )

    demand = {}
    demand[2020] = pd.DataFrame(
        [
            np.array(
                [
                    4380,
                    1e3,
                ]
            ),
            np.array(
                [
                    2190,
                    1e3,
                ]
            ),
        ],
        index=["PerfectLand", "ForesightLand"],
    ).T  # first investmentperiod
    demand[2025] = demand[2020]
    demand[2030] = pd.DataFrame(
        [
            np.array(
                [
                    2190,
                    1e3,
                ]
            ),
            np.array(
                [
                    4380,
                    1e3,
                ]
            ),
        ],
        index=["PerfectLand", "ForesightLand"],
    ).T
    demand[2035] = demand[2030]
    demand[2040] = demand[2030]

    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    return esM
