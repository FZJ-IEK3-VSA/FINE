import os
import sys

import pandas as pd
import xarray as xr

import FINE as fn

cwd = os.getcwd()
sys.path.append(os.path.join(cwd, "..", "Multi-regional_Energy_System_Workflow"))
from getData import getData

data = getData()


def getModel():

    # 1. Create an energy system model instance
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
        "electricity": "GW$_{el}$",
        "methane": "GW$_{CH_{4},LHV}$",
        "biogas": "GW$_{biogas,LHV}$",
        "CO2": "Mio. t$_{CO_2}$/h",
        "hydrogen": "GW$_{H_{2},LHV}$",
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

    CO2_reductionTarget = 1

    # 2. Add commodity sources to the energy system model

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

    # 3. Add conversion components to the energy system model

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

    # 4. Add commodity storages to the energy system model

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

    # 5. Add commodity transmission components to the energy system model

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

    # 6. Add commodity sinks to the energy system model

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
