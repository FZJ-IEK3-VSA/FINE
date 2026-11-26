import fine as fn
import pandas as pd


def getModel(data):
    """Build the multi-regional energy system model used in the MGA example.

    It is identical to the one built in
    examples/03_Multi-regional_Energy_System_Workflow; see that example for a
    detailed, step-by-step walkthrough of each component.
    """
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
    CO2_reductionTarget = 1
    balanceLimit = pd.DataFrame(
        index=["CO2 limit"],
        columns=["Total", "lowerBound"],
        data=[[-366 * (1 - CO2_reductionTarget), True]],
    )

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=8760,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
        balanceLimit=balanceLimit,
    )

    ### Commodity sources
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

    esM.add(
        fn.Source(
            esM=esM,
            name="Natural gas purchase",
            commodity="methane",
            hasCapacityVariable=False,
            commodityCost=0.0331 * 1e-3,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Biogas purchase",
            commodity="biogas",
            operationRateMax=data["Biogas, operationRateMax"],
            hasCapacityVariable=False,
            commodityCost=0.05409 * 1e-3,
        )
    )

    ### Commodity conversion
    esM.add(
        fn.Conversion(
            esM=esM,
            name="CCGT plants (methane)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "methane": -1 / 0.6,
                "CO2": 201 * 1e-6 / 0.6,
            },
            hasCapacityVariable=True,
            investPerCapacity=0.65,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (biogas)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.63},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (hydrogen)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "hydrogen": -1 / 0.63},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

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

    ### Commodity storages
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

    ### Commodity transmission
    esM.add(
        fn.Transmission(
            esM=esM,
            name="AC cables",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityFix=data["AC cables, capacityFix"],
        )
    )

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

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (biogas)",
            commodity="biogas",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=True,
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

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines (hydrogen)",
            commodity="hydrogen",
            distances=data["Pipelines, distances"],
            hasCapacityVariable=True,
            hasIsBuiltBinaryVariable=True,
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

    ### Commodity sinks
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=data["Electricity demand, operationRateFix"],
        )
    )

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

    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to enviroment",
            commodity="CO2",
            hasCapacityVariable=False,
            balanceLimitID="CO2 limit",
        )
    )

    return esM
