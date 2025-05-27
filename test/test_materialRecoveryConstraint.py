import fine as fn
import pandas as pd

# Test material recovery constraint in a simple perfect foresight model
# Here, material intensity of ip-1 and decommissioning of ip 0 and material recovery of ip 0 are multiplied and result in the operation rate of the material recovery source (copper/steel recovery)


def test_material_recovery_constraint():

    # Define the Energy System Model
    esM = fn.EnergySystemModel(
        locations={"A"},
        onlycommodities={"electricity"},
        onlycommodityUnitsDict={"electricity": r"GW$_{el}$"},
        onlymaterials={"steel", "copper"},
        onlymaterialUnitsDict={"steel": r"tons", "copper": r"tons"},
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        startYear=2020,
        numberOfTimeSteps=1,
        hoursPerTimeStep=1,
    )

    # Add electricity source with material intensity and recovery
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind Turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            economicLifetime=5,
            stockCommissioning = {2015: 10}, 
            materialIntensity={
                "A": {
                    "steel": pd.Series({2015:6.0, 2020: 4.0, 2025: 2.0}),
                    "copper": pd.Series({2015:5.0, 2020: 3.0, 2025: 1.0})
                }
            },
            materialRecovery={
                "A": {
                    "steel": pd.Series({2020: 0.8, 2025: 0.9}),
                    "copper": pd.Series({2020: 0.7, 2025: 0.8}),

                }
            },
        )
    )


    # Add electricity sink with fixed operation rate
    demand = {}
    demand[2020] = pd.DataFrame(index=[0], columns=["A"], data=[[15]])
    demand[2025] = pd.DataFrame(index=[0], columns=["A"], data=[[25]])

    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand
        )
    )


    # Add material sources (primary and secondary) for all materials

    esM.add(
        fn.Source(
            esM=esM,
            name="Steel Recovery",
            commodity="steel",
            hasCapacityVariable=False,
            material=True,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Steel Supply",
            commodity="steel",
            hasCapacityVariable=False,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Copper Recovery",
            commodity="copper",
            hasCapacityVariable=False,
            material=True,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Copper Supply",
            commodity="copper",
            hasCapacityVariable=False,
        )
    )

    # Generate material sinks for all materials
    esM.generationMaterialSinks()

    esM.optimize(solver="glpk")


    # Expected values for operation rate of the material recovery sources = material intensity(ip-1) * recovery rate(ip) * decommissioning(ip) 
    expected_recovery_2020_steel = 6.0* 0.8* 10
    expected_recovery_2025_steel = 4.0* 0.9* 12  

    res_recovery_2020_steel = esM.pyM.op_srcSnk["A", "Steel Recovery", 0, 0, 0]()
    res_recovery_2025_steel = esM.pyM.op_srcSnk["A", "Steel Recovery", 1, 0, 0]()


    expected_recovery_2020_copper = 5.0* 0.7* 10 
    expected_recovery_2025_copper = 3.0* 0.8* 12  

 
    res_recovery_2020_copper = esM.pyM.op_srcSnk["A", "Copper Recovery", 0, 0, 0]()
    res_recovery_2025_copper = esM.pyM.op_srcSnk["A", "Copper Recovery", 1, 0, 0]()


    # Check whether the constraint performs the calculation correctly 
    assert res_recovery_2020_steel <= expected_recovery_2020_steel
    assert res_recovery_2025_steel <= expected_recovery_2025_steel
    assert res_recovery_2020_copper <= expected_recovery_2020_copper
    assert res_recovery_2025_copper <= expected_recovery_2025_copper

