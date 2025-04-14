import fine as fn
import pyomo.environ as pyomo

# Step 1: Define the Energy System Model
esM = fn.EnergySystemModel(
    locations={"A"},
    onlycommodities={"electricity", "hydrogen"},
    onlycommodityUnitsDict={"electricity": "GW", "hydrogen": "kg"},
    onlymaterials={"steel", "copper", "iron"},
    onlymaterialUnitsDict={"steel": "tons", "copper": "kg", "iron": "kg"}
)

# Step 2: Add a Energy Source Component that Requires Materials                             
esM.add(
    fn.Source(
        esM=esM, 
        name="Wind Turbines",
        commodity="electricity",
        hasCapacityVariable=True,
        materialIntensity={
            'A': {0: {'iron':1.0, 'copper': 5.2, 'steel': 3.1}, 1: {'iron':1.0, 'copper': 5.3, 'steel': 3.2}},
            'B': {0: {'iron':1.0, 'copper': 4.8, 'steel': 2.9}, 1: {'iron':1.0, 'copper': 4.9, 'steel': 3.0}}
            },  # Materials required for commissioning
        # materialRecovery={"steel": 0.8, "copper": 0.3}   # Recovery fractions at decommissioning
    )
)

# Add a Energy Storage Component that Requires Materials
esM.add(
    fn.Storage(
        esM=esM,
        name="Battery",
        commodity="electricity",
        chargeEfficiency=0.9,
        dischargeEfficiency=0.9,
        materialIntensity={
            'A': {0: {'copper': 5.2, 'steel': 3.1}, 1: {'copper': 5.3, 'steel': 3.2}},
            'B': {0: {'copper': 4.8, 'steel': 2.9}, 1: {'copper': 4.9, 'steel': 3.0}}
            },  # Materials required for commissioning
        # materialRecovery={"steel": 0.8, "copper": 0.3}   # Recovery fractions at decommissioning
    )
) 

# Step 3: Add Material Source 
esM.add(
    fn.Source(
        esM=esM, 
        name="Steel Supply",
        commodity="steel",
        hasCapacityVariable=True,
        material=True,
    )
)

source = esM.add(
    fn.Source(
        esM=esM, 
        name="Copper Supply",
        commodity="copper",
        hasCapacityVariable=True,
        material=True,
    )
)

# Step 4: Add Energy Sink Component that consumes Energy 
sink = esM.add(
    fn.Sink(
        esM=esM,
        name="Electricity demand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix=50,
        
    )
)

# Step 5: Add Material Sink that consumes Materials 
sink = esM.add(
    fn.Sink(
        esM=esM,
        name="Steel demand",
        hasCapacityVariable=False,
        commodity="steel",
        material=True,      
    )
)

# Step 6: Aggregate and Construct Optimization PSroblem
esM.aggregateTemporally(numberOfTypicalPeriods=30)
esM.optimize()