import fine as fn
from getData import getData
import pandas as pd
import numpy as np

data = getData()

locations = {
    "cluster_1",
    "cluster_2"
}
commodityUnitDict = {
    "electricity": r"GW$_{el}$"
}
commodities = {"electricity"}
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

# Source in cluster_1

esM.add(
    fn.Source(
        esM=esM,
        name="Electricity grid",
        commodity="electricity",
        hasCapacityVariable=False,
        commodityCost=0.1
    )
)

# Sink in cluster_2

demand_series = data["Electricity demand, operationRateFix"]

demand = pd.DataFrame(index=demand_series.index)
demand["cluster_1"] = 0.0
demand["cluster_2"] = demand_series.values

esM.add(
    fn.Sink(
        esM=esM,
        name="Electricity demand",
        commodity="electricity",
        hasCapacityVariable=False,
        operationRateFix=demand,
    )
)

# Transmission
loc_list = ["cluster_1", "cluster_2"]

distances = np.array([[0, 100], [100, 0]])
distances = pd.DataFrame(distances, index=loc_list, columns=loc_list)

reactances = pd.DataFrame(
    [[0, 0.1], [0.1, 0]],
    index=loc_list,
    columns=loc_list
)

esM.add(
    fn.LinearOptimalPowerFlow(
        esM=esM,
        name="DC cables",
        commodity="electricity",
        distances=distances,
        hasCapacityVariable=True,
        reactances=reactances,
    )
)

esM.optimize(
    optimizationSpecs="OptimalityTol=1e-3 method=2 cuts=0 MIPGap=5e-3",
)

# Output
print(esM.componentModelingDict.keys())
# lopf_model = esM.componentModelingDict["lopf"]
# flows = lopf_model.getOptimalValues("operationVariablesOptimum")
# angles = lopf_model.getOptimalValues("phaseAngleVariablesOptimum")
# print(flows.keys())
# print(angles.keys())
