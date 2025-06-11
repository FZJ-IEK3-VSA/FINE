import fine as fn
import random

def calculateBeta(esM):

    components = []
    transmissionComponents = []

    for item in esM.componentModelingDict.values():
        for key,_item in item.componentsDict.items():
            components.append(key)

            if isinstance(_item, fn.transmission.Transmission):
                transmissionComponents.append(key)

    transmission_locations = []
    for loc1 in esM.locations:
        for loc2 in esM.locations:
            if loc1 != loc2:
                transmission_locations.append(loc1 + "_" + loc2)

    esM.beta = {location:
            {iteration+1:
            {component: random.random() for component in components
             if component not in transmissionComponents
            }
            for iteration in range(esM.iterations)
            }
            for location in esM.locations
            }

    transmission_data = {location:
            {iteration+1:
            {component: random.random() for component in transmissionComponents
            }
            for iteration in range(esM.iterations)
            }
            for location in transmission_locations
            }

    esM.beta.update(transmission_data)

esM = fn.EnergySystemModel(
    locations={"Region1","Region2"},
    commodities={"electricity","heat"},
    numberOfTimeSteps=6,
    commodityUnitsDict={"electricity": r"GW$_{el}$", "heat": r"GW$_{heat}$"},
    hoursPerTimeStep=1,
    costUnit="1e9 Euro",
    lengthUnit="km",
    verboseLogLevel=0
)

esM.add(
    fn.Sink(
        esM=esM,
        name="Electricity demand",
        commodity="electricity",
        hasCapacityVariable=False,
    )
)

esM.add(
    fn.Source(
        esM=esM,
        name="wind",
        commodity="electricity",
        hasCapacityVariable=True,
    )
)

esM.add(
    fn.Conversion(
        esM=esM,
        name="Heat Pump",
        physicalUnit=r"GW$_{el}$",
        commodityConversionFactors={"electricity": -1/2.5, "heat": 1},
        hasCapacityVariable=True,
    )
)

esM.add(
    fn.Storage(
        esM=esM,
        name="Li-ion batteries",
        commodity="electricity",
        hasCapacityVariable=True,
    )
)

esM.add(
    fn.Transmission(
        esM=esM,
        name="AC cables",
        commodity="electricity",
        hasCapacityVariable=True,
    )
)

esM.iterations = 2
calculateBeta(esM)
print(esM.beta)
