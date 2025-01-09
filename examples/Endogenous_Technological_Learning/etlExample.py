import fine as fn
import pandas as pd
import numpy as np

hoursPerTimeStep = 2190
# Create an energy system model instance
esM = fn.EnergySystemModel(
    locations={"loc1"},
    commodities={"electricity", "hydrogen"},
    commodityUnitsDict={
        "electricity": r"kW$_{el}$",
        "hydrogen": r"kW$_{H_{2},LHV}$",
    },
    numberOfTimeSteps=4,
    hoursPerTimeStep=hoursPerTimeStep,
    costUnit="1 Euro",
    numberOfInvestmentPeriods=1,
    investmentPeriodInterval=5,
    startYear=2020,
    lengthUnit="km",
    verboseLogLevel=0,
)

esM.add(
    fn.Source(
        esM=esM,
        name="PV",
        commodity="electricity",
        hasCapacityVariable=True,
        economicLifetime=18,
        investPerCapacity=1,
    )
)

esM.add(
    fn.Source(
        esM=esM,
        name="PV_ETL",
        commodity="electricity",
        hasCapacityVariable=True,
        economicLifetime=18,
        etlParameter={
            "initCost": 1,
            "learningRate": 0.18,
            "initCapacity": 10,
            "maxCapacity": 100000,
        },
    )
)

### Industry site
demand = pd.Series(
    np.array(
        [
            7e3,
            7e3,
            7e3,
            7e3,
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

esM.add(
    fn.Conversion(
        esM=esM,
        name="Electrolyzers_ETL",
        physicalUnit=r"kW$_{el}$",
        commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
        hasCapacityVariable=True,
        interestRate=0,
        economicLifetime=10,
        etlParameter={
            "initCost": 1,
            "learningRate": 0.18,
            "initCapacity": 10000,
            "maxCapacity": 100000,
            "noSegments": 5,
        },
        stockCommissioning={
            2005: 200,
            2010: 1000,
            2015: 500,
        }
    )
)

# esM.add(
#     fn.Conversion(
#         esM=esM,
#         name="Electrolyzers",
#         physicalUnit=r"kW$_{el}$",
#         commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
#         hasCapacityVariable=True,
#         investPerCapacity=0.9,  # euro/kW
#         interestRate=0,
#         economicLifetime=10,
#     )
# )

esM.optimize(solver="gurobi")
print(esM.pyM.Obj())