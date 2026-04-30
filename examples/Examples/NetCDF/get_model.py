import fine as fn
import pandas as pd
import numpy as np
np.random.seed(42)  # Sets a "seed" to produce the same random input data in each model run

esM = fn.EnergySystemModel(
    locations = {"regionN", "regionS"},
    commodities = {"electricity", "naturalGas", "CO2"},
    commodityUnitsDict = {
    "electricity": r"GW$_{el}$",
    "naturalGas": r"GW$_{CH_{4},LHV}$",
    "CO2": r"Mio. t$_{CO_2}$/h",
    }
    )

esM.add(
    fn.Source(
        esM = esM,
        name = "Wind turbines", # Name of the source
        commodity = "electricity", # Name of the commodity produced by the source, as set during initialization
        hasCapacityVariable = True, # Specifies whether the source has a capacity variable
        operationRateMax = pd.DataFrame(
            [[np.random.beta(a=2, b=7.5), np.random.beta(a=2, b=9)] for t in range(8760)],
            index=range(8760),
            columns=["regionN", "regionS"]
            ).round(6), # Defines the maximum operation rate for each time step and location. Was set here randomly.
        capacityMax = pd.Series([400, 200], index=["regionN", "regionS"]), # Indicates the maximum capacity of this source for each location
        investPerCapacity = 1200, # Describes the investment costs for one unit of the capacity
        opexPerCapacity = 24, # Describes the operational cost for one unit of capacity
        interestRate = 0.08, # Describes the interest rate which is considered for computing the annuities of the invest of the component (depreciates the invests over the economic lifetime)
        economicLifetime = 20, # Describes the economic lifetime of the component which is considered for computing the annuities of the invest of the component (i.e. the depreciation time)
        )
    )

dailyProfile = [
    0.6, # 12am - 1am
    0.6, # 1am - 2am
    0.6, # ...
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
    0.8 # 11pm - 12 am
    ]

electricityDemand = pd.DataFrame(
    [
        [(u + 0.1 * np.random.rand()) * 25, (u + 0.1 * np.random.rand()) * 40]
        for day in range(365)
        for u in dailyProfile
    ],
    index=range(8760), # Timesteps of the esM
    columns=["regionN", "regionS"],
).round(2)

esM.add(
    fn.Sink(
        esM = esM,
        name = "Electricity demand",
        commodity = "electricity",
        hasCapacityVariable = False,
        operationRateFix = electricityDemand
    )
)

esM.add(
    fn.Transmission(
        esM = esM,
        name = "AC cables", # Name of the transmission
        commodity = "electricity", # Name of the commodity produced by the source, as set during initialization
        hasCapacityVariable = True, # Specifies whether the source has a capacity variable
        capacityFix = pd.DataFrame(
            [[0, 30], [30, 0]], columns=["regionN", "regionS"], index=["regionN", "regionS"]
            ), # Indicates the fixed capacity of this transmission for each location
        distances = pd.DataFrame(
            [[0, 400], [400, 0]], columns=["regionN", "regionS"], index=["regionN", "regionS"]
            ), # Indicates the distance between the locations for each location
        losses = 0.0001, # Indicates the losses per unit of distance for the transmission
        )
    )

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