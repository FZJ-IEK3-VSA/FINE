import FINE as fn
import pandas as pd
import numpy as np


# Test autarky constraint
# In this test the following steps are performed:
# 1. Define lower limit for renewable energy sources dependent on the demand
# 2. Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3. Components are added: 'Electricity demand', 'Electricity purchase',
#    'Wind turbines', 'PV', 'Batteries'
# 'Electricity purchase' and 'AC cables' are included in the autarky analysis
# 4. The lower limit is compared to the outcome of the model
#   operation wind + operation pv >= lowerLimit
def test_lowerLimit():
    locations = {"Region1",
                 "Region2"
                 }
    commodityUnitDict = {'electricity': r'MW$_{el}$'}
    commodities = {'electricity'}
    ndays = 30
    nhours = 24 * ndays

    ## Define Electricity Demand
    dailyProfileSimple = [0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8]
    demand = pd.DataFrame([[(u + 0.1 * np.random.rand()) * 40, (u + 0.1 * np.random.rand()) * 60]
                           for day in range(ndays) for u in dailyProfileSimple],
                          index=range(nhours), columns=['Region1', 'Region2']).round(2)
    ## Define Autarky constraint in relation to demand in regions
    lowerLimit = pd.DataFrame(columns=['Region1', 'Region2'], index=["Renewables"])
    perLowerLimit = 0.75
    lowerLimit.loc["Renewables"] = (1 - perLowerLimit) * demand.sum()

    ## Initialize esM with two regions
    esM = fn.EnergySystemModel(locations=locations,
                               commodities=commodities,
                               numberOfTimeSteps=nhours,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='1e6 Euro',
                               lengthUnit='km', verboseLogLevel=2,
                               lowerLimit=lowerLimit)
    ## Add el. demand
    esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                    hasCapacityVariable=False,
                    operationRateFix=demand))

    ## Define Cheap purchase, which incentives to purchase
    esM.add(fn.Source(esM=esM, name='Electricity purchase',
                      commodity='electricity', hasCapacityVariable=False,
                      commodityCost=0.001))

    ## Wind turbines
    operationRateMax = pd.DataFrame([[np.random.beta(a=2, b=7.5), np.random.beta(a=2, b=9)]
                                     for t in range(nhours)],
                                    index=range(nhours), columns=['Region1', 'Region2']).round(6)
    capacityMax = pd.Series([400, 200], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20
    esM.add(fn.Source(esM=esM, name='Wind turbines', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
                      opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime,
                      lowerLimitID="Renewables"))

    ## PV
    operationRateMax = pd.DataFrame([[u, u] for day in range(ndays) for u in dailyProfileSimple],
                                    index=range(nhours), columns=['Region1', 'Region2'])
    capacityMax = pd.Series([100, 100], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 800, 800 * 0.02
    interestRate, economicLifetime = 0.08, 25
    esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
                      opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime,
                      lowerLimitID="Renewables"))

    ## Batteries
    chargeEfficiency, dischargeEfficiency, selfDischarge = 0.95, 0.95, 1 - (1 - 0.03) ** (1 / (30 * 24))
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 150, 150 * 0.01
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000

    esM.add(fn.Storage(esM=esM, name='Batteries', commodity='electricity', hasCapacityVariable=True,
                       chargeEfficiency=chargeEfficiency, cyclicLifetime=cyclicLifetime,
                       dischargeEfficiency=dischargeEfficiency, selfDischarge=selfDischarge, chargeRate=chargeRate,
                       dischargeRate=dischargeRate, investPerCapacity=investPerCapacity,
                       opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

    ## Optimize model
    esM.optimize(timeSeriesAggregation=False, solver='glpk')

    for i, loc in enumerate(esM.locations):
        # Get operation of Renewables for loc
        operation_wind = \
            esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.loc[
                "Wind turbines", loc].sum()
        operation_pv = \
            esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.loc[
                "PV", loc].sum()
        tolerance = 0.001
        ## Compare modelled lowerLimit to limit set in constraint.
        assert operation_wind + operation_pv + tolerance > lowerLimit.loc["Renewables", loc]
