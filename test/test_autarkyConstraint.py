import FINE as fn
import pandas as pd
import numpy as np


# Test autarky constraint
# In this test the following steps are performed:
# 1. Define autarky limit dependent on the demand
# 2. Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3. Components are added: 'Electricity demand', 'Electricity purchase', 
#    'Wind turbines', 'PV', 'Batteries', 'AC cables'
# 'Electricity purchase' and 'AC cables' are included in the autarky analysis
# 4. The random autarky percentage ('perNetAutarky') is compared to the outcome of the model
#   Net Autarky = (1 - (Purchase + Ac cables_in - Ac cables_out)/Demand)
def test_autarkyConstraint():
    locations = {"Region1",
                 "Region2"
                 }
    commodityUnitDict = {'electricity': r'MW$_{el}$',
                         'heat': r'MW$_{th}$'}
    commodities = {'electricity',
                   'heat'}
    ndays = 30
    nhours = 24 * ndays

    ## Define Electricity Demand
    dailyProfileSimple = [0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8]
    demand = pd.DataFrame([[(u + 0.1 * np.random.rand()) * 40, (u + 0.1 * np.random.rand()) * 60]
                           for day in range(ndays) for u in dailyProfileSimple],
                          index=range(nhours), columns=['Region1', 'Region2']).round(2)
    heat_demand = pd.DataFrame([[(u + 0.1 * np.random.rand()) * 10, (u + 0.1 * np.random.rand()) * 20]
                                for day in range(ndays) for u in dailyProfileSimple],
                               index=range(nhours), columns=['Region1', 'Region2']).round(2)
    ## Define Autarky constraint in relation to demand in regions
    input_autarky = pd.DataFrame(columns=['Region1', 'Region2'], index=["el", "heat"])
    perNetAutarky = 0.75
    perNetAutarky_h = 1
    input_autarky.loc["el"] = (1 - perNetAutarky) * demand.sum()
    input_autarky.loc["heat"] = (1 - perNetAutarky_h) * heat_demand.sum()

    ## Initialize esM with two regions
    esM = fn.EnergySystemModel(locations=locations,
                               commodities=commodities,
                               numberOfTimeSteps=nhours,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='1e6 Euro',
                               lengthUnit='km', verboseLogLevel=2,
                               autarkyLimit=input_autarky)
    ## Add el. demand
    esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                    hasCapacityVariable=False,
                    operationRateFix=demand))
    ## Add heat demand
    esM.add(fn.Sink(esM=esM, name='Heat demand', commodity='heat',
                    hasCapacityVariable=False,
                    operationRateFix=heat_demand))

    ## Define Cheap purchase, which incentives to purchase, but is limited because of autarky
    esM.add(fn.Source(esM=esM, name='Electricity purchase',
                      commodity='electricity', hasCapacityVariable=False,
                      commodityCost=0.001, autarkyID="el"))
    esM.add(fn.Source(esM=esM, name='Heat purchase',
                      commodity='heat', hasCapacityVariable=False,
                      commodityCost=0.001, autarkyID="heat"))
    ## Heat pump
    esM.add(fn.Conversion(esM=esM, name='heatpump',
                          physicalUnit=r'MW$_{el}$',
                          commodityConversionFactors={
                              'electricity': -1,
                              'heat': 2.5,
                          },
                          hasCapacityVariable=True,
                          capacityMax=1e6,
                          investPerCapacity=0.95, opexPerCapacity=0.95 * 0.01,
                          interestRate=0.08, economicLifetime=33))
    ## Wind turbines
    operationRateMax = pd.DataFrame([[np.random.beta(a=2, b=7.5), np.random.beta(a=2, b=9)]
                                     for t in range(nhours)],
                                    index=range(nhours), columns=['Region1', 'Region2']).round(6)
    capacityMax = pd.Series([400, 200], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20
    esM.add(fn.Source(esM=esM, name='Wind turbines', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
                      opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

    ## PV
    operationRateMax = pd.DataFrame([[u, u] for day in range(ndays) for u in dailyProfileSimple],
                                    index=range(nhours), columns=['Region1', 'Region2'])
    capacityMax = pd.Series([100, 100], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 800, 800 * 0.02
    interestRate, economicLifetime = 0.08, 25
    esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
                      opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

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

    # Transmission Components
    ## AC cables
    capacityFix = pd.DataFrame([[0, 30], [30, 0]], columns=['Region1', 'Region2'],
                               index=['Region1', 'Region2'])
    distances = pd.DataFrame([[0, 400], [400, 0]], columns=['Region1', 'Region2'],
                             index=['Region1', 'Region2'])
    losses = 0.0001
    esM.add(fn.Transmission(esM=esM, name='AC cables', commodity='electricity',
                            hasCapacityVariable=True, capacityFix=capacityFix,
                            distances=distances, losses=losses, autarkyID="el"))

    ## Heat pipes
    capacityFix = pd.DataFrame([[0, 30], [30, 0]], columns=['Region1', 'Region2'],
                               index=['Region1', 'Region2'])
    distances = pd.DataFrame([[0, 400], [400, 0]], columns=['Region1', 'Region2'],
                             index=['Region1', 'Region2'])
    losses = 0.0001
    esM.add(fn.Transmission(esM=esM, name='Heat pipes', commodity='heat',
                            hasCapacityVariable=True, capacityFix=capacityFix,
                            distances=distances, losses=losses, autarkyID="heat"))

    ## Optimize model
    esM.optimize(timeSeriesAggregation=False, solver='glpk')

    for i, loc in enumerate(esM.locations):
        # Get Electricity Purchase for location
        el_purchase = \
            esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.loc[
                "Electricity purchase", loc].sum()
        heat_purchase = \
            esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.loc[
                "Heat purchase", loc].sum()
        # Get Exchange going into region and out of region
        cables = esM.componentModelingDict["TransmissionModel"].operationVariablesOptimum.loc["AC cables"]
        pipes = esM.componentModelingDict["TransmissionModel"].operationVariablesOptimum.loc["Heat pipes"]
        for j, loc_ in enumerate(esM.locations):
            if loc != loc_:
                exch_in = (cables.loc[loc_, loc] *
                           (1 - losses * distances.loc[loc_, loc])).T.sum()
                exch_in_h = (pipes.loc[loc_, loc] *
                             (1 - losses * distances.loc[loc_, loc])).T.sum()
                exch_out = cables.loc[loc, loc_].T.sum()
                exch_out_h = pipes.loc[loc, loc_].T.sum()

        tolerance = 0.001
        ## Compare modelled autarky to limit set in constraint.
        assert el_purchase + exch_in - exch_out + tolerance > input_autarky.loc["el", loc]
        assert heat_purchase + exch_in_h - exch_out_h + tolerance > input_autarky.loc["heat", loc]
