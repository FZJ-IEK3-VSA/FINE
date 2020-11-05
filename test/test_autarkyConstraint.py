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
    commodityUnitDict = {'electricity': r'MW$_{el}$'}
    commodities = {'electricity'}

    ## Define Electricity Demand
    dailyProfileSimple = [0.6,0.6,0.6,0.6,0.6,0.7,0.9,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0.9,0.8]
    demand = pd.DataFrame([[(u+0.1*np.random.rand())*40,(u+0.1*np.random.rand())*60]
                            for day in range(365) for u in dailyProfileSimple],
                            index=range(8760), columns=['Region1', 'Region2']).round(2)

    ## Define Autarky constraint in relation to demand in regions
    perNetAutarky = round(np.random.random(), 2)
    _autarkyConstraint = (1-perNetAutarky)*demand.sum()
    autarkyConstraint = {}
    for i, val in _autarkyConstraint.iteritems():
        autarkyConstraint[i] = val
    autarkyLimit = {"el": autarkyConstraint}

    ## Initialize esM with two regions
    esM = fn.EnergySystemModel(locations=locations,
                               commodities=commodities,
                               numberOfTimeSteps=8760,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='1e6 Euro',
                               lengthUnit='km', verboseLogLevel=2,
                               autarkyLimit=autarkyLimit)
    ## Add el. demand
    esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                    hasCapacityVariable=False,
                    operationRateFix=demand))


    ## Define Cheap purchase, which incentives to purchase, but is limited because of autarky
    purchaseRate = pd.DataFrame([[1000, 1000] for t in range(8760)],
                                index=range(8760), columns=["Region1", "Region2"])
    esM.add(fn.Source(esM=esM, name='Electricity purchase',
                      commodity='electricity', hasCapacityVariable=False,
                      operationRateMax=purchaseRate,
                      commodityCost=0.001, autarkyID="el"))

    ## Wind turbines
    operationRateMax = pd.DataFrame([[np.random.beta(a=2,b=7.5),np.random.beta(a=2,b=9)]
                                    for t in range(8760)],
                                    index=range(8760), columns=['Region1', 'Region2']).round(6)
    capacityMax = pd.Series([400, 200], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 1200, 1200*0.02
    interestRate, economicLifetime = 0.08, 20
    esM.add(fn.Source(esM=esM, name='Wind turbines', commodity='electricity', hasCapacityVariable=True,
        operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
        opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

    ## PV
    operationRateMax = pd.DataFrame([[u,u] for day in range(365) for u in dailyProfileSimple],
                                    index=range(8760), columns=['Region1', 'Region2'])
    capacityMax = pd.Series([100, 100], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 800, 800*0.02
    interestRate, economicLifetime = 0.08, 25
    esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
        operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
        opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime))

    ## Batteries
    chargeEfficiency, dischargeEfficiency, selfDischarge = 0.95, 0.95, 1-(1-0.03)**(1/(30*24))
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 150, 150*0.01
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
    losses=0.0001

    esM.add(fn.Transmission(esM=esM, name='AC cables', commodity='electricity',
        hasCapacityVariable=True, capacityFix=capacityFix,
        distances=distances, losses=losses, autarkyID="el"))
    
    ## Optimize model
    esM.optimize(timeSeriesAggregation=False, solver='glpk')
    res_source = esM.getOptimizationSummary("SourceSinkModel", outputLevel=2)
    res_transmission = esM.getOptimizationSummary("TransmissionModel", outputLevel=2)
    for i, loc in enumerate(esM.locations):
        # Get Electricity Purchase for location
        if ("Electricity purchase", "operation") in res_source[loc].index:
            el_purchase = \
                res_source.loc["Electricity purchase", "operation"][loc].values[0]
        else:
            el_purchase = 0
        # Get Electricity demand for location
        el_demand = res_source.loc["Electricity demand"][loc].values[0]
        # Get Exchange going into region and out of region
        cables = res_transmission.loc["AC cables"]
        for j, loc_ in enumerate(esM.locations):
            if loc != loc_:
                try:
                    exch_in = cables.loc["operation", "[MW$_{el}$*h/a]", loc_][loc] * \
                        (1-losses * distances.loc[loc_, loc])
                except:
                    exch_in = 0
                try:
                    exch_out = cables.loc["operation", "[MW$_{el}$*h/a]", loc][loc_]
                except:
                    exch_out = 0
        if np.isnan(exch_out):
            exch_out = 0
        if np.isnan(exch_in):
            exch_in = 0
        netAutarky = (1 - (el_purchase + exch_in - exch_out) / el_demand)
        tolerance = 0.0001
        ## Compare modelled autarky to limit set in constraint.
        assert netAutarky > (perNetAutarky - tolerance)
