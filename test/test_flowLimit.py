import FINE as fn
import pandas as pd
import numpy as np

# Test autarky constraint
# In this test the following steps are performed:
# 1. Define autarky limit dependent on the electricity and heat demand
# 2. Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3. Components are added: 'Electricity demand', 'Heat demand', 'Electricity purchase', 'Heat purchase', 'Heat pump',
#    'Wind turbines', 'PV', 'Batteries', 'AC cables', 'Heat pipes'
# 'Electricity purchase' and 'AC cables' are included in the autarky analysis
# 4. The autarky limit is compared to the outcome of the model
#   purchase + exchange_in - exchange_out <= autarkyLimit
def test_flowLimitConstraint():
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
                               flowLimit=input_autarky)

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
                      commodityCost=0.001, flowLimitID="el"))
    esM.add(fn.Source(esM=esM, name='Heat purchase',
                      commodity='heat', hasCapacityVariable=False,
                      commodityCost=0.001, flowLimitID="heat"))
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
                            distances=distances, losses=losses, flowLimitID="el"))

    ## Heat pipes
    capacityFix = pd.DataFrame([[0, 30], [30, 0]], columns=['Region1', 'Region2'],
                               index=['Region1', 'Region2'])
    distances = pd.DataFrame([[0, 400], [400, 0]], columns=['Region1', 'Region2'],
                             index=['Region1', 'Region2'])
    losses = 0.0001
    esM.add(fn.Transmission(esM=esM, name='Heat pipes', commodity='heat',
                            hasCapacityVariable=True, capacityFix=capacityFix,
                            distances=distances, losses=losses, flowLimitID="heat"))

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
        assert el_purchase + exch_in - exch_out - tolerance < input_autarky.loc["el", loc]
        assert heat_purchase + exch_in_h - exch_out_h - tolerance < input_autarky.loc["heat", loc]


# Test loweLimit constraint
# In this test the following steps are performed:
# 1. Define lower limit for renewable energy sources dependent on the demand
# 2. Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3. Components are added: 'Electricity demand', 'Electricity purchase',
#    'Wind turbines', 'PV', 'Batteries'
# 'Electricity purchase' and 'AC cables' are included in the autarky analysis
# 4. The lower limit is compared to the outcome of the model
#   operation wind + operation pv >= lowerLimit
def test_electricitySourceDriver():
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
                               flowLimit=lowerLimit,
                               lowerBound=True)
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
                      flowLimitID="Renewables"))

    ## PV
    operationRateMax = pd.DataFrame([[u, u] for day in range(ndays) for u in dailyProfileSimple],
                                    index=range(nhours), columns=['Region1', 'Region2'])
    capacityMax = pd.Series([100, 100], index=['Region1', 'Region2'])
    investPerCapacity, opexPerCapacity = 800, 800 * 0.02
    interestRate, economicLifetime = 0.08, 25
    esM.add(fn.Source(esM=esM, name='PV', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=operationRateMax, capacityMax=capacityMax, investPerCapacity=investPerCapacity,
                      opexPerCapacity=opexPerCapacity, interestRate=interestRate, economicLifetime=economicLifetime,
                      flowLimitID="Renewables"))

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

# Test Hydrogen constraint
# In this test the following steps are performed:
# 1. Define lower limit for renewable energy sources dependent on the demand
# 2. Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3. Components are added: 'Electricity demand', 'Electricity purchase',
#    'Wind turbines', 'PV', 'Batteries'
# 'Electricity purchase' and 'AC cables' are included in the autarky analysis
# 4. The lower limit is compared to the outcome of the model
#   operation wind + operation pv >= lowerLimit
def test_hydrogenSinkDriver():
    locations = {"Region1"
                 }
    commodityUnitDict = {'electricity': r'MW$_{el}$',
                         'hydrogen': r'MW$_{LHV_H2}$'}
    commodities = {'electricity',
                   'hydrogen'}
    ndays = 20
    nhours = 24 * ndays

    ## Wind turbines
    dailyProfile = [0.21773151164616183, 0.034941022753796035, 0.06456056257136962, 0.18781756363388397,
                    0.07389116186240981, 0.1696331916932108, 0.4464499926416005, 0.1706764273756967,
                    0.15694190971549513, 0.2882743403035651, 0.32127549679527717, 0.14116461052786136,
                    0.5461613758054059, 0.21958149674207716, 0.25715084860896853, 0.21525778612323568,
                    0.15916222011475448, 0.11014921269063864, 0.2532504131880449, 0.3154483508270503,
                    0.08412368254727028, 0.06337996942065917, 0.12431489082721527, 0.17319120880651773]

    operationRateMax = pd.DataFrame([u
                                     for day in range(ndays) for u in dailyProfile],
                                    index=range(nhours), columns=['Region1']).round(6)
    capacityMaxWind = pd.Series([400], index=['Region1'])
    # Define min production
    minProduction = operationRateMax['Region1'].sum()*capacityMaxWind.loc["Region1"]*0.6
    print(minProduction)
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20
    flowLimit = pd.DataFrame(index=["hydrogenDriver"], columns=["Region1"])
    # Define negative flowLimit as driver. Because source is contributing negatively.
    flowLimit.loc["hydrogenDriver", "Region1"] = -1*minProduction

    ## Initialize esM with one region
    # Use upperBound in a mathematical sense. abs(Sink Operation) >= abs(flowLimit), both negative because Sink.
    esM = fn.EnergySystemModel(locations=locations,
                               commodities=commodities,
                               numberOfTimeSteps=nhours,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='1e6 Euro',
                               lengthUnit='km', verboseLogLevel=2,
                               flowLimit=flowLimit,
                               lowerBound=False)
    # Define hydrogen sink
    esM.add(fn.Sink(esM=esM, name="Hydrogen Annual Production", flowLimitID="hydrogenDriver",
                    commodity="hydrogen", hasCapacityVariable=False))

    esM.add(fn.Source(esM=esM, name='Wind turbines', commodity='electricity', hasCapacityVariable=True,
                      operationRateMax=operationRateMax, capacityMax=capacityMaxWind, investPerCapacity=investPerCapacity,
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

    # techno-economic parameters electrolysis
    invest_per_capacity_electrolysis = 0.5  # unit 1e9 EUR/GW
    opex_electrolysis = 0.025  # in %/100 of capex
    interestRate_electrolysis = 0.08  # in %/100
    economicLifetime_electrolysis = 10  # in years
    esM.add(fn.Conversion(esM=esM,
                          name='Electrolyzers',
                          physicalUnit=r'MW$_{el}$',
                          commodityConversionFactors={'electricity':-1, 'hydrogen':0.7},
                          hasCapacityVariable=True,
                          investPerCapacity=invest_per_capacity_electrolysis,
                          opexPerCapacity=invest_per_capacity_electrolysis * opex_electrolysis,
                          interestRate=interestRate_electrolysis,
                          economicLifetime=economicLifetime_electrolysis,
                          )
            )
    ## Optimize model
    esM.optimize(timeSeriesAggregation=False, solver='glpk')

    for i, loc in enumerate(esM.locations):
        # Get operation of Renewables for loc
        operation_hydrogen = \
            esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.loc[
                "Hydrogen Annual Production", loc].sum()
        tolerance = 0.001
        ## Compare modelled lowerLimit to limit set in constraint.
        test_min_production = ndays*sum(dailyProfile)*capacityMaxWind.loc["Region1"]*0.6
        assert operation_hydrogen > test_min_production * (1 - tolerance)
        assert operation_hydrogen < test_min_production * (1 + tolerance)

def test_CO2Limit():
    locations = {"Region1",
                 "Region2"
                 }
    commodityUnitDict = {'electricity': r'MW$_{el}$',
                         'methane': r'MW$_{CH_{4},LHV}$',
                         'CO2': r'Mio. kg$_{CO_2}$/h'}
    commodities = {'electricity',
                   'methane',
                   'CO2'}
    ndays = 30
    nhours = 24 * ndays

    ## Define Electricity Demand
    dailyProfileSimple = [0.6, 0.6, 0.6, 0.6, 0.6, 0.7, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.8]
    demand = pd.DataFrame([[(u + 0.1 * np.random.rand()) * 40, (u + 0.1 * np.random.rand()) * 60]
                           for day in range(ndays) for u in dailyProfileSimple],
                          index=range(nhours), columns=['Region1', 'Region2']).round(2)

    ## Define Autarky constraint in relation to demand in regions
    CO2_limit = pd.Series(index=["CO2 limit"])
    CO2_limit.loc["CO2 limit"] = -1*demand.sum().sum()*0.6*201 * 1e-6 / 0.6

    ## Initialize esM with two regions
    esM = fn.EnergySystemModel(locations=locations,
                               commodities=commodities,
                               numberOfTimeSteps=nhours,
                               commodityUnitsDict=commodityUnitDict,
                               hoursPerTimeStep=1, costUnit='1e6 Euro',
                               lengthUnit='km', verboseLogLevel=2,
                               flowLimit=CO2_limit,
                               lowerBound=True)
    ## Add el. demand
    esM.add(fn.Sink(esM=esM, name='Electricity demand', commodity='electricity',
                    hasCapacityVariable=False,
                    operationRateFix=demand))
    esM.add(fn.Source(esM=esM, name='Methane purchase', commodity='methane',
                    hasCapacityVariable=False))
    esM.add(fn.Conversion(esM=esM, name='ccgt',
                          physicalUnit=r'MW$_{el}$',
                          commodityConversionFactors={'electricity': 1,
                                                      'methane': -1 / 0.6,
                                                      'CO2': 201 * 1e-6 / 0.6},
                          hasCapacityVariable=True,
                          investPerCapacity=0.65, opexPerCapacity=0.021, interestRate=0.08,
                          economicLifetime=33))
    esM.add(fn.Sink(esM=esM, name='CO2 to environment', commodity='CO2',
                    hasCapacityVariable=False, flowLimitID="CO2 limit"))
    ## Wind turbines
    opexPerOperation = 150/1e6
    interestRate, economicLifetime = 0.08, 20
    esM.add(fn.Source(esM=esM, name='Fuel Cell', commodity='electricity', hasCapacityVariable=False,
                      opexPerOperation=opexPerOperation, interestRate=interestRate, economicLifetime=economicLifetime,
                      ))

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
    co2_to_environment = 0
    for i, loc in enumerate(esM.locations):
        # Get operation of Renewables for loc
        co2_to_environment += \
            esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.loc[
                "CO2 to environment", loc].sum()
    tolerance = 0.001
    ## Compare modelled lowerLimit to limit set in constraint.
    assert co2_to_environment * (1-tolerance) < -1*CO2_limit.loc["CO2 limit"]
    assert co2_to_environment * (1+tolerance) > -1*CO2_limit.loc["CO2 limit"]
