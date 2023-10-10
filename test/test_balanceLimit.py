import FINE as fn
import pandas as pd
import numpy as np

# Test balanceLimit constraint for autarky analysis
# In this test the following steps are performed:
# 0) Preprocess energy system model
# 1) Define balanceLimit dependent on the electricity and heat demand
# 2) Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3) Components are added: 'Electricity demand', 'Heat demand', 'Electricity purchase', 'Heat purchase', 'Heat pump',
#   'Wind turbines', 'PV', 'Batteries' as well as transmission components 'AC cables', 'Heat pipes'
#   'Electricity purchase' and 'AC cables' are included in the autarky analysis
# 4) Optimize model
# 5) The balanceLimit is compared to the outcome of the model
#   purchase + exchange_in - exchange_out <= balanceLimit


def test_balanceLimitConstraint(balanceLimitConstraint_test_esM):
    def check_selfSufficiency(esM, losses, distances, balanceLimit):
        for i, loc in enumerate(esM.locations):
            # Get Electricity Purchase for location
            el_purchase = (
                esM.componentModelingDict["SourceSinkModel"]
                .operationVariablesOptimum.loc["Electricity purchase", loc]
                .sum()
            )
            heat_purchase = (
                esM.componentModelingDict["SourceSinkModel"]
                .operationVariablesOptimum.loc["Heat purchase", loc]
                .sum()
            )
            # Get Exchange going into region and out of region
            cables = esM.componentModelingDict[
                "TransmissionModel"
            ].operationVariablesOptimum.loc["AC cables"]

            pipes = esM.componentModelingDict[
                "TransmissionModel"
            ].operationVariablesOptimum.loc["Heat pipes"]

            for j, loc_ in enumerate(esM.locations):
                if loc != loc_:
                    exch_in = (
                        cables.loc[loc_, loc] * (1 - losses * distances.loc[loc_, loc])
                    ).T.sum()
                    exch_in_h = (
                        pipes.loc[loc_, loc] * (1 - losses * distances.loc[loc_, loc])
                    ).T.sum()
                    exch_out = cables.loc[loc, loc_].T.sum()
                    exch_out_h = pipes.loc[loc, loc_].T.sum()

            tolerance = 0.001

            ## Compare modelled autarky to limit set in constraint.
            assert np.isclose(
                (el_purchase + exch_in - exch_out), balanceLimit.loc["el", loc]
            )
            assert np.isclose(
                (heat_purchase + exch_in_h - exch_out_h), balanceLimit.loc["heat", loc]
            )

    # Test without segmentation:
    esM, losses, distances, balanceLimit = balanceLimitConstraint_test_esM
    # 1) Optimize model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    # 2) The balanceLimit is compared to the outcome of the model
    #   purchase + exchange_in - exchange_out <= balanceLimit
    check_selfSufficiency(esM, losses, distances, balanceLimit)

    # Test self sufficiency with segmenation
    esM_segmentation, losses, distances, balanceLimit = balanceLimitConstraint_test_esM
    esM_segmentation.aggregateTemporally(
        numberOfTypicalPeriods=10,
        numberOfTimeStepsPerPeriod=24,
        storeTSAinstance=False,
        segmentation=True,
        numberOfSegmentsPerPeriod=4,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        sortValues=False,
        rescaleClusterPeriods=False,
    )

    # 1) Optimize model
    esM_segmentation.optimize(timeSeriesAggregation=True, solver="glpk")
    # 2) The balanceLimit is compared to the outcome of the model
    #   purchase + exchange_in - exchange_out <= balanceLimit
    check_selfSufficiency(esM_segmentation, losses, distances, balanceLimit)


# In this test the following steps are performed:
# 0) Preprocess energy system model
# 1) Define lower limit for renewable energy sources dependent on the demand
# 2) Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3) Components are added: 'Electricity demand', 'Electricity purchase',
#    'Wind turbines', 'PV', 'Batteries'
# 'Electricity purchase' and 'AC cables' are included in the lower limit analysis
# 4) Optimize model
# 5) The balanceLimit is compared to the outcome of the model
#   operation wind + operation pv >= balanceLimit


def test_electricitySourceDriver():
    # 0) Preprocess energy system model
    locations = {"Region1", "Region2"}
    commodityUnitDict = {"electricity": r"MW$_{el}$"}
    commodities = {"electricity"}
    ndays = 30
    nhours = 24 * ndays

    ## Define Electricity Demand
    dailyProfileSimple = [
        0.6,
        0.6,
        0.6,
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
        0.8,
    ]
    demand = pd.DataFrame(
        [
            [(u + 0.1 * np.random.rand()) * 40, (u + 0.1 * np.random.rand()) * 60]
            for day in range(ndays)
            for u in dailyProfileSimple
        ],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)
    ## Define balanceLimit constraint in relation to demand in two regions
    balanceLimit = pd.DataFrame(columns=["Region1", "Region2"], index=["Renewables"])
    balanceLimit.loc["Renewables"] = 0.25 * demand.sum()
    balanceLimit["lowerBound"] = True

    # 2) Initialize esM with two regions
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=balanceLimit,
    )

    # 3) Components are added: 'Electricity demand', 'Electricity purchase', 'Wind turbines', 'PV', 'Batteries'
    # Define Electricity demand and added to Energysystem
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    ## Define Cheap purchase, which incentives to purchase, and added to Energysystem
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity purchase",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.001,
        )
    )

    # Define Wind turbines and added to Energysystem
    operationRateMax = pd.DataFrame(
        [[np.random.beta(a=2, b=7.5), np.random.beta(a=2, b=9)] for t in range(nhours)],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(6)
    capacityMax = pd.Series([400, 200], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=operationRateMax,
            capacityMax=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            balanceLimitID="Renewables",
        )
    )

    # Define PV and added to Energysystem
    operationRateMax = pd.DataFrame(
        [[u, u] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    )
    capacityMax = pd.Series([100, 100], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 800, 800 * 0.02
    interestRate, economicLifetime = 0.08, 25
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=operationRateMax,
            capacityMax=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
            balanceLimitID="Renewables",
        )
    )

    # Define Batteries and added to Energysystem
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 150, 150 * 0.01
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000

    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=chargeEfficiency,
            cyclicLifetime=cyclicLifetime,
            dischargeEfficiency=dischargeEfficiency,
            selfDischarge=selfDischarge,
            chargeRate=chargeRate,
            dischargeRate=dischargeRate,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # 4) Optimize model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # 5) The balanceLimit is compared to the outcome of the model
    #   operation wind + operation pv >= balanceLimit
    for i, loc in enumerate(esM.locations):
        # Get operation of Renewables for loc
        operation_wind = (
            esM.componentModelingDict["SourceSinkModel"]
            .operationVariablesOptimum.loc["Wind turbines", loc]
            .sum()
        )
        operation_pv = (
            esM.componentModelingDict["SourceSinkModel"]
            .operationVariablesOptimum.loc["PV", loc]
            .sum()
        )
        tolerance = 0.001
        ## Compare modelled lowerLimit to limit set in constraint.
        assert (
            operation_wind + operation_pv + tolerance
            > balanceLimit.loc["Renewables", loc]
        )


# Test Hydrogen (hydrogenSinkDriver) constraint
# In this test the following steps are performed:
# 0) Preprocess energy system model
# 1) Define balanceLimit (=lower limit) for hydrogen production dependent on the demand
# 2) Initialize EnergySystemModel with one Region ('Region1') with use of parameters balanceLimit and lowerBound
# 3) Components are added: 'Wind turbines', 'Electrolyzer', 'Batteries'
#    'Hydrogen Annual Production' is included in the balanceLimit analysis
# 4) Optimize Model
# 5) The balanceLimit is compared to the outcome of the model
#   Hydrogen Annual Production >= hydrogenDriver (as balanceLimitID)
def test_hydrogenSinkDriver():
    # 0) Preprocess energy system model
    locations = {"Region1"}
    commodityUnitDict = {"electricity": r"MW$_{el}$", "hydrogen": r"MW$_{LHV_H2}$"}
    commodities = {"electricity", "hydrogen"}
    ndays = 20
    nhours = 24 * ndays

    # Wind turbines
    dailyProfile = [
        0.21773151164616183,
        0.034941022753796035,
        0.06456056257136962,
        0.18781756363388397,
        0.07389116186240981,
        0.1696331916932108,
        0.4464499926416005,
        0.1706764273756967,
        0.15694190971549513,
        0.2882743403035651,
        0.32127549679527717,
        0.14116461052786136,
        0.5461613758054059,
        0.21958149674207716,
        0.25715084860896853,
        0.21525778612323568,
        0.15916222011475448,
        0.11014921269063864,
        0.2532504131880449,
        0.3154483508270503,
        0.08412368254727028,
        0.06337996942065917,
        0.12431489082721527,
        0.17319120880651773,
    ]
    operationRateMax = pd.DataFrame(
        [u for day in range(ndays) for u in dailyProfile],
        index=range(nhours),
        columns=["Region1"],
    ).round(6)
    capacityMaxWind = pd.Series([400], index=["Region1"])
    # Define min production
    minProduction = (
        operationRateMax["Region1"].sum() * capacityMaxWind.loc["Region1"] * 0.6
    )
    investPerCapacity, opexPerCapacity = 1200, 1200 * 0.02
    interestRate, economicLifetime = 0.08, 20

    # 1) specify balanceLimit

    balanceLimit = pd.DataFrame(index=["hydrogenDriver"], columns=["Region1"])
    # Define negative balanceLimit as driver. Because sink is contributing negatively.
    balanceLimit.loc["hydrogenDriver", "Region1"] = -1 * minProduction

    # 2) Initialize esM with one region with use of parameters balanceLimit and lowerBound

    # Use upperBound in a mathematical sense. abs(Sink Operation) >= abs(balanceLimit), both negative because Sink.
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=balanceLimit,
    )

    # 3) Components are added: 'Wind turbines', 'Electrolyzer', 'Batteries' and 'Hydrogen Annual Production'
    # 'Hydrogen Annual Production' is included in the balanceLimit analysis

    # Define Hydrogen Annual Production and added to Energysystem
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen Annual Production",
            balanceLimitID="hydrogenDriver",
            commodity="hydrogen",
            hasCapacityVariable=False,
        )
    )

    # Define Wind turbines and added to Energysystem
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind turbines",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=operationRateMax,
            capacityMax=capacityMaxWind,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Define techno-economic parameters of Batteries and added to Energysystem
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 150, 150 * 0.01
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000

    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=chargeEfficiency,
            cyclicLifetime=cyclicLifetime,
            dischargeEfficiency=dischargeEfficiency,
            selfDischarge=selfDischarge,
            chargeRate=chargeRate,
            dischargeRate=dischargeRate,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Define techno-economic parameters Electrolyzers and added to Energysystem
    invest_per_capacity_electrolysis = 0.5  # unit 1e9 EUR/GW
    opex_electrolysis = 0.025  # in %/100 of capex
    interestRate_electrolysis = 0.08  # in %/100
    economicLifetime_electrolysis = 10  # in years
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"MW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=invest_per_capacity_electrolysis,
            opexPerCapacity=invest_per_capacity_electrolysis * opex_electrolysis,
            interestRate=interestRate_electrolysis,
            economicLifetime=economicLifetime_electrolysis,
        )
    )
    # 4) Optimize model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # 5) The balanceLimit is compared to the outcome of the model
    # Hydrogen Annual Production >= hydrogenDriver (as balanceLimitID)
    for i, loc in enumerate(esM.locations):
        # Get operation of Renewables for loc
        operation_hydrogen = (
            esM.componentModelingDict["SourceSinkModel"]
            .operationVariablesOptimum.loc["Hydrogen Annual Production", loc]
            .sum()
        )
        tolerance = 0.001
        ## Compare modelled lowerLimit to limit set in constraint.
        test_min_production = (
            ndays * sum(dailyProfile) * capacityMaxWind.loc["Region1"] * 0.6
        )
        # Assert that the min. production requirement was indeed fulfilled (must be).
        assert operation_hydrogen > test_min_production * (1 - tolerance)
        # Assert that produced volume does also not exceed the min. requirements (should be
        # the case for linear models and TAC as cost function but could change in the future).
        assert operation_hydrogen < test_min_production * (1 + tolerance)


# In this test the following steps are performed:
# 0) Preprocess energy system model
# 1) Define CO2_limit (balanceLimit) for CO2 limit dependent on the demand
# 2) Initialize EnergySystemModel with two Regions ('Region1', 'Region2')
# 3) Components are added: 'Electricity demand', 'Methane purchase', 'cctg', 'CO2 to environment',
# 'Fuel Cell', 'Batteries'
# 4) Optimize Model
# 5) The CO2_limit is compared to the outcome of the model
# (sink are defined negative)
def test_CO2Limit():
    # 0) Preprocess energy system model
    locations = {"Region1", "Region2"}
    commodityUnitDict = {
        "electricity": r"MW$_{el}$",
        "methane": r"MW$_{CH_{4},LHV}$",
        "CO2": r"Mio. kg$_{CO_2}$/h",
    }
    commodities = {"electricity", "methane", "CO2"}
    ndays = 30
    nhours = 24 * ndays

    ## Define Electricity Demand
    dailyProfileSimple = [
        0.6,
        0.6,
        0.6,
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
        0.8,
    ]
    demand = pd.DataFrame(
        [
            [(u + 0.1 * np.random.rand()) * 40, (u + 0.1 * np.random.rand()) * 60]
            for day in range(ndays)
            for u in dailyProfileSimple
        ],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)

    # 1) Define CO2-Limit with balanceLimitConstraint (sink are defined negative)
    CO2_limit = pd.DataFrame(index=["CO2 limit"], columns=["Total", "lowerBound"])
    CO2_limit.loc["CO2 limit"] = [
        -1 * demand.sum().sum() * 0.6 * 201 * 1e-6 / 0.6,
        True,
    ]
    # 2) Initialize EnergySystemModel with two Regions
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=CO2_limit,
    )

    # 3) Components are added: 'Electricity demand', 'Methane purchase', 'cctg', 'CO2 to environment',
    # 'Fuel Cell', 'Batteries'

    # Define Electricity demand and added to Energysystem
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    # Define Methane purchase and added to Energysystem
    esM.add(
        fn.Source(
            esM=esM,
            name="Methane purchase",
            commodity="methane",
            hasCapacityVariable=False,
        )
    )
    # Define ccgt and added to Energysystem
    esM.add(
        fn.Conversion(
            esM=esM,
            name="ccgt",
            physicalUnit=r"MW$_{el}$",
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
    # Define CO2 to environment and added to Energysystem
    esM.add(
        fn.Sink(
            esM=esM,
            name="CO2 to environment",
            commodity="CO2",
            hasCapacityVariable=False,
            balanceLimitID="CO2 limit",
        )
    )
    ## Wind turbines
    # Define Fuel Cell and added to Energysystem
    opexPerOperation = 150 / 1e6
    interestRate, economicLifetime = 0.08, 20
    esM.add(
        fn.Source(
            esM=esM,
            name="Fuel Cell",
            commodity="electricity",
            hasCapacityVariable=False,
            opexPerOperation=opexPerOperation,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # Define Batteries and added to Energysystem
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 150, 150 * 0.01
    interestRate, economicLifetime, cyclicLifetime = 0.08, 22, 10000

    esM.add(
        fn.Storage(
            esM=esM,
            name="Batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            chargeEfficiency=chargeEfficiency,
            cyclicLifetime=cyclicLifetime,
            dischargeEfficiency=dischargeEfficiency,
            selfDischarge=selfDischarge,
            chargeRate=chargeRate,
            dischargeRate=dischargeRate,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )

    # 4) Optimize model
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    co2_to_environment = 0

    # 5) The CO2_limit is compared to the outcome of the model
    # (sinks are defined negative)
    for i, loc in enumerate(esM.locations):
        # Get operation of Renewables for loc
        co2_to_environment += (
            esM.componentModelingDict["SourceSinkModel"]
            .operationVariablesOptimum.loc["CO2 to environment", loc]
            .sum()
        )
    tolerance = 0.001
    ## Compare modeled co2 emissions to limit set in constraint.
    assert (
        co2_to_environment * (1 - tolerance) < -1 * CO2_limit.loc["CO2 limit", "Total"]
    )
    assert (
        co2_to_environment * (1 + tolerance) > -1 * CO2_limit.loc["CO2 limit", "Total"]
    )
