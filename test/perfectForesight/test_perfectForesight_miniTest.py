import FINE as fn
import numpy as np
import pandas as pd
import math


def test_perfectForesight_mini(perfectForesight_test_esM):
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    np.testing.assert_almost_equal(
        perfectForesight_test_esM.pyM.Obj(), 11861.771783274202
    )


def test_perfectForesight_stock(perfectForesight_test_esM):
    esM = perfectForesight_test_esM
    PVoperationRateMax = esM.getComponent("PV").operationRateMax

    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            capacityMax=4e6,
            investPerCapacity=1e3,
            opexPerCapacity=1,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=10,
            stockCommissioning={
                2005: pd.Series([10, 5], index=["ForesightLand", "PerfectLand"]),
                2010: pd.Series([10, 5], index=["ForesightLand", "PerfectLand"]),
                2015: pd.Series([0.5, 0.25], index=["ForesightLand", "PerfectLand"]),
            },
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # CHECKS
    # check the objective value
    np.testing.assert_almost_equal(esM.pyM.Obj(), 11861.771783274202)

    # check some commissioning and decommissioning variables
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", -1)] == 0.25
    assert esM.pyM.decommis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 0.25
    assert esM.pyM.commis_srcSnk.get_values()[("ForesightLand", "PV", -1)] == 0.5
    assert esM.pyM.decommis_srcSnk.get_values()[("ForesightLand", "PV", 1)] == 0.5
    assert esM.pyM.commis_srcSnk.get_values()[("ForesightLand", "PV", -2)] == 10
    assert esM.pyM.decommis_srcSnk.get_values()[("ForesightLand", "PV", 0)] == 10
    assert esM.pyM.commis_srcSnk.get_values()[("ForesightLand", "PV", 0)] == 1.5
    assert esM.pyM.cap_srcSnk.get_values()[("ForesightLand", "PV", 0)] == 2

    # check processedStockCommissioning
    assert list(esM.getComponent("PV").processedStockCommissioning.keys()) == [-1, -2]
    assert perfectForesight_test_esM.getComponent("PV").processedStockYears == [-2, -1]

    # check that parameters are correctly setup
    # a) parameters which need to include stock years as commissioning year dependent
    assert list(esM.getComponent("PV").processedInvestPerCapacity.keys()) == [
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(esM.getComponent("PV").processedOpexPerCapacity.keys()) == [
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(esM.getComponent("PV").processedOpexIfBuilt.keys()) == [
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(esM.getComponent("PV").processedInvestIfBuilt.keys()) == [
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(esM.getComponent("PV").QPbound.keys()) == [-2, -1, 0, 1, 2, 3, 4]

    # b) parameters which do not need to include stock years
    assert list(esM.getComponent("PV").processedOpexPerOperation.keys()) == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert list(esM.getComponent("PV").processedOperationRateMax.keys()) == [
        0,
        1,
        2,
        3,
        4,
    ]

    # check the optimization summary
    srcSnk_optSum_2020 = esM.getOptimizationSummary("SourceSinkModel", ip=2020)
    assert (
        srcSnk_optSum_2020.loc[
            ("PV", "decommissioning", "[kW$_{el}$]"), "ForesightLand"
        ]
        == 10
    )
    assert (
        srcSnk_optSum_2020.loc[("PV", "capacity", "[kW$_{el}$]"), "ForesightLand"] == 2
    )
    assert (
        srcSnk_optSum_2020.loc[("PV", "commissioning", "[kW$_{el}$]"), "ForesightLand"]
        == 1.5
    )
    assert math.isnan(
        srcSnk_optSum_2020.loc[
            ("EDemand", "commissioning", "[kW$_{el}$]"), "ForesightLand"
        ]
    )

    # check getOptimalValue function
    assert (
        esM.componentModelingDict["SourceSinkModel"]
        .getOptimalValues(ip=2020)["commissioningVariablesOptimum"]["values"]
        .loc["PV", "ForesightLand"]
        == 1.5
    )
    assert (
        esM.componentModelingDict["SourceSinkModel"]
        .getOptimalValues(ip=2020)["decommissioningVariablesOptimum"]["values"]
        .loc["PV", "ForesightLand"]
        == 10
    )


def test_perfectForesight_storage_transmission(perfectForesight_test_esM):
    esM = perfectForesight_test_esM

    ### Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    ### Hydrogen filled somewhere
    esM.add(
        fn.Storage(
            esM=esM,
            name="Pressure tank",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ### Hydrogen pipelines
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines",
            commodity="hydrogen",
            hasCapacityVariable=True,
            investPerCapacity=0.177,
            interestRate=0.08,
            economicLifetime=40,
        )
    )

    ### Industry site
    demand = (
        pd.DataFrame(
            [
                np.array(
                    [
                        0.0,
                        0.0,
                    ]
                ),
                np.array(
                    [
                        6e3,
                        6e3,
                    ]
                ),
            ],
            index=["ForesightLand", "PerfectLand"],
        ).T
        * 4380
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
    esM.optimize(timeSeriesAggregation=False, solver="glpk")


def test_perfectForesight_binary():
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        numberOfTimeSteps=2,
        hoursPerTimeStep=4380,
        costUnit="1 Euro",
        numberOfInvestmentPeriods=3,
        investmentPeriodInterval=5,
        startYear=2020,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    # add a sink
    demand = {}
    demand[2020] = pd.DataFrame(
        columns=["PerfectLand"],
        data=[
            4380 * 1.1,
            1e3,
        ],
    )
    demand[2025] = pd.DataFrame(
        columns=["PerfectLand"],
        data=[
            4380 * 1.9,
            1e3,
        ],
    )
    demand[2030] = demand[2025]
    esM.add(
        fn.Sink(
            esM=esM,
            name="EDemand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # add PV
    PVoperationRateMax = pd.DataFrame(columns=["PerfectLand"], data=[1, 1])
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateMax=PVoperationRateMax,
            investPerCapacity=1e3,
            investIfBuilt=1e3,
            opexPerCapacity=1,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=10,
            hasIsBuiltBinaryVariable=True,
            bigM=10,
            capacityMin=2,
            stockCommissioning={
                2015: pd.Series([1], index=["PerfectLand"]),
            },
        )
    )

    # cheap electricity purchase -> no new PV required
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity purchase",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.2,
        )
    )

    esM.optimize(solver="glpk")

    # test commissioning variables
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", -1)] == 1
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 0
    assert esM.pyM.commis_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 2
    # test binary commissioning variables
    assert esM.pyM.commisBin_srcSnk.get_values()[("PerfectLand", "PV", 0)] == 0
    assert esM.pyM.commisBin_srcSnk.get_values()[("PerfectLand", "PV", 1)] == 1

    # check binary costs of stock capacity of component
    np.testing.assert_almost_equal(
        esM.getOptimizationSummary("SourceSinkModel", ip=2020).loc[
            "PV", "capexIfBuilt", "[1 Euro/a]"
        ]["PerfectLand"],
        535.2277429418442,
    )
    np.testing.assert_almost_equal(
        esM.getOptimizationSummary("SourceSinkModel", ip=2030).loc[
            "PV", "capexIfBuilt", "[1 Euro/a]"
        ]["PerfectLand"],
        439.0731689683585,
    )
    np.testing.assert_almost_equal(
        esM.getOptimizationSummary("SourceSinkModel", ip=2025).loc[
            "PV", "commissioning", "[kW$_{el}$]"
        ]["PerfectLand"],
        2,
    )


def test_perfectForsight_annuityPerpetuity(perfectForesight_test_esM):
    perfectForesight_test_esM.annuityPerpetuity = True
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    np.testing.assert_almost_equal(
        perfectForesight_test_esM.pyM.Obj(), 31984.802368949295
    )
