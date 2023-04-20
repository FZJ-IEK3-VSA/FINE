import pytest
import sys
import os

import numpy as np
import pandas as pd

import FINE as fn


def test_Stock_wrongStockYears():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"PerfectLand"},
        commodities={"electricity"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        startYear=2020,
        numberOfInvestmentPeriods=6,
        investmentPeriodInterval=1,
        lengthUnit="km",
        verboseLogLevel=2,
    )

    with pytest.warns(None) as warnings:
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=4e6,
            investPerCapacity=2 * 2190,
            opexPerCapacity=0,
            interestRate=0.02,
            opexPerOperation=0.01,
            economicLifetime=5,
            technicalLifetime=6,
            stockCommissioning={2013: 2, 2017: 0, 2018: 5},
        )

    if not any(w for w in warnings if "Stock of component" in str(w)):
        raise ValueError(
            "Warning for stock with capacity older than technical lifetime is not raised."
        )


def stock_esM():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190

    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation", "IndustryLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
    )

    # time step length [h]
    timeStepLength = numberOfTimeSteps * hoursPerTimeStep

    ### Buy electricity at the electricity market
    costs = pd.DataFrame(
        [
            np.array(
                [
                    0.05,
                    0.0,
                    0.1,
                    0.051,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    revenues = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    maxpurchase = (
        pd.DataFrame(
            [
                np.array(
                    [
                        1e6,
                        1e6,
                        1e6,
                        1e6,
                    ]
                ),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
            ],
            index=["ElectrolyzerLocation", "IndustryLocation"],
        ).T
        * hoursPerTimeStep
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Electricity market",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=maxpurchase,
            commodityCostTimeSeries=costs,
            opexPerCapacity={0: 500 * 0.025},
            commodityRevenueTimeSeries=revenues,
        )
    )  # eur/kWh

    ### Electrolyzers
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity={-2: 550, -1: 350, 0: 500},  # euro/kW
            opexPerCapacity={-2: 550 * 0.025, -1: 350 * 0.025, 0: 500 * 0.025},
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={
                -2: pd.Series(
                    [1, 2], index=["ElectrolyzerLocation", "IndustryLocation"]
                ),
                -1: pd.Series(
                    [0, 0], index=["ElectrolyzerLocation", "IndustryLocation"]
                ),
            },
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
            investPerCapacity={0: 0.5, -1: 0.75, -5: 1, -19: 1.5},  # eur/kWh
            opexPerCapacity={
                -19: 550 * 0.025,
                -5: 350 * 0.025,
                -1: 350 * 0.025,
                0: 500 * 0.025,
            },
            opexPerChargeOperation={0: 0.002},
            opexPerDischargeOperation={0: 0.001},
            interestRate=0.08,
            economicLifetime=30,
            stockCommissioning={
                -1: pd.Series(
                    [1, 2], index=["ElectrolyzerLocation", "IndustryLocation"]
                ),
                -5: pd.Series(
                    [2, 2], index=["ElectrolyzerLocation", "IndustryLocation"]
                ),
                -19: pd.Series(
                    [1, 1], index=["ElectrolyzerLocation", "IndustryLocation"]
                ),
                -31: pd.Series(
                    [4, 8], index=["ElectrolyzerLocation", "IndustryLocation"]
                ),
            },
        )
    )

    ### Hydrogen pipelines
    pipeline_stock = {}
    pipeline_stock[-1] = pd.DataFrame()
    pipeline_stock[-1].loc["ElectrolyzerLocation", "IndustryLocation"] = 1
    pipeline_stock[-1].loc["IndustryLocation", "ElectrolyzerLocation"] = 1
    pipeline_stock[-1].loc["ElectrolyzerLocation", "ElectrolyzerLocation"] = 0
    pipeline_stock[-1].loc["IndustryLocation", "IndustryLocation"] = 0

    esM.add(
        fn.Transmission(
            esM=esM,
            name="Pipelines",
            commodity="hydrogen",
            hasCapacityVariable=True,
            opexPerCapacity={-2: 550 * 0.025, -1: 350 * 0.025, 0: 500 * 0.025},
            investPerCapacity={-1: 0.3, 0: 0.177},
            interestRate=0.08,
            economicLifetime=40,
            stockCommissioning=pipeline_stock,
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
                        0.0,
                        0.0,
                    ]
                ),
                np.array(
                    [
                        6e3,
                        6e3,
                        6e3,
                        6e3,
                    ]
                ),
            ],
            index=["ElectrolyzerLocation", "IndustryLocation"],
        ).T
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

    return esM


def test_stock():
    esM = stock_esM()

    # Check input of optimization
    # electrolyzers

    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedInvestPerCapacity")[-2][
            "IndustryLocation"
        ],
        550,
        rtol=0.005,
    )

    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedInvestPerCapacity")[-1][
            "IndustryLocation"
        ],
        350,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedInvestPerCapacity")[0][
            "IndustryLocation"
        ],
        500,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedOpexPerCapacity")[-2][
            "IndustryLocation"
        ],
        13.75,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedOpexPerCapacity")[-1][
            "IndustryLocation"
        ],
        8.75,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedOpexPerCapacity")[0][
            "IndustryLocation"
        ],
        12.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedStockCommissioning")[-1][
            "IndustryLocation"
        ],
        0,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedStockCommissioning")[-2][
            "IndustryLocation"
        ],
        2,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electrolyzers", "processedStockCommissioning")[-2][
            "ElectrolyzerLocation"
        ],
        1,
        rtol=0.005,
    )

    # storages
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedInvestPerCapacity")[0][
            "IndustryLocation"
        ],
        0.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedInvestPerCapacity")[-1][
            "IndustryLocation"
        ],
        0.75,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedInvestPerCapacity")[-19][
            "IndustryLocation"
        ],
        1.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedOpexPerCapacity")[0][
            "IndustryLocation"
        ],
        12.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedOpexPerCapacity")[-1][
            "IndustryLocation"
        ],
        8.75,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedOpexPerCapacity")[-19][
            "IndustryLocation"
        ],
        13.75,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedStockCommissioning")[-19][
            "IndustryLocation"
        ],
        1,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedStockCommissioning")[-5][
            "ElectrolyzerLocation"
        ],
        2,
        rtol=0.005,
    )

    # pipelines
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pipelines", "processedInvestPerCapacity")[-1][
            "ElectrolyzerLocation_IndustryLocation"
        ],
        0.3 * 0.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pipelines", "processedInvestPerCapacity")[0][
            "ElectrolyzerLocation_IndustryLocation"
        ],
        0.177 * 0.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pipelines", "processedOpexPerCapacity")[0][
            "ElectrolyzerLocation_IndustryLocation"
        ],
        500 * 0.025 * 0.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pipelines", "processedOpexPerCapacity")[-1][
            "ElectrolyzerLocation_IndustryLocation"
        ],
        350 * 0.025 * 0.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pipelines", "processedStockCommissioning")[-1][
            "ElectrolyzerLocation_IndustryLocation"
        ],
        1,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pipelines", "processedStockCommissioning")[-1][
            "IndustryLocation_ElectrolyzerLocation"
        ],
        1,
        rtol=0.005,
    )

    # special case for single year optimization
    # if parameter exists for only one investment period, values instead of
    # dict with values per investment period is returned.- not ip depending as only one year in dict
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electricity market", "processedOpexPerCapacity")[
            "IndustryLocation"
        ],
        12.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Electricity market", "processedOpexPerCapacity")[
            "IndustryLocation"
        ],
        12.5,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponentAttribute("Pressure tank", "processedOpexPerChargeOperation")[
            "IndustryLocation"
        ],
        0.002,
        rtol=0.005,
    )

    # check optimization output and pym model -> check commissioning and stock
    esM.optimize(solver="glpk")

    # pipelines
    np.testing.assert_allclose(
        esM.pyM.commis_trans.get_values()[
            ("ElectrolyzerLocation_IndustryLocation", "Pipelines", -1)
        ],
        1,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.pyM.commis_trans.get_values()[
            ("ElectrolyzerLocation_IndustryLocation", "Pipelines", 0)
        ],
        5999.001529680365,
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.pyM.commis_trans.get_values()[
            ("ElectrolyzerLocation_IndustryLocation", "Pipelines", -1)
        ]
        + esM.pyM.commis_trans.get_values()[
            ("ElectrolyzerLocation_IndustryLocation", "Pipelines", 0)
        ],
        esM.pyM.cap_trans.get_values()[
            ("ElectrolyzerLocation_IndustryLocation", "Pipelines", 0)
        ],
        rtol=0.005,
    )

    # storages
    assert sum(esM.pyM.commis_stor.get_values().values()), 9
    np.testing.assert_allclose(
        sum(esM.pyM.commis_stor.get_values().values()),
        esM.getComponent("Pressure tank").stockCapacityStartYear.sum(),
        rtol=0.005,
    )
    np.testing.assert_allclose(
        esM.getComponent("Pressure tank").stockCapacityStartYear["ElectrolyzerLocation"]
        + esM.pyM.commis_stor.get_values()[
            ("ElectrolyzerLocation", "Pressure tank", 0)
        ],
        esM.pyM.cap_stor.get_values()[("ElectrolyzerLocation", "Pressure tank", 0)],
        rtol=0.005,
    )

    np.testing.assert_allclose(
        esM.getOptimizationSummary("ConversionModel").loc[
            ("Electrolyzers", "commissioning", "[kW$_{el}$]"), "ElectrolyzerLocation"
        ],
        esM.pyM.commis_conv.get_values()[("ElectrolyzerLocation", "Electrolyzers", 0)],
        rtol=0.005,
    )
