import numpy as np
import pandas as pd
import pytest
import FINE as fn
import copy


def test_perfectForesight_variableConversions_input(
    perfectForesight_test_esM,
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    # 1. Variation of the commodity conversion per investment period
    # e.g. due to weather differencecs
    # note: electrolyzers just exemplary for usage
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                2020: {"electricity": -1, "hydrogen": 0.7},
                2025: {"electricity": -1, "hydrogen": 0.71},
                2030: {"electricity": -1, "hydrogen": 0.72},
                2035: {"electricity": -1, "hydrogen": 0.73},
                2040: {"electricity": -1, "hydrogen": 0.74},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    assert list(
        esM.getComponent("Electrolyzers").processedCommodityConversionFactors
    ) == [0, 1, 2, 3, 4]
    assert (
        esM.getComponent("Electrolyzers").processedCommodityConversionFactors[1][
            "hydrogen"
        ]
        == 0.71
    )

    # 2. Variation for commissioning year and investment period
    # e.g. due to efficiency and weather dependency
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers1",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.65},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.7},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.7},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.71},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.71},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.72},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.72},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.73},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.73},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.74},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    assert (-1, 0) in esM.getComponent(
        "Electrolyzers1"
    ).processedCommodityConversionFactors.keys()
    assert (
        esM.getComponent("Electrolyzers1").processedCommodityConversionFactors[(2, 3)][
            "hydrogen"
        ]
        == 0.72
    )

    # 3. test for time-series
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers2",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.7, 0.84], [0.7, 0.84]]),
                    ),
                },
                (2020, 2020): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.84], [0.71, 0.84]]),
                    ),
                },
                (2020, 2025): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
                (2025, 2025): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
                (2025, 2030): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2030, 2030): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
                (2030, 2035): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2035, 2035): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2035, 2040): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2040, 2040): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )

    assert (
        esM.getComponent("Electrolyzers2")
        .fullCommodityConversionFactors[(2, 3)]["hydrogen"]
        .values
        == np.array([[0.83, 0.71], [0.83, 0.71]])
    ).all()

    # 4. error in the input of the commodity conversion
    with pytest.raises(
        ValueError, match=r".*Wrong format for commodityConversionFactors.*"
    ):
        esM.add(
            fn.Conversion(
                esM=esM,
                name="Electrolyzers3",
                physicalUnit=r"kW$_{el}$",
                commodityConversionFactors={
                    (2015, 2025): {"electricity": -1, "hydrogen": 0.7},
                    (2020, 2020): {"electricity": -1, "hydrogen": 0.7},
                    (2020, 2025): {"electricity": -1, "hydrogen": 0.7},
                    (2025, 2025): {"electricity": -1, "hydrogen": 0.71},
                    (2025, 2030): {"electricity": -1, "hydrogen": 0.71},
                    (2030, 2030): {"electricity": -1, "hydrogen": 0.72},
                    (2030, 2035): {"electricity": -1, "hydrogen": 0.72},
                    (2035, 2035): {"electricity": -1, "hydrogen": 0.73},
                    (2035, 2040): {"electricity": -1, "hydrogen": 0.73},
                    (2040, 2040): {"electricity": -1, "hydrogen": 0.74},
                },
                hasCapacityVariable=True,
                investPerCapacity=500,  # euro/kW
                opexPerCapacity=500 * 0.025,
                interestRate=0.08,
                economicLifetime=10,
                stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
            )
        )
    # 5. error in the input of the commodity conversion
    with pytest.raises(
        ValueError, match=r".*Unallowed data type variation.*"
    ):  # only one hydrogen conversion is a time-series
        esM.add(
            fn.Conversion(
                esM=esM,
                name="Electrolyzers4",
                physicalUnit=r"kW$_{el}$",
                commodityConversionFactors={
                    (2015, 2020): {
                        "electricity": -1,
                        "hydrogen": pd.Series(index=[0, 1], data=[0.7, 0.8]),
                    },
                    (2020, 2020): {"electricity": -1, "hydrogen": 0.7},
                    (2020, 2025): {"electricity": -1, "hydrogen": 0.7},
                    (2025, 2025): {"electricity": -1, "hydrogen": 0.71},
                    (2025, 2030): {"electricity": -1, "hydrogen": 0.71},
                    (2030, 2030): {"electricity": -1, "hydrogen": 0.72},
                    (2030, 2035): {"electricity": -1, "hydrogen": 0.72},
                    (2035, 2035): {"electricity": -1, "hydrogen": 0.73},
                    (2035, 2040): {"electricity": -1, "hydrogen": 0.73},
                    (2040, 2040): {"electricity": -1, "hydrogen": 0.74},
                },
                hasCapacityVariable=True,
                investPerCapacity=500,  # euro/kW
                opexPerCapacity=500 * 0.025,
                interestRate=0.08,
                economicLifetime=10,
                stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
            )
        )

    # 6. add commodity conversion commis depending, even if data is not
    # depending on commis. correct internally to reduce computational load
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzers5",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.7},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.7},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.75},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.75},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.76},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.76},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.78},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.78},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.79},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.79},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    assert esM.getComponent("Electrolyzers5").isCommisDepending == False


@pytest.mark.parametrize("use_tsa", [True, False])
def test_perfectForesight_variableConversions_timeindependent(
    use_tsa,
    perfectForesight_test_esM,
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.39},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.4},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.4},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.5},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.5},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.6},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.6},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.7},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.7},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.8},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    # add a hydrogen demand
    demand = {}
    _demand = pd.DataFrame(
        columns=["PerfectLand", "ForesightLand"],
        data=[
            [
                1000,
                2000,
            ],
            [2000, 1000],
        ],
    )
    demand[2020] = _demand
    demand[2025] = _demand
    demand[2030] = _demand * 1.2
    demand[2035] = _demand * 1.2
    demand[2040] = _demand * 1.2

    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    if use_tsa is False:
        esM.optimize()

    else:
        esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=1,
            segmentation=False,
            sortValues=True,
            representationMethod=None,
            rescaleClusterPeriods=True,
        )

        esM.optimize(timeSeriesAggregation=True, solver="glpk")
    assert (
        round(
            esM.pyM.op_commis_conv.get_values()[
                "PerfectLand", "Electrolyzer", 0, 0, 0, 0
            ]
            * 0.4
            + esM.pyM.op_commis_conv.get_values()[
                "PerfectLand", "Electrolyzer", -1, 0, 0, 0
            ]
            * 0.39,
            4,
        )
        == 1000
    )


@pytest.mark.parametrize("use_tsa", [True, False])
def test_perfectForesight_variableConversions_timedepending(
    use_tsa, perfectForesight_test_esM
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.6, 0.84], [0.6, 0.84]]),
                    ),
                },
                (2020, 2020): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.84], [0.71, 0.84]]),
                    ),
                },
                (2020, 2025): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
                (2025, 2025): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
                (2025, 2030): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2030, 2030): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
                (2030, 2035): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2035, 2035): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2035, 2040): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.83], [0.71, 0.83]]),
                    ),
                },
                (2040, 2040): {
                    "electricity": -1,
                    "hydrogen": pd.DataFrame(
                        index=[0, 1],
                        columns=["PerfectLand", "ForesightLand"],
                        data=np.array([[0.71, 0.8], [0.71, 0.8]]),
                    ),
                },
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    # add a hydrogen demand
    demand = {}
    _demand = pd.DataFrame(
        columns=["PerfectLand", "ForesightLand"],
        data=[
            [
                1000,
                2000,
            ],
            [2000, 1000],
        ],
    )
    demand[2020] = _demand
    demand[2025] = _demand
    demand[2030] = _demand * 1.2
    demand[2035] = _demand * 1.2
    demand[2040] = _demand * 1.2

    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    if use_tsa is False:
        esM.optimize()

        assert (
            round(
                esM.pyM.op_commis_conv.get_values()[
                    "PerfectLand", "Electrolyzer", 0, 0, 0, 0
                ]
                * 0.71
                + esM.pyM.op_commis_conv.get_values()[
                    "PerfectLand", "Electrolyzer", -1, 0, 0, 0
                ]
                * 0.7,
                5,
            )
            == 1000
        )
    else:
        esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=1,
            segmentation=False,
            sortValues=True,
            representationMethod=None,
            rescaleClusterPeriods=True,
        )

        esM.optimize(timeSeriesAggregation=True, solver="glpk")

        # check that aggregation is correct
        assert np.array_equal(
            esM.getComponent("Electrolyzer")
            .aggregatedCommodityConversionFactors[(-1, 0)]["hydrogen"]
            .values[0],
            np.array([0.84, 0.6]),
        )
        assert np.array_equal(
            esM.getComponent("Electrolyzer")
            .aggregatedCommodityConversionFactors[(0, 0)]["hydrogen"]
            .values[0],
            np.array([0.84, 0.71]),
        )

        # get commodity conversion factor of perfect land with commis in -1 in the first time stepü
        commodConv_CommisYearMinusOne = (
            esM.getComponent("Electrolyzer")
            .processedCommodityConversionFactors[(-1, 0)]["hydrogen"]
            .loc[:, "PerfectLand"][0][0]
        )
        commodConv_CommisYearZero = (
            esM.getComponent("Electrolyzer")
            .processedCommodityConversionFactors[(0, 0)]["hydrogen"]
            .loc[:, "PerfectLand"][0][0]
        )

        assert (
            round(
                esM.pyM.op_commis_conv.get_values()[
                    "PerfectLand", "Electrolyzer", 0, 0, 0, 0
                ]
                * commodConv_CommisYearZero
                + esM.pyM.op_commis_conv.get_values()[
                    "PerfectLand", "Electrolyzer", -1, 0, 0, 0
                ]
                * commodConv_CommisYearMinusOne,
                4,
            )
            == 1000
        )


@pytest.mark.parametrize("use_tsa", [True, False])
def test_perfectForesight_variableConversions_opeationRateMax(
    use_tsa,
    perfectForesight_test_esM,
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.39},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.4},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.4},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.5},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.5},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.6},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.6},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.7},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.7},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.8},
            },
            hasCapacityVariable=True,
            operationRateMax=pd.DataFrame(
                index=[0, 1], columns=["PerfectLand", "ForesightLand"], data=0.570776
            ),
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    # add a hydrogen demand
    demand = {}
    _demand = pd.DataFrame(
        columns=["PerfectLand", "ForesightLand"],
        data=[
            [
                1000,
                2000,
            ],
            [2000, 1000],
        ],
    )
    demand[2020] = _demand
    demand[2025] = _demand
    demand[2030] = _demand * 1.2
    demand[2035] = _demand * 1.2
    demand[2040] = _demand * 1.2
    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    if use_tsa is False:
        esM.optimize()
        timeStepList = [0, 1]

    else:
        esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=1,
            segmentation=False,
            sortValues=True,
            representationMethod=None,
            rescaleClusterPeriods=True,
        )
        esM.optimize(timeSeriesAggregation=True, solver="glpk")
        timeStepList = [0]

    # test the sum of the operation
    processedOperation = (
        esM.getComponent("H2Demand")
        .processedOperationRateFix[0]
        .loc[:, "PerfectLand"][0][0]
    )
    assert (
        round(
            esM.pyM.op_commis_conv.get_values()[
                "PerfectLand", "Electrolyzer", 0, 0, 0, 0
            ]
            * 0.4
            + esM.pyM.op_commis_conv.get_values()[
                "PerfectLand", "Electrolyzer", -1, 0, 0, 0
            ]
            * 0.39,
            2,
        )
        == processedOperation
    )

    # check that operationRateMax is kept in for installed capacity in each commissioning year
    # operation rate max : 0.570776
    # duration of time step after aggregation: 8760
    for region in ["PerfectLand", "ForesightLand"]:
        for ip in [0, 1, 2, 3, 4]:
            for commis in [-1, 0, 1, 2, 3, 4]:
                if ip - commis >= 2 or commis > ip:  # only 10 years of lifetime
                    continue
                for ts in [0]:
                    allowed_energy_production = round(
                        esM.pyM.commis_conv.get_values()[region, "Electrolyzer", commis]
                        * 0.570776
                        * esM.hoursPerTimeStep,
                        2,
                    )
                    produced_energy = round(
                        esM.pyM.op_commis_conv.get_values()[
                            region, "Electrolyzer", commis, ip, 0, ts
                        ],
                        2,
                    )
                    assert allowed_energy_production >= produced_energy


@pytest.mark.parametrize("use_tsa", [True])
def test_perfectForesight_variableConversions_opeationRateFix(
    use_tsa,
    perfectForesight_test_esM,
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    # add a operation rate fix, so that the additional h2 source must be used to meet the demand
    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.39},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.4},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.4},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.5},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.5},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.6},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.6},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.7},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.7},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.8},
            },
            hasCapacityVariable=True,
            operationRateFix=pd.DataFrame(
                index=[0, 1], columns=["PerfectLand", "ForesightLand"], data=0.570776
            ),
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    # add a hydrogen demand
    demand = {}
    _demand = pd.DataFrame(
        columns=["PerfectLand", "ForesightLand"],
        data=[
            [
                1000,
                2000,
            ],
            [2000, 1000],
        ],
    )
    demand[2020] = _demand
    demand[2025] = _demand
    demand[2030] = _demand * 1.2
    demand[2035] = _demand * 1.2
    demand[2040] = _demand * 1.2
    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # add expensive source just used to balance out for operation rate fix
    esM.add(
        fn.Source(
            esM=esM,
            name="H2Source",
            commodity="hydrogen",
            hasCapacityVariable=False,
            opexPerOperation=100,
        )
    )
    if use_tsa is False:
        esM.optimize()
        timeStepList = [0, 1]

    else:
        esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=1,
            segmentation=False,
            sortValues=True,
            representationMethod=None,
            rescaleClusterPeriods=True,
        )
        esM.optimize(timeSeriesAggregation=True, solver="glpk")
        timeStepList = [0]

    # test the sum of the operation
    assert (
        round(
            esM.pyM.op_commis_conv.get_values()[
                "PerfectLand", "Electrolyzer", 0, 0, 0, 0
            ]
            * 0.4
            + esM.pyM.op_commis_conv.get_values()[
                "PerfectLand", "Electrolyzer", -1, 0, 0, 0
            ]
            * 0.39,
            2,
        )
        == esM.pyM.op_srcSnk.get_values()["PerfectLand", "H2Demand", 0, 0, 0]
    )

    # check that operationRateFix is kept in for installed capacity in each commissioning year
    # operation rate fix : 0.570776
    for region in ["PerfectLand", "ForesightLand"]:
        for ip in [0, 1, 2, 3, 4]:
            for commis in [-1, 0, 1, 2, 3, 4]:
                if ip - commis >= 2 or commis > ip:  # only 10 years of lifetime
                    continue
                for ts in timeStepList:
                    allowed_energy_production = round(
                        esM.pyM.commis_conv.get_values()[region, "Electrolyzer", commis]
                        * 0.570776
                        * esM.hoursPerTimeStep,
                        2,
                    )
                    produced_energy = round(
                        esM.pyM.op_commis_conv.get_values()[
                            region, "Electrolyzer", commis, ip, 0, ts
                        ],
                        2,
                    )
                    assert allowed_energy_production == produced_energy


@pytest.mark.parametrize("use_tsa", [True, False])
def test_perfectForesight_variableConversions_fullLoadHoursMax(
    use_tsa,
    perfectForesight_test_esM,
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    # check if the full load hour max is kept with a variable commodity conversion over the transformation pathway
    fullLoadHoursMax = 100

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.39},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.4},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.4},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.5},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.5},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.6},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.6},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.7},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.7},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.8},
            },
            hasCapacityVariable=True,
            yearlyFullLoadHoursMax=fullLoadHoursMax,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    # add a hydrogen demand
    demand = {}
    _demand = pd.DataFrame(
        columns=["PerfectLand", "ForesightLand"],
        data=[
            [
                1000,
                2000,
            ],
            [2000, 1000],
        ],
    )
    demand[2020] = _demand
    demand[2025] = _demand
    demand[2030] = _demand * 1.2
    demand[2035] = _demand * 1.2
    demand[2040] = _demand * 1.2
    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # add expensive source just used to balance out for operation rate fix
    esM.add(
        fn.Source(
            esM=esM,
            name="H2Source",
            commodity="hydrogen",
            hasCapacityVariable=False,
            opexPerOperation=100,
        )
    )
    if use_tsa is False:
        esM.optimize()
        timeStepList = [0, 1]
        factor = 1

    else:
        esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=1,
            segmentation=False,
            sortValues=True,
            representationMethod=None,
            rescaleClusterPeriods=True,
        )
        esM.optimize(timeSeriesAggregation=True, solver="glpk")
        timeStepList = [0]
        factor = 2

    # check that yearly full load hours max is kept for the install capacities for each commissioning year
    for region in ["PerfectLand", "ForesightLand"]:
        for ip in [0, 1, 2, 3, 4]:
            for commisYear in [-1, 0, 1, 2, 3, 4]:
                if ip - commisYear >= 2 or commisYear > ip:  # only 10 years of lifetime
                    continue
                if (
                    round(
                        esM.pyM.commis_conv.get_values()[
                            region, "Electrolyzer", commisYear
                        ],
                        4,
                    )
                    == 0
                ):
                    continue
                output_yearly_full_load_hours_max = 0
                for ts in timeStepList:
                    # get commissioning and operation
                    commis = round(
                        esM.pyM.commis_conv.get_values()[
                            region, "Electrolyzer", commisYear
                        ],
                        8,
                    )
                    operation_of_timestep = round(
                        esM.pyM.op_commis_conv.get_values()[
                            region, "Electrolyzer", commisYear, ip, 0, ts
                        ],
                        8,
                    )

                    # calculate full load hour of the time step (energy divided by capacity) if there is a commissioning
                    if commis > 0:
                        output_yearly_full_load_hours_max += round(
                            operation_of_timestep / commis, 4
                        )

                assert output_yearly_full_load_hours_max * factor <= fullLoadHoursMax


@pytest.mark.parametrize("use_tsa", [True, False])
def test_perfectForesight_variableConversions_fullLoadHoursMin(
    use_tsa,
    perfectForesight_test_esM,
):
    esM = copy.deepcopy(perfectForesight_test_esM)
    # check if the full load hour min is kept with a variable commodity conversion over the transformation pathway
    fullLoadHoursMin = 100

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2015, 2020): {"electricity": -1, "hydrogen": 0.39},
                (2020, 2020): {"electricity": -1, "hydrogen": 0.4},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.4},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.5},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.5},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.6},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.6},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.7},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.7},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.8},
            },
            hasCapacityVariable=True,
            capacityMin=2,  # add capacity min to force a commissioning
            yearlyFullLoadHoursMin=fullLoadHoursMin,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
            stockCommissioning={2015: pd.Series(index=esM.locations, data=[0, 1])},
        )
    )
    # add a hydrogen demand as sink
    demand = {}
    _demand = pd.DataFrame(
        columns=["PerfectLand", "ForesightLand"],
        data=[
            [
                1000,
                2000,
            ],
            [2000, 1000],
        ],
    )
    demand[2020] = _demand
    demand[2025] = _demand
    demand[2030] = _demand * 1.2
    demand[2035] = _demand * 1.2
    demand[2040] = _demand * 1.2
    esM.add(
        fn.Sink(
            esM=esM,
            name="H2Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    # add cheap source
    esM.add(
        fn.Source(
            esM=esM,
            name="H2Source",
            commodity="hydrogen",
            hasCapacityVariable=False,
            opexPerOperation=0.001,
        )
    )
    if use_tsa is False:
        esM.optimize()
        timeStepList = [0, 1]
        factor = 1

    else:
        esM.aggregateTemporally(
            numberOfTypicalPeriods=1,
            numberOfTimeStepsPerPeriod=1,
            segmentation=False,
            sortValues=True,
            representationMethod=None,
            rescaleClusterPeriods=True,
        )
        esM.optimize(timeSeriesAggregation=True, solver="glpk")
        timeStepList = [0]
        factor = 2

    # check that yearly full load hours min is kept for the install capacities for each commissioning year
    # duration of time step :  4380
    for region in ["PerfectLand", "ForesightLand"]:
        for ip in [0, 1, 2, 3, 4]:
            for commisYear in [-1, 0, 1, 2, 3, 4]:
                if ip - commisYear >= 2 or commisYear > ip:  # only 10 years of lifetime
                    continue
                if (
                    round(
                        esM.pyM.commis_conv.get_values()[
                            region, "Electrolyzer", commisYear
                        ],
                        4,
                    )
                    == 0
                ):
                    continue
                output_yearly_full_load_hours_min = 0
                for ts in timeStepList:
                    # get commissioning and operation
                    commis = round(
                        esM.pyM.commis_conv.get_values()[
                            region, "Electrolyzer", commisYear
                        ],
                        8,
                    )
                    operation_of_timestep = round(
                        esM.pyM.op_commis_conv.get_values()[
                            region, "Electrolyzer", commisYear, ip, 0, ts
                        ],
                        8,
                    )

                    # calculate full load hour of the time step (energy divided by capacity) if there is a commissioning
                    output_yearly_full_load_hours_min += round(
                        operation_of_timestep / commis, 2
                    )
                assert output_yearly_full_load_hours_min * factor >= fullLoadHoursMin
