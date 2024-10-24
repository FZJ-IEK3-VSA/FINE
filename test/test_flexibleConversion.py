import fine as fn
import pandas as pd
import pytest

def test_flexibleConversion_init():

    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"electricity", "hydrogen", "nat_gas"},
        numberOfTimeSteps=1,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
            "nat_gas": r"kW$_{CH_{4},LHV}$",
        },
        hoursPerTimeStep=8760,
        startYear=2020,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="hydrogen_import",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost=1,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="nat_gas_import",
            commodity="nat_gas",
            hasCapacityVariable=False,
            commodityCost=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="FC_flex",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "in": {
                    "hydrogen": -4,
                    "nat_gas": -2,
                }
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            interestRate=0,
            economicLifetime=10,
            # flexibleConversion=True,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="flex",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": -1,
                "in": {
                    "hydrogen": 4,
                    "nat_gas": 2,
                }
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            interestRate=0,
            economicLifetime=10,
            # flexibleConversion=True,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="FC_fix",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "hydrogen": -2,
                "nat_gas": -1,
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Industry site",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                [100],
                columns=["loc1"]
            )
        )
    )
    with pytest.raises(ValueError, match=r".*All commodity conversion factors of.*"):
        esM.add(
            fn.Conversion(
                esM=esM,
                name="conversion_flex",
                physicalUnit=r"kW$_{el}$",
                commodityConversionFactors={
                    "electricity": 1,
                    "in": {
                        "hydrogen": 4,
                        "nat_gas": -2,
                    }
                },
                hasCapacityVariable=True,
                investPerCapacity=0,
                interestRate=0,
                economicLifetime=10,
            )
        )

    with pytest.raises(ValueError, match=r".*Commodity group names must be different from commodity names.*"):
        esM.add(
            fn.Conversion(
                esM=esM,
                name="conversion_flex2",
                physicalUnit=r"kW$_{el}$",
                commodityConversionFactors={
                    "electricity": 1,
                    "hydrogen": {
                        "hydrogen": -4,
                        "hydrogen_ren": -4,
                    },
                },
                hasCapacityVariable=True,
                investPerCapacity=0,
                interestRate=0,
                economicLifetime=10,
            )
        )
    with pytest.raises(NotImplementedError, match=r".*The combination of flexible and.*"):
        esM.add(
            fn.Conversion(
                esM=esM,
                name="conversion_flex2",
                physicalUnit=r"kW$_{el}$",
                commodityConversionFactors={
                    (2020, 2020): {
                        "electricity": 1,
                        "hydrogen": -4,
                        "ch4": {
                            "nat_gas": -2,
                            "sng": -2.5,
                            "bio_gas": -3,
                        },
                    },
                    (2020, 2025): {
                        "electricity": 1,
                        "hydrogen": -4,
                        "ch4": {
                            "nat_gas": -2,
                            "sng": -2.5,
                            "bio_gas": -3,
                        },
                    },
                    (2025, 2025): {
                        "electricity": 1,
                        "hydrogen": -4,
                        "ch4": {
                            "nat_gas": -2,
                            "sng": -2.5,
                            "bio_gas": -3,
                        },
                    },
                },
                hasCapacityVariable=True,
                investPerCapacity=0,
                interestRate=0,
                economicLifetime=10,
            )
        )

def test_flexibleConversion_groups():
    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"co2", "electricity", "hydrogen", "hydrogen_ren", "nat_gas", "bio_gas", "sng"},
        numberOfTimeSteps=1,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
            "hydrogen_ren": r"kW$_{H_{2},LHV}$",
            "nat_gas": r"kW$_{CH_{4},LHV}$",
            "bio_gas": r"kW$_{CH_{4},LHV}$",
            "sng": r"kW$_{CH_{4},LHV}$",
            "co2": r"Mio. t$_{CO_2}$/h",
        },
        hoursPerTimeStep=8760,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        startYear=2020,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="hydrogen_import",
            commodity="hydrogen",
            hasCapacityVariable=False,
            #commodityCost=1,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="nat_gas_import",
            commodity="nat_gas",
            hasCapacityVariable=False,
            commodityCost=1/8760,
            interestRate=0
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="bio_gas_import",
            commodity="bio_gas",
            hasCapacityVariable=False,
            commodityCost=10,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="sng_import",
            commodity="sng",
            hasCapacityVariable=False,
            commodityCost=10,
        )
    )
    esM.add(
        fn.Conversion(
            esM=esM,
            name="conversion_norm",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "nat_gas": -2,
            },
            hasCapacityVariable=True,
            investPerCapacity=100,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="conversion_flex",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "h2": {
                    "hydrogen": -4,
                    "hydrogen_ren": -5,
                },
                "ch4": {
                    "nat_gas": -2,
                    "sng": -2.5,
                    "bio_gas": -3,
                },
            },
            emissionFactors={
                "co2": {
                    "nat_gas": 3,
                    "sng": 2,
                    "bio_gas": 1,
                }
            },
            hasCapacityVariable=True,
            investPerCapacity=10,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="conversion_flex2",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                2020: {
                    "electricity": 1,
                    "hydrogen": -4,
                    "ch4": {
                        "nat_gas": -2,
                        "sng": -2.5,
                        "bio_gas": -3,
                    },
                },
                2025: {
                    "electricity": 1,
                    "hydrogen": -4,
                    "ch4": {
                        "nat_gas": -2,
                        "sng": -2.5,
                        "bio_gas": -3,
                    },
                },
            },
            emissionFactors={
                "co2": {
                    "nat_gas": 3,
                    "sng": 2,
                    "bio_gas": 1,
                }
            },
            hasCapacityVariable=True,
            investPerCapacity=0,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series([8760])
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver='glpk')
    print("Objective Value: \n" + str(esM.pyM.Obj()))

@pytest.mark.parametrize("use_balanceLimit", [False, True])
def test_flexibleConversion_emissionFactors(use_balanceLimit):
    if use_balanceLimit:
        balanceLimit = {
            2020: pd.DataFrame(data=[[-38280, True]], columns=["loc1", "lowerBound"], index=["co2"]),
            2025: pd.DataFrame(data=[[-26280, True]], columns=["loc1", "lowerBound"], index=["co2"])
        }
    else:
        balanceLimit = None

    esM = fn.EnergySystemModel(
        locations={"loc1"},
        commodities={"co2", "electricity", "nat_gas", "bio_gas"},
        numberOfTimeSteps=1,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "nat_gas": r"kW$_{CH_{4},LHV}$",
            "bio_gas": r"kW$_{CH_{4},LHV}$",
            "co2": r"Mio. t$_{CO_2}$/h",
        },
        hoursPerTimeStep=8760,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        startYear=2020,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=balanceLimit,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="nat_gas_import",
            commodity="nat_gas",
            hasCapacityVariable=False,
            commodityCost=1/8760,
            interestRate=0
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="bio_gas_import",
            commodity="bio_gas",
            hasCapacityVariable=False,
            commodityCost=1/8760,
            interestRate = 0
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.Series([8760])
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="co2Sink",
            commodity="co2",
            hasCapacityVariable=False,
            balanceLimitID="co2"
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="co2Source",
            commodity="co2",
            hasCapacityVariable=False,
            balanceLimitID="co2"
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="conversion_flex",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "ch4": {
                    "nat_gas": -2,
                    "bio_gas": -3,
                },
            },
            emissionFactors={
                "co2": {
                    "nat_gas": 3,
                    "bio_gas": 1,
                }
            },
            hasCapacityVariable=True,
            investPerCapacity=10,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver='glpk')
    if use_balanceLimit:
        assert (
                esM.getOptimizationSummary('SourceSinkModel', ip=2020).
                loc['co2Sink', 'operation', '[Mio. t$_{CO_2}$/h*h/a]'].values[0] == 38280
        )
        assert (
                esM.getOptimizationSummary('SourceSinkModel', ip=2025).
                loc['co2Sink', 'operation', '[Mio. t$_{CO_2}$/h*h/a]'].values[0] == 26280
        )
        assert esM.pyM.op_flex_conv['loc1', 'conversion_flex', 0, 'ch4', 'nat_gas', 0, 0]._value == 4000
        assert esM.pyM.op_flex_conv['loc1', 'conversion_flex', 0, 'ch4', 'bio_gas', 0, 0]._value == 4760
        assert esM.pyM.op_flex_conv['loc1', 'conversion_flex', 1, 'ch4', 'nat_gas', 0, 0]._value == 0
        assert esM.pyM.op_flex_conv['loc1', 'conversion_flex', 1, 'ch4', 'bio_gas', 0, 0]._value == 8760

    else:
        assert esM.pyM.op_flex_conv['loc1', 'conversion_flex', 0, 'ch4', 'nat_gas', 0, 0]._value == 8760
        assert esM.pyM.op_flex_conv['loc1', 'conversion_flex', 1, 'ch4', 'nat_gas', 0, 0]._value == 8760

def test_flexibleConversionFlowShare():
    esM = fn.EnergySystemModel(
        locations={"loc1", "loc2"},
        commodities={"electricity", "nat_gas", "hydrogen"},
        numberOfTimeSteps=1,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
            "nat_gas": r"kW$_{CH_{4},LHV}$",
        },
        hoursPerTimeStep=8760,
        numberOfInvestmentPeriods=2,
        investmentPeriodInterval=5,
        startYear=2020,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="hydrogen_import",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost=1 / 8760,
        )
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="nat_gas_import",
            commodity="nat_gas",
            hasCapacityVariable=False,
            commodityCost=1 / 8760,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="conversion_flex",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "in": {
                    "hydrogen": -1,
                    "nat_gas": -2,
                }
            },
            flowShares={
                2020: {
                    'max': {
                        "hydrogen": pd.Series([0.5], index=['loc1'])
                    }
                },
                2025: {
                    'max': {
                        "hydrogen": 0.8
                    },
                    'min': {
                        "hydrogen": 0.6
                    }
                },
            },
            hasCapacityVariable=True,
            investPerCapacity=10,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="conversion_expensive",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                "electricity": 1,
                "nat_gas": -1.5,
            },
            hasCapacityVariable=True,
            investPerCapacity=10000,
            interestRate=0,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame([[8760, 8760]], columns=list(esM.locations))
        )
    )

    esM.optimize(timeSeriesAggregation=False, solver='glpk')
    assert (esM.pyM.op_flex_conv['loc1', 'conversion_flex', 0, 'in', 'hydrogen', 0, 0]._value <=
            0.5 * esM.pyM.op_conv['loc1', 'conversion_flex', 0, 0, 0]._value)
    assert (esM.pyM.op_flex_conv['loc2', 'conversion_flex', 1, 'in', 'hydrogen', 0, 0]._value >=
            0.6 * esM.pyM.op_conv['loc2', 'conversion_flex', 1, 0, 0]._value)
