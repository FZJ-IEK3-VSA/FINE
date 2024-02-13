import pandas as pd
import pytest

import fine as fn


def test_capacityBounds():
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation", "IndustryLocation"},
        commodities={"electricity", "hydrogen"},
        startYear=2020,
        numberOfInvestmentPeriods=5,
        investmentPeriodInterval=5,
        numberOfTimeSteps=4,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=4,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
        balanceLimit=None,
    )
    locations = ["ElectrolyzerLocation", "IndustryLocation"]

    # 1. Test without ip-dependency
    # 1.1 Test without ip-dependency for 1 dim components
    capMaxSeries = pd.Series(index=locations, data=5)
    capMinSeries = pd.Series(index=locations, data=1)
    capacityTypes = [(5, 1, "_scalar"), (capMaxSeries, capMinSeries, "_series")]
    for capacityMax, capacityMin, type in capacityTypes:
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank" + type,
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                capacityMax=capacityMax,
                capacityMin=capacityMin,
                capacityFix=None,
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )
        # capacityFix
        assert esM.getComponent("PressureTank" + type).capacityFix == None
        assert esM.getComponent("PressureTank" + type).processedCapacityFix is None
        # capacity max and min
        assert isinstance(
            esM.getComponent("PressureTank" + type).processedCapacityMax, dict
        )
        assert isinstance(
            esM.getComponent("PressureTank" + type).processedCapacityMax, dict
        )
        for ip in esM.investmentPeriods:
            assert (
                esM.getComponent("PressureTank" + type).processedCapacityMax[ip]
                == capMaxSeries
            ).all()
            assert (
                esM.getComponent("PressureTank" + type).processedCapacityMin[ip]
                == capMinSeries
            ).all()

    # 1.2 Test without ip-dependency for 2 dim components
    capMaxDataFrame = pd.DataFrame(index=locations, columns=locations)
    capMaxDataFrame.loc["ElectrolyzerLocation", "IndustryLocation"] = 5
    capMaxDataFrame.loc["IndustryLocation", "ElectrolyzerLocation"] = 5
    capMinDataFrame = pd.DataFrame(index=locations, columns=locations, data=0)
    capacityTypes = [(5, 0, "_scalar"), (capMaxDataFrame, capMinDataFrame, "_series")]
    for capacityMax, capacityMin, type in capacityTypes:
        esM.add(
            fn.Transmission(
                esM=esM,
                name="Pipelines" + type,
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityMax=capacityMax,
                capacityMin=capacityMin,
                investPerCapacity=0.177,
                interestRate=0.08,
                economicLifetime=40,
            )
        )
        assert isinstance(
            esM.getComponent("Pipelines" + type).processedCapacityMin, dict
        )
        assert isinstance(
            esM.getComponent("Pipelines" + type).processedCapacityMax, dict
        )
        assert esM.getComponent("Pipelines" + type).processedCapacityFix is None
        assert list(
            esM.getComponent("Pipelines" + type).locationalEligibility.index
        ) == [
            "ElectrolyzerLocation_IndustryLocation",
            "IndustryLocation_ElectrolyzerLocation",
        ]
        for ip in esM.investmentPeriods:
            assert (
                esM.getComponent("Pipelines" + type).processedCapacityMin[ip][
                    "ElectrolyzerLocation_IndustryLocation"
                ]
                == 0
            )
            assert (
                esM.getComponent("Pipelines" + type).processedCapacityMin[ip][
                    "IndustryLocation_ElectrolyzerLocation"
                ]
                == 0
            )
            assert (
                esM.getComponent("Pipelines" + type).processedCapacityMax[ip][
                    "ElectrolyzerLocation_IndustryLocation"
                ]
                == 5
            )
            assert (
                esM.getComponent("Pipelines" + type).processedCapacityMax[ip][
                    "IndustryLocation_ElectrolyzerLocation"
                ]
                == 5
            )

    # 2. Test with ip-dependency - dict format
    # 2.1 Test with ip-dependency for 1 dim components
    # test for dict format - no failure
    esM.add(
        fn.Storage(
            esM=esM,
            name="PressureTank",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            capacityMax={2020: 5, 2025: 4, 2030: 3.5, 2035: 3, 2040: 2},
            capacityMin={2020: 1, 2025: 1, 2030: 0.5, 2035: 0, 2040: 0},
            capacityFix=None,
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    # error for None value
    with pytest.raises(ValueError, match=r".*a dict containing None values.*"):
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank",
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                capacityMax={2020: 5, 2025: 4, 2030: None, 2035: 3, 2040: 0},
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )

    # error for capacityMax<capacityMin - in year 4
    with pytest.raises(
        ValueError, match=r".*capacityMin values > capacityMax values detected.*"
    ):
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank",
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                capacityMax={2020: 5, 2025: 4, 2030: 3.5, 2035: 3, 2040: 0},
                capacityMin={2020: 1, 2025: 1, 2030: 0.5, 2035: 0, 2040: 2},
                capacityFix=None,
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )

    # error for wrong ip input - missing of year 4
    with pytest.raises(ValueError, match=r".*but the expected years are.*"):
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank",
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                capacityMax={2020: 5, 2025: 4, 2030: 3.5, 2035: 3},
                capacityMin={2020: 1, 2025: 1, 2030: 0.5, 2035: 0},
                capacityFix=None,
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )

    # error for capacityMax and capacityFix
    with pytest.raises(
        ValueError, match=r".*capacityFix values > capacityMax values detected.*"
    ):
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank",
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                capacityFix={2020: 5, 2025: 4, 2030: 3.5, 2035: 3, 2040: 3},
                capacityMax={2020: 1, 2025: 1, 2030: 0.5, 2035: 0, 2040: 0},
                capacityMin=None,
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )

    # Not error if capacity Fix is set, but it matches the stock decommissioning
    stockCommissioning = {
        1990: pd.Series(index=locations, data=[0, 1]),
        1995: pd.Series(index=locations, data=[0, 2]),
        2000: pd.Series(index=locations, data=[0, 3]),
        2005: pd.Series(index=locations, data=[0, 4]),
        2010: pd.Series(index=locations, data=[0, 5]),
    }
    capacityFix = {
        2020: pd.Series(index=locations, data=[0, 14]),
        2025: pd.Series(index=locations, data=[0, 12]),
        2030: pd.Series(index=locations, data=[0, 9]),
        2035: pd.Series(index=locations, data=[0, 5]),
        2040: pd.Series(index=locations, data=[0, 0]),
    }
    esM.add(
        fn.Storage(
            esM=esM,
            name="PressureTank_new",
            commodity="hydrogen",
            hasCapacityVariable=True,
            capacityVariableDomain="continuous",
            stockCommissioning=stockCommissioning,
            capacityFix=capacityFix,
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    # error for a decreasing capacityFix which cannot be fulfilled by the technicalLifetime
    with pytest.raises(ValueError, match=r".*Decreasing capacity fix set.*"):
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank",
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                capacityFix={2020: 5, 2025: 2, 2030: 1, 2035: 0, 2040: 0},
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )

    # implement error for mismatch for decommissioning of stock with capacityFix
    with pytest.raises(ValueError, match=r".*exceeds its capacityFix of.*"):
        stockCommissioning = {
            1990: pd.Series(index=locations, data=[0, 1]),
            1995: pd.Series(index=locations, data=[0, 2]),
            2000: pd.Series(index=locations, data=[0, 3]),
            2005: pd.Series(index=locations, data=[0, 4]),
            2010: pd.Series(index=locations, data=[0, 5]),
        }
        esM.add(
            fn.Storage(
                esM=esM,
                name="PressureTank",
                commodity="hydrogen",
                hasCapacityVariable=True,
                capacityVariableDomain="continuous",
                stockCommissioning=stockCommissioning,
                capacityFix={2020: 10, 2025: 12, 2030: 9, 2035: 5, 2040: 0},
                stateOfChargeMin=0.33,
                investPerCapacity=0.5,  # eur/kWh
                interestRate=0.08,
                economicLifetime=30,
            )
        )
