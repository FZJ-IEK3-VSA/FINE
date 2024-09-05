import pandas as pd
from fine import utils
import fine as fn
import numpy as np
import pytest


def test_checkSimultaneousChargeDischarge():
    """
    Test a minimal example, with two regions and 10 days, where simultaneous charge and discharge occurs.
    """
    locations = {"Region1", "Region2"}
    commodityUnitDict = {"electricity": r"MW$_{el}$"}
    commodities = {"electricity"}
    ndays = 10
    nhours = 24 * ndays
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=nhours,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e6 Euro",
        lengthUnit="km",
        verboseLogLevel=1,
    )
    # Create synthetic daily demand profile
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
        [[u * 40, u * 60] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    ).round(2)
    esM.add(
        fn.Sink(
            esM=esM,
            name="Electricity demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    # Add storage 'Batteries'
    chargeEfficiency, dischargeEfficiency, selfDischarge = (
        0.95,
        0.95,
        1 - (1 - 0.03) ** (1 / (30 * 24)),
    )
    chargeRate, dischargeRate = 1, 1
    investPerCapacity, opexPerCapacity = 1000, 0
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
    # Create synthetic profile for PV and add PV with fixed operationRate. Therefore, it cannot be curtailed.
    # To achieve a curtailment, the system 'burns' energy by charging and discharging the storage simultaneously.
    dailyProfileSimple = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.05,
        0.15,
        0.2,
        0.4,
        0.8,
        0.7,
        0.4,
        0.2,
        0.15,
        0.05,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    operationRateFix = pd.DataFrame(
        [[u, u] for day in range(ndays) for u in dailyProfileSimple],
        index=range(nhours),
        columns=["Region1", "Region2"],
    )
    capacityMax = pd.Series([10000, 10000], index=["Region1", "Region2"])
    investPerCapacity, opexPerCapacity = 100, 10
    interestRate, economicLifetime = 0.08, 25
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            operationRateFix=operationRateFix,
            capacityFix=capacityMax,
            investPerCapacity=investPerCapacity,
            opexPerCapacity=opexPerCapacity,
            interestRate=interestRate,
            economicLifetime=economicLifetime,
        )
    )
    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    # Get the charge and discharge time series of the Batteries and use the check in the utils.
    tsCharge = esM.componentModelingDict[
        "StorageModel"
    ].chargeOperationVariablesOptimum.loc["Batteries"]
    tsDischarge = esM.componentModelingDict[
        "StorageModel"
    ].dischargeOperationVariablesOptimum.loc["Batteries"]
    simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
        tsCharge, tsDischarge
    )

    assert (
        simultaneousChargeDischarge
    ), "Check for simultaneous charge & discharge should have returned True"


def test_functionality_checkSimultaneousChargeDischarge():
    """
    Simple functionality test for utils.checkSimultaneousChargeDischarge
    """
    # Define charge and discharge time series for one region
    tsCharge = pd.DataFrame(columns=["Region1"])
    tsCharge["Region1"] = 3 * [1] + 1 * [0]
    tsDischarge = pd.DataFrame(columns=["Region1"])
    tsDischarge["Region1"] = 2 * [0] + 2 * [1]
    simultaneousChargeDischarge = utils.checkSimultaneousChargeDischarge(
        tsCharge, tsDischarge
    )

    assert (
        simultaneousChargeDischarge
    ), "Check for simultaneous charge & discharge should have returned True"



def test_check_and_set_cost_parameter():
    numberOfTimeSteps = 4
    hoursPerTimeStep = 2190
    # Create an energy system model instance
    esM = fn.EnergySystemModel(
        locations={"ElectrolyzerLocation"},
        commodities={"electricity", "hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={
            "electricity": r"kW$_{el}$",
            "hydrogen": r"kW$_{H_{2},LHV}$",
        },
        hoursPerTimeStep=hoursPerTimeStep,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )
 
    # Test with valid integer data (1dim)
    assert utils.checkAndSetCostParameter(esM, "testParam", 10, "1dim", None).equals(
        pd.Series([10.0], index=esM.locations)
    )
 
    # Test with valid series data (1dim)
    valid_series_1dim = pd.Series([10], index=esM.locations)
    assert utils.checkAndSetCostParameter(esM, "testParam", valid_series_1dim, "1dim", None).equals(
        valid_series_1dim.astype(float)
    )
 
    # Test with NaN in integer data (1dim)
    with pytest.raises(AssertionError):
        assert utils.checkAndSetCostParameter(esM, "testParam", np.nan, "1dim", None).equals(
            pd.Series([np.nan], index=esM.locations)
        )
 
    # Test with NaN in series data (2dim)
    with pytest.raises(AssertionError):
        invalid_series_with_nan = pd.Series([10, np.nan], index=["loc1", "loc2"])
        assert utils.checkAndSetCostParameter(esM, "testParam", invalid_series_with_nan, "2dim", None).equals(
            invalid_series_with_nan, index=esM.locations)