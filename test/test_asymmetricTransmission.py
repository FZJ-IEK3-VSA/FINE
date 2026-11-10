import fine as fn
import pandas as pd
import pytest

LOCS = ["A", "B"]


def _build_esM(transmissionClass, capacityMax=None, balanceLimit=None):
    """Two nodes, two time steps.

    t=0: cheap supply in A, demand of 10 in B  -> flow A->B
    t=1: cheap supply in B, demand of 4 in A   -> flow B->A
    Expensive backup supply exists in both nodes.
    """
    esM = fn.EnergySystemModel(
        locations=set(LOCS),
        commodities={"electricity"},
        numberOfTimeSteps=2,
        commodityUnitsDict={"electricity": "GW"},
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=balanceLimit,
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="cheap",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.DataFrame({"A": [1000.0, 0.0], "B": [0.0, 1000.0]}),
            commodityCost=0.001,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="backup",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateMax=pd.DataFrame(
                {"A": [1000.0, 1000.0], "B": [1000.0, 1000.0]}
            ),
            commodityCost=1.0,
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame({"A": [0.0, 4.0], "B": [10.0, 0.0]}),
        )
    )
    distances = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]], index=LOCS, columns=LOCS)
    esM.add(
        transmissionClass(
            esM=esM,
            name="line",
            commodity="electricity",
            distances=distances,
            capacityMax=capacityMax,
            balanceLimitID="import" if balanceLimit is not None else None,
            investPerCapacity=0.001,
            economicLifetime=1,
            interestRate=0,
        )
    )
    return esM


def _capacities(esM, modelName):
    caps = (
        esM.componentModelingDict[modelName]
        .getOptimalValues("capacityVariablesOptimum")["values"]
        .loc["line"]
    )
    return caps.loc["A", "B"], caps.loc["B", "A"]


def test_transmission_is_symmetric():
    esM = _build_esM(fn.Transmission)
    esM.optimize(solver="glpk")
    capAB, capBA = _capacities(esM, "TransmissionModel")
    assert capAB == pytest.approx(10)
    assert capBA == pytest.approx(10)


def test_asymmetric_transmission_has_independent_capacities():
    esM = _build_esM(fn.AsymmetricTransmission)
    esM.optimize(solver="glpk")
    capAB, capBA = _capacities(esM, "AsymmetricTransmissionModel")
    assert capAB == pytest.approx(10)
    assert capBA == pytest.approx(4)


def test_asymmetric_transmission_capacityMax_per_direction():
    # data[loc1][loc2] is the connection loc1 -> loc2
    capacityMax = pd.DataFrame(
        [[0.0, 2.0], [100.0, 0.0]], index=LOCS, columns=LOCS
    )  # A->B: 100, B->A: 2
    esM = _build_esM(fn.AsymmetricTransmission, capacityMax=capacityMax)
    esM.optimize(solver="glpk")
    capAB, capBA = _capacities(esM, "AsymmetricTransmissionModel")
    assert capAB == pytest.approx(10)
    assert capBA == pytest.approx(2)

    flows = esM.componentModelingDict["AsymmetricTransmissionModel"].getOptimalValues(
        "operationVariablesOptimum"
    )["values"]
    assert flows.loc[("line", "B", "A"), 1] == pytest.approx(2)


def test_asymmetric_transmission_balanceLimit():
    # yearly net import into B (import A->B minus export B->A = 4) is limited to 3
    # -> the line A->B is only built with 7 (without the limit it would be 10)
    balanceLimit = pd.DataFrame({"A": [1e6], "B": [3.0]}, index=["import"])
    balanceLimit["lowerBound"] = False
    esM = _build_esM(fn.AsymmetricTransmission, balanceLimit=balanceLimit)
    esM.optimize(solver="glpk")
    capAB, capBA = _capacities(esM, "AsymmetricTransmissionModel")
    assert capAB == pytest.approx(7)
    assert capBA == pytest.approx(4)


def test_asymmetric_transmission_xarray_roundtrip():
    capacityMax = pd.DataFrame([[0.0, 2.0], [100.0, 0.0]], index=LOCS, columns=LOCS)
    esM = _build_esM(fn.AsymmetricTransmission, capacityMax=capacityMax)
    datasets = fn.xarrayIO.convertOptimizationInputToDatasets(esM)
    esM_new = fn.xarrayIO.convertDatasetsToEnergySystemModel(datasets)

    assert isinstance(esM_new.getComponent("line"), fn.AsymmetricTransmission)
    esM_new.optimize(solver="glpk")
    capAB, capBA = _capacities(esM_new, "AsymmetricTransmissionModel")
    assert capAB == pytest.approx(10)
    assert capBA == pytest.approx(2)


def test_asymmetric_transmission_updateComponent():
    esM = _build_esM(fn.AsymmetricTransmission)
    capacityMax = pd.DataFrame([[0.0, 3.0], [100.0, 0.0]], index=LOCS, columns=LOCS)
    esM.updateComponent("line", {"capacityMax": capacityMax})

    assert isinstance(esM.getComponent("line"), fn.AsymmetricTransmission)
    esM.optimize(solver="glpk")
    capAB, capBA = _capacities(esM, "AsymmetricTransmissionModel")
    assert capAB == pytest.approx(10)
    assert capBA == pytest.approx(3)
