import fine as fn
import numpy as np
import pandas as pd
import pytest


def _create_esM():
    """Two-node system: cheap source only in A, expensive source only in B.

    Demand in B (4 per hour) is larger than demand in A (1 per hour), so the
    optimal transmission capacity from A to B is 4, while no capacity from B to A
    is needed.
    """
    locations = {"A", "B"}
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"electricity": "GW_el"},
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="cheap source",
            commodity="electricity",
            hasCapacityVariable=False,
            locationalEligibility=pd.Series([1, 0], index=["A", "B"]),
            opexPerOperation=0.01,
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="expensive source",
            commodity="electricity",
            hasCapacityVariable=False,
            locationalEligibility=pd.Series([0, 1], index=["A", "B"]),
            opexPerOperation=10,
        )
    )
    demand = pd.DataFrame(
        {"A": [1.0] * 4, "B": [4.0] * 4}, index=range(esM.numberOfTimeSteps)
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    return esM


def _distances():
    return pd.DataFrame([[0, 100], [100, 0]], index=["A", "B"], columns=["A", "B"])


def _getCapacities(esM, modelName, compName):
    caps = esM.componentModelingDict[modelName].getOptimalValues(
        "capacityVariablesOptimum"
    )["values"]
    return caps.loc[compName]


def test_asymmetricTransmission_importable():
    assert issubclass(fn.AsymmetricTransmission, fn.Transmission)


def test_asymmetricTransmission_capacitiesPerDirection():
    """The capacity of each direction is optimized independently."""
    esM = _create_esM()
    esM.add(
        fn.AsymmetricTransmission(
            esM=esM,
            name="line",
            commodity="electricity",
            distances=_distances(),
            investPerCapacity=0.001,
        )
    )
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    caps = _getCapacities(esM, "AsymmetricTransmissionModel", "line")
    assert caps.loc["A", "B"] == pytest.approx(4)
    assert caps.loc["B", "A"] == pytest.approx(0)

    # The symmetric Transmission component has to build the same capacity in both directions
    esM_sym = _create_esM()
    esM_sym.add(
        fn.Transmission(
            esM=esM_sym,
            name="line",
            commodity="electricity",
            distances=_distances(),
            investPerCapacity=0.001,
        )
    )
    esM_sym.optimize(timeSeriesAggregation=False, solver="glpk")
    caps_sym = _getCapacities(esM_sym, "TransmissionModel", "line")
    assert caps_sym.loc["A", "B"] == pytest.approx(4)
    assert caps_sym.loc["B", "A"] == pytest.approx(4)

    # Half of the symmetric investment is saved
    assert esM.pyM.Obj() < esM_sym.pyM.Obj()


def test_asymmetricTransmission_directionalCapacityMax():
    """capacityMax[loc1][loc2] limits the flow from loc1 to loc2 only."""
    esM = _create_esM()
    capacityMax = pd.DataFrame(
        [[0, 1], [3, 0]], index=["A", "B"], columns=["A", "B"]
    )  # column A, row B -> A_B = 3 ; column B, row A -> B_A = 1
    esM.add(
        fn.AsymmetricTransmission(
            esM=esM,
            name="line",
            commodity="electricity",
            distances=_distances(),
            capacityMax=capacityMax,
            investPerCapacity=0.001,
        )
    )
    esM.optimize(timeSeriesAggregation=False, solver="glpk")

    caps = _getCapacities(esM, "AsymmetricTransmissionModel", "line")
    assert caps.loc["A", "B"] == pytest.approx(3)
    assert caps.loc["B", "A"] == pytest.approx(0)

    op = esM.componentModelingDict["AsymmetricTransmissionModel"].getOptimalValues(
        "operationVariablesOptimum"
    )["values"]
    assert np.allclose(op.loc[("line", "A", "B")].values, 3)
    # remaining demand in B is covered by the expensive source
    src = esM.componentModelingDict["SourceSinkModel"].getOptimalValues(
        "operationVariablesOptimum"
    )["values"]
    assert np.allclose(src.loc[("expensive source", "B")].values, 1)


def test_asymmetricTransmission_oneDirectionOnly():
    """A connection can be eligible in one direction only."""
    esM = _create_esM()
    eligibility = pd.DataFrame([[0, 0], [1, 0]], index=["A", "B"], columns=["A", "B"])
    esM.add(
        fn.AsymmetricTransmission(
            esM=esM,
            name="line",
            commodity="electricity",
            distances=_distances(),
            locationalEligibility=eligibility,
            investPerCapacity=0.001,
        )
    )
    comp = esM.getComponent("line")
    assert list(comp.locationalEligibility.index) == ["A_B"]

    esM.optimize(timeSeriesAggregation=False, solver="glpk")
    caps = _getCapacities(esM, "AsymmetricTransmissionModel", "line")
    assert caps.loc["A", "B"] == pytest.approx(4)


def test_asymmetricTransmission_componentLimitID():
    """The AsymmetricTransmission accepts a componentLimitID like Transmission."""
    esM = _create_esM()
    esM.add(
        fn.AsymmetricTransmission(
            esM=esM,
            name="line",
            commodity="electricity",
            distances=_distances(),
            componentLimitID="limit",
        )
    )
    assert esM.getComponent("line").componentLimitID == "limit"


def test_asymmetricTransmission_dictIO_roundtrip():
    esM = _create_esM()
    capacityMax = pd.DataFrame([[0, 1], [3, 0]], index=["A", "B"], columns=["A", "B"])
    esM.add(
        fn.AsymmetricTransmission(
            esM=esM,
            name="line",
            commodity="electricity",
            distances=_distances(),
            capacityMax=capacityMax,
            investPerCapacity=0.001,
        )
    )
    esm_dict, comp_dict = fn.dictIO.exportToDict(esM)
    esM_new = fn.dictIO.importFromDict(esm_dict, comp_dict)

    comp_new = esM_new.getComponent("line")
    assert isinstance(comp_new, fn.AsymmetricTransmission)
    pd.testing.assert_series_equal(
        comp_new.processedCapacityMax[0].sort_index(),
        esM.getComponent("line").processedCapacityMax[0].sort_index(),
        check_names=False,
    )
