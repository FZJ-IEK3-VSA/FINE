"""Tests for `fine.utils.runInfeasibilityPrechecks`.

The energy system model built in `_build_infeasible_esM` is taken from the
example notebook that was used to manually validate the infeasibility
pre-checks. It combines a capacity-limited Wind source, an Electrolyzer,
a battery storage, electricity/hydrogen transmission between three regions
and a fixed hydrogen demand that spikes in the last time step.

With the capacities as given in the notebook, the model is infeasible:
the Electrolyzer's electricity demand (needed to cover the hydrogen
demand) exceeds what the Wind source can supply, both over the whole
time horizon and in the last time step. `runInfeasibilityPrechecks`
is expected to catch this before the optimization problem is even
declared, via `checkJointInputDemandAggregated` and
`checkJointInputDemandPerTimeStep`.

`_build_feasible_esM` is the same model with the Wind, AC cable and
hydrogen pipeline capacities raised, which removes the shortage. It is
used as a control case to guard against false positives.
"""

import numpy as np
import pandas as pd
import pytest

import fine as fn
from fine.utils import runInfeasibilityPrechecks
from fine.utils import checkJointInputDemandAggregated



def _build_esm(windCapacityMax, transmissionCapacityMax):
    """Build the example ESM from the notebook with configurable capacities."""
    locations = {"Region1", "Region2", "Region3"}
    commodityUnitDict = {
        "electricity": "GW_el",
        "hydrogen": "GW_H2",
        "heat": "GW_th",
    }
    commodities = {"electricity", "hydrogen", "heat"}

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities=commodities,
        numberOfTimeSteps=4,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="1e9 Euro",
        lengthUnit="km",
        verboseLogLevel=0,
    )

    windOperationRateMax = pd.DataFrame(
        {
            "Region1": [1, 1, 1, 1],
            "Region2": [1, 1, 1, 1],
            "Region3": [1, 1, 0, 0],
        }
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Wind",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=windCapacityMax,
            operationRateMax=windOperationRateMax,
            investPerCapacity=2 * 2190,
            opexPerCapacity=0,
            interestRate=0,
            opexPerOperation=0,
            economicLifetime=1,
        )
    )

    esM.add(
        fn.Conversion(
            esM=esM,
            name="Electrolyzer",
            physicalUnit="GW_el",
            commodityConversionFactors={"electricity": -1, "hydrogen": 0.7},
            hasCapacityVariable=True,
            investPerCapacity=0.5,
            opexPerCapacity=0.5 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    esM.add(
        fn.Storage(
            esM=esM,
            name="Li-ion batteries",
            commodity="electricity",
            hasCapacityVariable=True,
            capacityMax=1,
            chargeEfficiency=0.95,
            cyclicLifetime=10000,
            dischargeEfficiency=0.95,
            selfDischarge=1 - (1 - 0.03) ** (1 / (30 * 24)),
            chargeRate=1,
            dischargeRate=1,
            doPreciseTsaModeling=False,
            investPerCapacity=0.151,
            opexPerCapacity=0.002,
            interestRate=0.08,
            economicLifetime=22,
        )
    )

    locs = sorted(locations)
    distances = pd.DataFrame(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), index=locs, columns=locs
    )
    eligibility = pd.DataFrame(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), index=locs, columns=locs
    )

    esM.add(
        fn.Transmission(
            esM=esM,
            name="AC cable",
            commodity="electricity",
            losses=0,
            distances=distances,
            hasCapacityVariable=True,
            capacityMax=transmissionCapacityMax,
            locationalEligibility=eligibility,
            investPerCapacity=0.1,
            interestRate=0.08,
            economicLifetime=50,
        )
    )

    esM.add(
        fn.Transmission(
            esM=esM,
            name="hydrogen pipeline",
            commodity="hydrogen",
            losses=0,
            distances=distances,
            capacityMax=transmissionCapacityMax,
            hasCapacityVariable=True,
            locationalEligibility=eligibility,
            investPerCapacity=0.1,
            interestRate=0.08,
            economicLifetime=50,
        )
    )

    demand = pd.DataFrame(
        {
            "Region1": [1, 1, 1, 1],
            "Region2": [0, 0, 0, 0],
            "Region3": [0, 0, 0, 4],
        }
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Hydrogen demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    return esM


@pytest.fixture
def infeasible_esM():
    """ESM from the notebook: Wind cannot cover the Electrolyzer's demand."""
    return _build_esm(windCapacityMax=1, transmissionCapacityMax=1)


@pytest.fixture
def feasible_esM():
    """Return the same ESM with enough Wind/transmission capacity to remove the shortage."""
    return _build_esm(windCapacityMax=5, transmissionCapacityMax=5)


def test_runInfeasibilityPrechecks_raises_for_infeasible_esM(infeasible_esM):
    """The notebook's example ESM must be caught before the solver runs."""
    with pytest.raises(ValueError) as excinfo:
        runInfeasibilityPrechecks(infeasible_esM)

    message = str(excinfo.value)
    assert "checkJointInputDemandAggregated" in message
    assert "checkJointInputDemandPerTimeStep" in message
    assert "electricity" in message


def test_runInfeasibilityPrechecks_raiseError_false_returns_problems(
    infeasible_esM,
):
    """With raiseError=False, problems are returned instead of raised."""
    problems = runInfeasibilityPrechecks(infeasible_esM, raiseError=False)

    assert problems, "expected at least one detected problem"
    assert any("checkJointInputDemandAggregated" in p for p in problems)
    assert any("checkJointInputDemandPerTimeStep" in p for p in problems)


def test_runInfeasibilityPrechecks_passes_for_feasible_esM(feasible_esM):
    """Control case: raising the capacities removes the infeasibility.

    Guards against false positives / an overly strict check.
    """
    problems = runInfeasibilityPrechecks(feasible_esM, raiseError=False)
    assert problems == []


def test_runInfeasibilityPrechecks_single_check_reports_the_shortage(
    infeasible_esM,
):
    """The aggregated joint-input-demand check alone must already flag the model."""

    problems = checkJointInputDemandAggregated(infeasible_esM)

    assert len(problems) == 1
    assert "electricity" in problems[0]
    assert "Region1" in problems[0] and "Region2" in problems[0] and "Region3" in problems[0]
