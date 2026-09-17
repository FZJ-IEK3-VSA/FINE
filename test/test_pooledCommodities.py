import numpy as np
import pandas as pd
import fine as fn

from fine.utils import ImplementedSolvers


def test_pooledCommodityBalanceConstraint():
    """Test that a pooled commodity is balanced across all locations of its
    pool instead of individually at each location.

    Hydrogen can only be purchased at location A, while demand occurs at
    locations B and C. Without pooling this would be infeasible since there is
    no Transmission component connecting the locations. With "hydrogen" pooled
    across A, B and C, the purchase at A covers the combined demand of B and C.
    """
    locations = {"A", "B", "C"}
    numberOfTimeSteps = 40

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"hydrogen"},
        numberOfTimeSteps=numberOfTimeSteps,
        commodityUnitsDict={"hydrogen": "GW"},
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
        pooledCommodities={"hydrogen": {"pool1": ["A", "B", "C"]}},
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Purchase",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost=0.01,
            locationalEligibility=pd.Series({"A": 1, "B": 0, "C": 0}),
        )
    )

    # varying demand profiles so the per-time-step pool balance is actually
    # exercised
    rng = np.random.default_rng(42)
    demandProfile_B = rng.uniform(1.0, 10.0, numberOfTimeSteps)
    demandProfile_C = rng.uniform(1.0, 15.0, numberOfTimeSteps)

    demand = pd.DataFrame(
        {
            "A": [0.0] * numberOfTimeSteps,
            "B": demandProfile_B,
            "C": demandProfile_C,
        }
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    sourceSinkModel = esM.componentModelingDict["SourceSinkModel"]
    purchase = sourceSinkModel.operationVariablesOptimum.xs("Purchase")

    # the combined demand of B and C is covered time step by time step by the
    # purchase at A via the pooled balance
    np.testing.assert_array_almost_equal(
        purchase.loc["A"].values, (demand["B"] + demand["C"]).values
    )
