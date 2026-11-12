import numpy as np
import pandas as pd
import fine as fn
import fine.IOManagement.xarrayIO as xrIO
import pyomo.environ as pyomo

# first solver that is installed; the branch tests otherwise hard-code glpk
SOLVER = next(
    s
    for s in ("glpk", "gurobi", "highs")
    if pyomo.SolverFactory(s).available(exception_flag=False)
)


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
        solver=SOLVER,
    )

    sourceSinkModel = esM.componentModelingDict["SourceSinkModel"]
    purchase = sourceSinkModel.operationVariablesOptimum.xs("Purchase")

    # the combined demand of B and C is covered time step by time step by the
    # purchase at A via the pooled balance
    np.testing.assert_array_almost_equal(
        purchase.loc["A"].values, (demand["B"] + demand["C"]).values
    )


def test_pooledCommodities_netcdf_roundtrip(tmp_path):
    """The nested pooledCommodities dict survives writing to and reading from
    netCDF, so a saved model with pooled commodities can be rebuilt.
    """
    pooled = {"hydrogen": {"pool1": ["A", "B"], "pool2": ["C"]}}
    esM = fn.EnergySystemModel(
        locations={"A", "B", "C"},
        commodities={"hydrogen"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"hydrogen": "GW"},
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
        pooledCommodities=pooled,
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                {"A": [1.0] * 4, "B": [1.0] * 4, "C": [0.0] * 4}
            ),
        )
    )
    path = tmp_path / "pooled.nc"
    xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath=str(path))
    esM_read = xrIO.readNetCDFtoEnergySystemModel(filePath=str(path))
    assert esM_read.pooledCommodities == pooled


def test_pooledCommodities_with_componentLimit():
    """A componentLimit on a source of a pooled commodity still binds.

    Hydrogen is pooled over A, B and C. A cheap source at A and an expensive
    source at C can supply the demand at B. A componentLimit (operation, upper)
    caps the cheap source at half of the total demand, so the expensive source
    at C must deliver the rest through the pool.
    """
    locations = {"A", "B", "C"}
    n = 10
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"hydrogen"},
        numberOfTimeSteps=n,
        commodityUnitsDict={"hydrogen": "GW"},
        hoursPerTimeStep=1,
        costUnit="EUR",
        lengthUnit="km",
        pooledCommodities={"hydrogen": {"pool1": ["A", "B", "C"]}},
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Cheap",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost=0.01,
            locationalEligibility=pd.Series({"A": 1, "B": 0, "C": 0}),
            componentLimitID="LIM_cheap",
        )
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="Expensive",
            commodity="hydrogen",
            hasCapacityVariable=False,
            commodityCost=1.0,
            locationalEligibility=pd.Series({"A": 0, "B": 0, "C": 1}),
        )
    )
    demand = pd.DataFrame({"A": [0.0] * n, "B": [10.0] * n, "C": [0.0] * n})
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="hydrogen",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )
    cap = 0.5 * demand.values.sum()
    esM.processedComponentLimit = pd.DataFrame(
        columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
        index=["LIM_cheap"],
        data=[[cap, "upper", "operation", None, 0, None]],
    )
    esM.processedComponentLimitEligibility = pd.DataFrame(
        {"LIM_cheap": [1, 1, 1]}, index=sorted(locations)
    )

    esM.optimize(timeSeriesAggregation=False, solver=SOLVER)

    op = esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum
    cheap = op.loc[("Cheap", "A")].sum()
    expensive = op.loc[("Expensive", "C")].sum()
    np.testing.assert_almost_equal(cheap, cap)
    np.testing.assert_almost_equal(cheap + expensive, demand.values.sum())
