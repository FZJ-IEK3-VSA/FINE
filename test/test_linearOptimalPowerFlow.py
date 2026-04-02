import fine as fn
import pandas as pd
import numpy as np


def test_lopf_full_workflow():
    locations = {"cluster_1", "cluster_2"}
    loc_list = ["cluster_1", "cluster_2"]

    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=1,
        commodityUnitsDict={"electricity": "GW"},
        hoursPerTimeStep=1,
        verboseLogLevel=0,
    )

    # Source only in cluster_1
    operation_rate_max = pd.DataFrame(
        {"cluster_1": [1000.0], "cluster_2": [0.0]}
    )

    esM.add(
        fn.Source(
            esM=esM,
            name="Source",
            commodity="electricity",
            hasCapacityVariable=False,
            commodityCost=0.1,
            operationRateMax=operation_rate_max,
        )
    )

    # Demand only in cluster_2
    demand = pd.DataFrame(
        {"cluster_1": [0.0], "cluster_2": [100.0]}
    )

    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=demand,
        )
    )

    distances = pd.DataFrame(
        [[0.0, 100.0], [100.0, 0.0]],
        index=loc_list,
        columns=loc_list,
    )

    reactances = pd.DataFrame(
        [[0.0, 0.1], [0.1, 0.0]],
        index=loc_list,
        columns=loc_list,
    )

    esM.add(
        fn.LinearOptimalPowerFlow(
            esM=esM,
            name="DC cables",
            commodity="electricity",
            distances=distances,
            reactances=reactances,
            hasCapacityVariable=False,
        )
    )

    esM.optimize()

    # 1) Check modeling class exists
    assert "LOPFModel" in esM.componentModelingDict
    lopf_model = esM.componentModelingDict["LOPFModel"]

    # 2) Check Pyomo objects exist
    pyM = esM.pyM
    assert hasattr(pyM, "phaseAngleVarSet_lopf")
    assert hasattr(pyM, "phaseAngle_lopf")
    assert hasattr(pyM, "ConstrpowerFlowDC_lopf")
    assert hasattr(pyM, "ConstrBasePhaseAngle_lopf")

    # 3) Check optimization results
    angles = lopf_model.getOptimalValues("phaseAngleVariablesOptimum")["values"]
    flows = lopf_model.getOptimalValues("operationVariablesOptimum")["values"]

    # Angle difference should be 10.0 (since reactance is 0.1 and flow is 100.0)
    assert abs(angles.loc[("DC cables", "cluster_1"), 0] - 0.0) < 1e-6
    assert abs(angles.loc[("DC cables", "cluster_2"), 0] - (-10.0)) < 1e-6

    # Flow should be 100.0 from cluster_1 to cluster_2 and 0.0 in the reverse direction
    assert abs(flows.loc[("DC cables", "cluster_1", "cluster_2"), 0] - 100.0) < 1e-6
    assert abs(flows.loc[("DC cables", "cluster_2", "cluster_1"), 0] - 0.0) < 1e-6

    # 4) Check reference node angle is zero
    # In the current implementation the alphabetically first node is the reference node.
    assert abs(angles.loc[("DC cables", "cluster_1"), 0]) < 1e-6

test_lopf_full_workflow()