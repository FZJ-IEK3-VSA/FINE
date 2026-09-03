import fine as fn
import pandas as pd
import numpy as np
import pytest


def _create_delayed_transmission_esM(timeDelay=1, losses=0):
    """Create a two-location system whose demand can only be met via transmission."""
    esM = fn.EnergySystemModel(
        locations={"A", "B"},
        commodities={"commodity"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"commodity": "unit"},
        hoursPerTimeStep=1,
        costUnit="cost_unit",
        lengthUnit="length_unit",
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="supply",
            commodity="commodity",
            hasCapacityVariable=False,
            operationRateMax=pd.DataFrame({"A": [10, 0, 0, 0], "B": [0, 0, 0, 0]}),
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="demand",
            commodity="commodity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame(
                {"A": [0, 0, 0, 0], "B": [0, 10 * (1 - losses), 0, 0]}
            ),
        )
    )
    eligibility = pd.DataFrame([[0, 1], [1, 0]], index=["A", "B"], columns=["A", "B"])
    esM.add(
        fn.Transmission(
            esM=esM,
            name="shipment",
            commodity="commodity",
            losses=losses,
            distances=1,
            timeDelay=timeDelay,
            hasCapacityVariable=False,
            locationalEligibility=eligibility,
        )
    )
    return esM


def test_transmission_time_delay_shifts_arrival_and_applies_losses():
    esM = _create_delayed_transmission_esM(losses=0.1)
    esM.optimize(solver="glpk")

    operation = esM.componentModelingDict[
        "TransmissionModel"
    ].operationVariablesOptimum[esM.investmentPeriodNames[0]]
    assert operation.loc[("shipment", "A", "B"), 0] == pytest.approx(10)
    assert operation.loc[("shipment", "A", "B"), 1] == pytest.approx(0)


def test_transmission_time_delay_closes_horizon():
    esM = _create_delayed_transmission_esM()
    esM.declareOptimizationProblem()

    constraint = esM.pyM.ConstrTimeDelay_trans["A_B", "shipment", 0, 0, 3]
    assert constraint.lower() == 0
    assert constraint.upper() == 0


def test_transmission_time_delay_accepts_connection_specific_dataframe():
    timeDelay = pd.DataFrame([[0, 1], [2, 0]], index=["A", "B"], columns=["A", "B"])
    esM = _create_delayed_transmission_esM(timeDelay=timeDelay)
    transmission = esM.getComponent("shipment")

    assert transmission.timeDelay["A_B"] == 2
    assert transmission.timeDelay["B_A"] == 1

    esM.declareOptimizationProblem()
    assert esM.pyM.ConstrTimeDelay_trans["A_B", "shipment", 0, 0, 2].upper() == 0
    assert esM.pyM.ConstrTimeDelay_trans["B_A", "shipment", 0, 0, 3].upper() == 0


@pytest.mark.parametrize(
    "timeDelay, exception",
    [
        (-1, ValueError),
        (0.5, TypeError),
        (4, ValueError),
        (
            pd.DataFrame([[0, 1.5], [1, 0]], index=["A", "B"], columns=["A", "B"]),
            ValueError,
        ),
    ],
)
def test_transmission_time_delay_is_validated(timeDelay, exception):
    with pytest.raises(exception, match="timeDelay"):
        _create_delayed_transmission_esM(timeDelay=timeDelay)


def test_transmission_time_delay_rejects_time_series_aggregation():
    transmission = _create_delayed_transmission_esM().getComponent("shipment")
    with pytest.raises(ValueError, match="timeDelay.*time series aggregation"):
        transmission.setTimeSeriesData(hasTSA=True)


def test_initializeTransmission():
    """Tests if Transmission components are initialized without error if
    just required parameters are given.
    """
    # Define general parameters for esM-instance
    locations = ["cluster_1", "cluster_2", "cluster_3", "cluster_4"]
    commodityUnitDict = {"commodity1": "commodity_unit"}
    commodities = {"commodity1"}

    # Initialize esM-instance
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities=commodities,
        numberOfTimeSteps=4,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="cost_unit",
        lengthUnit="length_unit",
    )

    # Initialize Transmission
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Transmission_1",
            commodity="commodity1",
            hasCapacityVariable=True,
        )
    )


def test_initializeTransmission_withDataFrame():
    """Tests if Transmission components are initialized without error if
    additional parameters are given as DataFrame.
    """
    # Define general parameters for esM-instance
    locations = ["cluster_1", "cluster_2", "cluster_3", "cluster_4"]
    commodityUnitDict = {"commodity1": "commodity_unit"}
    commodities = {"commodity1"}

    # Initialize esM-instance
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities=commodities,
        numberOfTimeSteps=4,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="cost_unit",
        lengthUnit="length_unit",
    )

    # Set locationalEligibility, capacityMin, capacityMax, opexPerOperation and opexPerCapacity as DataFrame
    elig_data = np.array([[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])

    elig_df = pd.DataFrame(elig_data, index=locations, columns=locations)

    capMin_df = elig_df * 2
    capMax_df = elig_df * 3

    opexPerOp_df = elig_df * 0.02
    opexPerOp_df.loc["cluster_1", "cluster_2"] = 0.03

    opexPerCap_df = elig_df * 0.1

    # Initialize Transmission
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Transmission_1",
            commodity="commodity1",
            hasCapacityVariable=True,
            locationalEligibility=elig_df,
            capacityMax=capMax_df,
            capacityMin=capMin_df,
            opexPerOperation=opexPerOp_df,
            opexPerCapacity=opexPerCap_df,
        )
    )


def test_initializeTransmission_withFloat():
    """Tests if Transmission components are initialized without error if
    additional parameters are given as float.
    """
    # Define general parameters for esM-instance
    locations = ["cluster_1", "cluster_2", "cluster_3", "cluster_4"]
    commodityUnitDict = {"commodity1": "commodity_unit"}
    commodities = {"commodity1"}

    # Initialize esM-instance
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities=commodities,
        numberOfTimeSteps=4,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="cost_unit",
        lengthUnit="length_unit",
    )

    # Set capacityMin, capacityMax, opexPerOperation and opexPerCapacity as float
    capMin = 2
    capMax = 3

    opexPerOp = 0.02
    opexPerCap = 0.1

    # Initialize Transmission
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Transmission_1",
            commodity="commodity1",
            hasCapacityVariable=True,
            capacityMax=capMax,
            capacityMin=capMin,
            opexPerOperation=opexPerOp,
            opexPerCapacity=opexPerCap,
        )
    )


def test_initializeTransmission_withSeries():
    """Tests if Transmission components are initialized without error if
    additional parameters are given as data series.
    """
    # Define general parameters for esM-instance
    locations = ["cluster_1", "cluster_2", "cluster_3", "cluster_4"]
    commodityUnitDict = {"commodity1": "commodity_unit"}
    commodities = {"commodity1"}

    # Initialize esM-instance
    esM = fn.EnergySystemModel(
        locations=set(locations),
        commodities=commodities,
        numberOfTimeSteps=4,
        commodityUnitsDict=commodityUnitDict,
        hoursPerTimeStep=1,
        costUnit="cost_unit",
        lengthUnit="length_unit",
    )

    # Set capacityMin, capacityMax, opexPerOperation and opexPerCapacity as float
    idx = [
        "cluster_1_cluster_2",
        "cluster_1_cluster_3",
        "cluster_1_cluster_4",
        "cluster_2_cluster_1",
        "cluster_2_cluster_3",
        "cluster_2_cluster_4",
        "cluster_3_cluster_1",
        "cluster_3_cluster_2",
        "cluster_3_cluster_4",
        "cluster_4_cluster_1",
        "cluster_4_cluster_2",
        "cluster_4_cluster_3",
    ]
    capMax = pd.Series([2, 3, 3, 4, 5, 6, 2, 3, 3, 1, 6, 4], index=idx)
    opexPerOp = capMax * 0.02

    # Initialize Transmission
    esM.add(
        fn.Transmission(
            esM=esM,
            name="Transmission_1",
            commodity="commodity1",
            hasCapacityVariable=True,
            capacityMin=capMax,
            opexPerOperation=opexPerOp,
        )
    )
