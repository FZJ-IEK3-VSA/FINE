import FINE as fn
import pandas as pd
import numpy as np


def test_initializeTransmission():
    """
    Tests if Transmission components are initialized without error if
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
    """
    Tests if Transmission components are initialized without error if
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
    """
    Tests if Transmission components are initialized without error if
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
    """
    Tests if Transmission components are initialized without error if
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
