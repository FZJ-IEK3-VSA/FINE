import pandas as pd
import os
from pathlib import Path


def getData(engine="openpyxl"):
    current_directory = (
        Path(__file__).parents[1].joinpath("01_1node_Energy_System_Workflow").absolute()
    )
    inputDataPath = os.path.join(current_directory, "InputData")
    data = {}

    # Onshore data
    capacityMax = pd.read_excel(
        os.path.join(
            inputDataPath, "SpatialData", "Wind", "maxCapacityOnshore_GW_el.xlsx"
        ),
        index_col=0,
        engine=engine,
    ).squeeze("columns")
    operationRateMax = pd.read_excel(
        os.path.join(
            inputDataPath, "SpatialData", "Wind", "maxOperationRateOnshore_el.xlsx"
        ),
        engine=engine,
    )

    data.update({"Wind (onshore), capacityMax": capacityMax.loc["cluster_0"]})
    data.update(
        {"Wind (onshore), operationRateMax": operationRateMax.loc[:, "cluster_0"]}
    )

    # Hydrogen salt cavern data
    capacityMax = (
        pd.read_excel(
            os.path.join(
                inputDataPath,
                "SpatialData",
                "GeologicalStorage",
                "existingSaltCavernsCapacity_GWh_methane.xlsx",
            ),
            index_col=0,
            engine=engine,
        ).squeeze("columns")
        * 3
        / 10
    )

    data.update({"Salt caverns (hydrogen), capacityMax": capacityMax.loc["cluster_0"]})

    # Electricity demand data
    operationRateFix = pd.read_excel(
        os.path.join(
            inputDataPath, "SpatialData", "Demands", "electricityDemand_GWh_el.xlsx"
        ),
        engine=engine,
    )

    data.update(
        {"Electricity demand, operationRateFix": operationRateFix.loc[:, "cluster_0"]}
    )

    # Hydrogen demand data
    operationRateFix = pd.read_excel(
        os.path.join(
            inputDataPath, "SpatialData", "Demands", "hydrogenDemand_GWh_hydrogen.xlsx"
        ),
        engine=engine,
    )

    data.update(
        {"Hydrogen demand, operationRateFix": operationRateFix.loc[:, "cluster_0"]}
    )

    return data
