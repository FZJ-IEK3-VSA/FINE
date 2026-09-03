"""Shared input data for the FINE examples.

Several examples use the same input data. The data is stored one time in
``examples/data``. Each example imports the loader from this module through its
own ``getData.py``.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.absolute() / "data"

#: Input data of the multi-region examples (03, 07 and 11).
MULTI_REGION_DATA_PATH = DATA_PATH / "multiRegion"

#: Input data of the one-node examples (01, 09 and 10).
ONE_NODE_DATA_PATH = DATA_PATH / "oneNode"


def getMultiRegionData(engine="openpyxl"):
    """Get the input data of the multi-region examples."""
    inputDataPath = MULTI_REGION_DATA_PATH
    data = {}

    # Onshore data
    capacityMax = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "Wind" / "maxCapacityOnshore_GW_el.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")
    operationRateMax = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Wind"
        / "maxOperationRateOnshore_el.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Wind (onshore), capacityMax": capacityMax})
    data.update({"Wind (onshore), operationRateMax": operationRateMax})

    # Offshore data
    capacityMax = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "Wind" / "maxCapacityOffshore_GW_el.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")
    operationRateMax = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Wind"
        / "maxOperationRateOffshore_el.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Wind (offshore), capacityMax": capacityMax})
    data.update({"Wind (offshore), operationRateMax": operationRateMax})

    # PV data
    capacityMax = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "PV" / "maxCapacityPV_GW_el.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")
    operationRateMax = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "PV" / "maxOperationRatePV_el.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"PV, capacityMax": capacityMax})
    data.update({"PV, operationRateMax": operationRateMax})

    # Run of river data
    capacityFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "HydroPower"
        / "fixCapacityROR_GW_el.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")
    operationRateFix = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "HydroPower" / "fixOperationRateROR.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Existing run-of-river plants, capacityFix": capacityFix})
    data.update({"Existing run-of-river plants, operationRateFix": operationRateFix})

    # Biogas data
    operationRateMax = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Biogas"
        / "biogasPotential_GWh_biogas.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Biogas, operationRateMax": operationRateMax})

    biogasCommodityCostTimeSeries = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Biogas"
        / "biogasPriceTimeSeries_MrdEuro_GWh.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Biogas, commodityCostTimeSeries": biogasCommodityCostTimeSeries})

    # Natural gas data
    naturalGasCommodityCostTimeSeries = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "NaturalGas"
        / "naturalGasPriceTimeSeries_MrdEuro_GWh.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update(
        {"Natural Gas, commodityCostTimeSeries": naturalGasCommodityCostTimeSeries}
    )

    # Natural gas plant data
    capacityMax = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "NaturalGasPlants"
        / "existingCombinedCycleGasTurbinePlantsCapacity_GW_el.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")

    data.update({"Existing CCGT plants (methane), capacityMax": capacityMax})

    # Hydrogen salt cavern data
    capacityMax = (
        pd.read_excel(
            Path(inputDataPath)
            / "SpatialData"
            / "GeologicalStorage"
            / "existingSaltCavernsCapacity_GWh_methane.xlsx",
            index_col=0,
            engine=engine,
        ).squeeze("columns")
        * 3
        / 10
    )

    data.update({"Salt caverns (hydrogen), capacityMax": capacityMax})

    # Methane salt cavern data
    capacityMax = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "GeologicalStorage"
        / "existingSaltCavernsCapacity_GWh_methane.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")

    data.update({"Salt caverns (methane), capacityMax": capacityMax})

    # Pumped hydro storage data
    capacityFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "HydroPower"
        / "fixCapacityPHS_storage_GWh_energyPHS.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")

    data.update({"Pumped hydro storage, capacityFix": capacityFix})

    # AC cables data
    capacityFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "ElectricGrid"
        / "ACcableExistingCapacity_GW_el.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )

    data.update({"AC cables, capacityFix": capacityFix})

    reactances = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "ElectricGrid" / "ACcableReactance.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )

    data.update({"AC cables, reactances": reactances})

    # DC cables data
    capacityFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "ElectricGrid"
        / "DCcableExistingCapacity_GW_el.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )
    distances = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "ElectricGrid" / "DCcableLength_km.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )
    losses = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "ElectricGrid" / "DCcableLosses.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )

    data.update({"DC cables, capacityFix": capacityFix})
    data.update({"DC cables, distances": distances})
    data.update({"DC cables, losses": losses})

    # Pipelines data
    eligibility = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "Pipelines" / "pipelineIncidence.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )
    distances = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "Pipelines" / "pipelineLength.xlsx",
        index_col=0,
        header=0,
        engine=engine,
    )

    data.update({"Pipelines, eligibility": eligibility})
    data.update({"Pipelines, distances": distances})

    # Electricity demand data
    operationRateFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Demands"
        / "electricityDemand_GWh_el.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Electricity demand, operationRateFix": operationRateFix})

    # Hydrogen demand data
    operationRateFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Demands"
        / "hydrogenDemand_GWh_hydrogen.xlsx",
        header=0,
        index_col=0,
        engine=engine,
    )

    data.update({"Hydrogen demand, operationRateFix": operationRateFix})

    return data


def getOneNodeData(engine="openpyxl"):
    """Get the input data of the one-node examples."""
    inputDataPath = ONE_NODE_DATA_PATH
    data = {}

    # Onshore data
    capacityMax = pd.read_excel(
        Path(inputDataPath) / "SpatialData" / "Wind" / "maxCapacityOnshore_GW_el.xlsx",
        index_col=0,
        engine=engine,
    ).squeeze("columns")
    operationRateMax = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Wind"
        / "maxOperationRateOnshore_el.xlsx",
        engine=engine,
    )

    data.update({"Wind (onshore), capacityMax": capacityMax.loc["cluster_0"]})
    data.update(
        {"Wind (onshore), operationRateMax": operationRateMax.loc[:, "cluster_0"]}
    )

    # Hydrogen salt cavern data
    capacityMax = (
        pd.read_excel(
            Path(inputDataPath)
            / "SpatialData"
            / "GeologicalStorage"
            / "existingSaltCavernsCapacity_GWh_methane.xlsx",
            index_col=0,
            engine=engine,
        ).squeeze("columns")
        * 3
        / 10
    )

    data.update({"Salt caverns (hydrogen), capacityMax": capacityMax.loc["cluster_0"]})

    # Electricity demand data
    operationRateFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Demands"
        / "electricityDemand_GWh_el.xlsx",
        engine=engine,
    )

    data.update(
        {"Electricity demand, operationRateFix": operationRateFix.loc[:, "cluster_0"]}
    )

    # Hydrogen demand data
    operationRateFix = pd.read_excel(
        Path(inputDataPath)
        / "SpatialData"
        / "Demands"
        / "hydrogenDemand_GWh_hydrogen.xlsx",
        engine=engine,
    )

    data.update(
        {"Hydrogen demand, operationRateFix": operationRateFix.loc[:, "cluster_0"]}
    )

    return data
