"""
Created on Thu Nov  8 10:02:46 2018

|br| @author: FINE Developer Team (FZJ IEK-3)
"""

import pandas as pd
import os


def getData(engine="openpyxl"):
    cwd = os.getcwd()
    inputDataPath = os.path.join(cwd, "InputData")
    data = {}

    # Locations
    locations = set(
        pd.read_excel(os.path.join(inputDataPath, "Locations.xlsx"), engine=engine)
    )

    data.update({"locations": locations})

    # PV data
    gen_capacityMax = pd.read_excel(
        os.path.join(inputDataPath, "PV_Capacity.xlsx"),
        sheet_name="GenerationCapacities",
        index_col=0,
        engine=engine,
    )
    gen_capacityMax = gen_capacityMax.loc["PV_south"].T
    gen_operationRateMax = pd.read_excel(
        os.path.join(inputDataPath, "PV_Generation.xlsx"), engine=engine
    )

    data.update({"PV, capacityMax": gen_capacityMax})
    data.update({"PV, operationRateMax": gen_operationRateMax})

    # Heat & Battery Storage
    st_capacityMax = pd.read_excel(
        os.path.join(inputDataPath, "Storage_capacities.xlsx"),
        index_col=0,
        engine=engine,
    )
    st_capacityMax_TS = st_capacityMax.loc["Thermal Storage"]
    st_capacityMax_BS = st_capacityMax.loc["Battery"]

    data.update({"TS, capacityMax": st_capacityMax_TS})
    data.update({"BS, capacityMax": st_capacityMax_BS})

    # Transmission Technologies
    tr_distances_el = pd.read_excel(
        os.path.join(inputDataPath, "grid_length_matrix.xlsx"),
        index_col=0,
        engine=engine,
    )
    tr_capacityFix_el = pd.read_excel(
        os.path.join(inputDataPath, "grid_capacity_matrix.xlsx"),
        index_col=0,
        engine=engine,
    )

    data.update({"cables, capacityFix": tr_capacityFix_el})
    data.update({"cables, distances": tr_distances_el})

    data.update({"NG, capacityFix": tr_capacityFix_el})
    data.update({"NG, distances": tr_distances_el})

    # Denand Data
    Edemand_operationRateFix = pd.read_excel(
        os.path.join(inputDataPath, "E_Demand.xlsx"), engine=engine
    )
    Hdemand_operationRateFix = pd.read_excel(
        os.path.join(inputDataPath, "Heat_Demand.xlsx"), engine=engine
    )

    data.update({"Electricity demand, operationRateFix": Edemand_operationRateFix})
    data.update({"Heat demand, operationRateFix": Hdemand_operationRateFix})

    # Purchase Data
    Pu_operationRateMax_El = pd.read_excel(
        os.path.join(inputDataPath, "purchaseElectricity.xlsx"),
        index_col=0,
        engine=engine,
    )
    Pu_operationRateMax_NG = pd.read_excel(
        os.path.join(inputDataPath, "purchaseNaturalGas.xlsx"),
        index_col=0,
        engine=engine,
    )

    data.update({"El Purchase, operationRateMax": Pu_operationRateMax_El})
    data.update({"NG Purchase, operationRateMax": Pu_operationRateMax_NG})

    return data
