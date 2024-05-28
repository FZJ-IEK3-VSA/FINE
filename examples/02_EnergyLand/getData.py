import pandas as pd
import os
from pathlib import Path


def getData(engine="openpyxl"):
    current_directory = Path(__file__).parent.absolute()
    inProfileDataPath = os.path.join(current_directory, "Input_profiles_fine.xlsx")
    outProfileDataPath = os.path.join(current_directory, "Output_profiles_fine.xlsx")
    esDataPath = os.path.join(current_directory, "Potentials.xlsx")

    data = {}

    inProfile = pd.read_excel(inProfileDataPath, index_col=0, engine=engine)
    outProfile = pd.read_excel(outProfileDataPath, index_col=0, engine=engine)
    esMaxCap = pd.read_excel(esDataPath, index_col=0, engine=engine)

    # Onshore data

    data.update({"Wind_onshore, capacityMax": esMaxCap.loc["Onshore", "Potential"]})
    data.update({"Wind_onshore, operationRateMax": inProfile.loc[:, "OnshoreEnergy"]})

    # Offshore Data

    data.update({"Wind_offshore, capacityMax": esMaxCap.loc["Offshore", "Potential"]})
    data.update({"Wind_offshore, operationRateMax": inProfile.loc[:, "OffshoreEnergy"]})

    # PV data

    data.update({"PV, capacityMax": esMaxCap.loc["PV", "Potential"]})
    data.update({"PV, operationRateMax": inProfile.loc[:, "SolarEnergy"]})

    # Electricity Import data

    data.update({"el_Import, operationRateMax": inProfile.loc[:, "total_impres2050"]})
    data.update({"el_Import, capacityMax": 100})

    # Hydrogen Import data

    data.update({"H2_Import, operationRateMax": inProfile.loc[:, "HydroEnergy"]})

    # Electricity demand data

    data.update({"Electricity demand, operationRateFix": outProfile.loc[:, "EDemand"]})

    # Transport

    data.update({"T_demand, operationRateFix": outProfile.loc[:, "TDemand"]})

    # Low temperature residential heat demand

    data.update({"LtHeat_demand, operationRateFix": outProfile.loc[:, "HHPHDemand"]})

    # Process heat demand

    data.update({"pHeat_demand, operationRateFix": outProfile.loc[:, "INDPHDemand"]})

    # biomass Source

    data.update({"wood_source, capacityMax": 16.2})
    data.update({"bioslurry_source, capacityMax": 2.9})
    data.update({"biowaste_source, capacityMax": 0.7})
    return data
