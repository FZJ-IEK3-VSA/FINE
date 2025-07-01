#%%
import os


import modelBuilder
import geokit as gk

shapeFilePath = "/storage_cluster/shared_data/2023_gears/FineUnion/regions/gadm36_GID1(0)_EastWest_and_largeRegion_split_epsg4326_withUnion_v2_1.shp"
shape = gk.vector.extractFeatures(shapeFilePath)


# select AUSNZL = 23
shape = shape[shape["Union_No"] == 2]
model_base_folder = "/storage_cluster/projects/2022-a-burdack-phd/workspace/modelresults/testmodelbuilder/"  # set your result folder


commodityUnitsDict = {
    "electricity": r"GW$_{el}$",
    "hydrogen_gas": r"GW$_{H_{2},LHV}$",
}

modelManager = modelBuilder.modelManager(
    location_shape=shape,
    locationID_column="GID_1",
    commodityUnitsDict=commodityUnitsDict,
    cost_year=2050,
    model_base_folder=model_base_folder,  # Note: A new intermediates folder will be created in the same directory as your main git modelBuilder repository
    srs=4326,
    path_to_techno_economic_data_yaml=None,  # Use default data
)

#%%

modelManager.technoEconomicData_setup()
modelManager.inputHandlerSetup()
modelManager.modelSetup()

#%%

mB = modelManager

base_folder = None  # TODO: we need some pv data here to load

path_grids = r"/storage_cluster/shared_data/2023_gears/FineUnion/existing_grids/electricity_grid/osm_all_e_grids_with_caps.shp"

mB.addGridBrownfield(technology="electricity_grid", model_unit="GW")


#%%

mB.esM.getComponentAttribute("electricity_grid_brownfield", "capacityFix")


#%%

from shapely.geometry import LineString

#%%

import pandas as pd

pd.DataFrame([[1,2,66],[2,3],[4,5]])