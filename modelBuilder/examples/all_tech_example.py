# Workflow for a multi-regional energy system
 
# The workflow is structures as follows:
# 1. The required packages are imported
# 2. The shape file is loaded and aggregated
# 3. An energy system model instance is created
# 4. Commodity transmission components are added to the energy system model
# 5. Commodity sources are added to the energy system model
# 6. Commodity sinks are added to the energy system model
# 7. Commodity storage are added to the energy system model
# 8. Commodity conversion components are added to the energy system model
# 9. The energy system model is optimized
# 10. The selected results are postprocessed
# 11. Selected optimization results are presented
 
# %% Start

# 1. Import required packages 
import datetime
print(f'Scipt Started at {str(datetime.datetime.now())}')
import modelBuilder
import os
import sys
import geokit as gk
import pandas as pd
from modelBuilder import outputDataHandler
from modelBuilder.inputDataHandler import preprocess_union_shape
from modelBuilder.data import default_path_information
import yaml
import ast
from modelBuilder.data import data_folder

print("imports done",flush=True)

# 2. Set model base folder, load and process shape file 

#TODO: Name your own model base folder and set as arbitrary path to an existing folder
model_base_folder = "."
current_directory = os.path.dirname(__file__)
model_base_folder = os.path.join(current_directory, "results_all_tech_example") # TODO: set your base folder

# OPTIONAL: set intermediate folder / custom filepath to potentials
#intermediates_folder = "/fast/home/a-burdack/Intermediates_GID_0/windsolarglobal/global_intermediates/" # TODO: can be activated and used, if no intermediates in folder, new ones will be created with first model run

# Load shape file TODO: Choose one covering your need, also custom shapefile, none of the proposed can be overhanded
# GID_1:
shapeFilePath = yaml.load(open(os.path.abspath(os.path.join(data_folder,"default_general_paths.yml"))), Loader=yaml.FullLoader)["default_regions"]["filepath"] # TODO: choose this line for GID_1 default
# GID_0:
#shapeFilePath = yaml.load(open(os.path.abspath(os.path.join(data_folder,"default_general_paths.yml"))), Loader=yaml.FullLoader)["countries"]["filepath"]

# list with regions that you want to consider
regions_l = ['DEU.9_1','DEU.10_1'] # Germany: Niedersachsen, NRW
# make list to sql readable string
region_str = ",".join(f"'{region}'" for region in regions_l)

shape = gk.vector.extractFeatures(
    shapeFilePath,
    where=f"GID_1 in ({region_str})",
)


# OPTIONAL:Reduce number of regions to max_regions if necessary
#shape = preprocess_union_shape(
#    location_shape=shape,
#    max_regions=16,
#    return_as_gk=True,
#)


# 3. Create an energy system model instance

commodityUnitsDict = {
                "electricity": ("GW$_{el}$","GW"),
                "hydrogen_gas": ("GW$_{H_{2},LHV}$","GW"),
                "coal": ("GW$_{coal}$","GW")
                }


# Init Model Manager only writes vars to self.xyz
mb = modelBuilder.modelManager(
    location_shape=shape,
    locationID_column="GID_1",
    commodityUnitsDict=commodityUnitsDict,
    cost_year=2020,
    number_of_investment_periods=3,
    investment_period_interval=10,
    model_base_folder=model_base_folder,
    srs=4326,
    path_to_techno_economic_data=None,
    path_to_custom_input_data=None,
    weather_year=2018,
    #intermediates_folder=intermediates_folder,   # TODO: can be activated and used, if no intermediates in folder, new ones will be created with first model run. folder must be set above
)

print("model setup", flush=True)


# potential
mb.addPotentialConstGreenfield(
    technology="geothermal_EGS",
    N_cluster=3,
)

mb.addPotentialWithTSGreenfield(
    "ofpv_hsat",
    N_clusters=3,
    model_unit="GW",
)

mb.addPotentialWithTSGreenfield(
    "wind_onshore",
    N_clusters=10,
    model_unit="GW",
)

mb.addPotentialWithTSGreenfield(
    "wind_offshore",
    N_clusters=10,
    model_unit="GW",
)

# renewable stock potential

mb.addStockWithTSBrownfield(
    "ofpv_fixed_stock",
    N_clusters=3,
    model_unit="GW",
)

mb.addStockWithTSBrownfield(
    technology="wind_onshore_stock",
    N_clusters=10,
    model_unit='GW',
)

mb.addStockWithTSBrownfield(
    technology="wind_offshore_stock",
    N_clusters=10,
    model_unit="GW",
)


# fossil potential INFO: if db_stock_path overhanded stocks will be considered
mb.addPotentialConst(
    technology="coal_pp",
    model_unit="GW",
    db_stock_path="/storage_cluster/shared_data/2023_gears/FineUnion/potentials/global_plant_database/pypsa_power-plant-matching-tool_global.csv",
    db_unit='MW',
    db_name='pypsa_power-plant-matching-tool_global',
)

print("potentials added", flush=True)

# # conversion
mb.addConversionUnlimitedGreenfield(technology='electrolyzer_pem_compressor',regionalized_commodityConversionFactors=False)
print("Conversions added", flush=True)

# demand

# electricity
mb.addDemand(
    technology="electricity_demand", 
    factor=0.2, 
    path_abs_demands="/storage_cluster/shared_data/2023_gears/FineUnion/demands/electricity_demand_shitab/<YEAR>/abs_el_demands_<YEAR>_Downscaling__GCAM_5.3__NGFS__Net_Zero_2050_GWh_gid1.csv", 
    path_ts="/storage_cluster/shared_data/2023_gears/FineUnion/demands/plexos_time_series/All_Demand_UTC_2015_processed_all_gid0.csv")

# hydrogen
mb.addDemand(
    technology="hydrogen_gas_demand", 
    factor=0.1, 
    path_abs_demands="/storage_cluster/shared_data/2023_gears/FineUnion/demands/hydrogen_demand/results/<YEAR>/hydrogen_demand_GWh_gid1.csv", 
    path_ts="")
print("demands added", flush=True)

# storage
mb.addStorageUnlimitedGreenfield(technology='battery_LIIon')
mb.addStorageUnlimitedGreenfield(technology='hydrogenGas_vessel')
mb.addStorageLimitedGreenfield(technology='hydrogen_saltcavern')

print("storage added", flush=True)

# transmission
mb.addGridGreenfield(
    technology="electricity_grid",
)

mb.addGridGreenfield(
    technology="hydrogenGas_pipeline",
)
print("grids added", flush=True)

mb.addCommodityPurchase(
    technology="coal",
    model_unit="GW",
)
print("Commodity purchase added", flush=True)

# Optimization

mb.optimizeModel(
    timeSeriesAggregation=True,
    numberOfTypicalPeriods=40,
    threads=3,
    optimizationSpecs="Method=2 BarConvTol=1e-5 Presolve=2 AggFill=1 Crossover=0",
    )
print("optimized", flush=True)

mb.saveToNC4()
print("saved", flush=True)

# %% Postprocessing

postpro = outputDataHandler.OutputHandler(
    model_base_folder= model_base_folder,
    xr_dss=None,
    regions_shp=None,
    transmission_shp=None,
)

postpro.store_standard_evaluation()
postpro.store_default_plots()

# %% End