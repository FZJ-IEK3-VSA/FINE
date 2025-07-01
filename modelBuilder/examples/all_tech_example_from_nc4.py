# If you allready have an otimization result stored as NC4 and you want to do the postprocessing, 
# you do not need to run the whole optimization again. 
# 
# 1. You just need to set up the model, as it was set up for your nc4 run
# 2. You run the post processing
 
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