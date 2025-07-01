import modelBuilder
import os
import fine as fn
import geokit as gk
import numpy as np
import pandas as pd
import pytest
import shutil
import numbers

from .test_data import test_data_folder


    #################
    ##  INIT MODEL ##
    #################

@pytest.fixture
def ted():
    '''get the techo economic data dict via model builder path

    Returns
    -------
    dict
        techno economic data ted
    '''
    model_base_folder = os.path.join(test_data_folder, "test_output_data")
    os.makedirs(model_base_folder, exist_ok=True)

    location_shape_path = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    location_shape = gk.vector.extractFeatures(location_shape_path)

    commodityUnitsDict = {
        "electricity": r"GW$_{el}$",
        "hydrogen_gas": r"GW$_{H_{2},LHV}$",
    }

    modelManager = modelBuilder.modelManager(
        location_shape=location_shape,
        locationID_column="GID_1",
        commodityUnitsDict=commodityUnitsDict,
        cost_year=2050,
        model_base_folder=model_base_folder, #Note: A new intermediates folder will be created in the same directory as your main git modelBuilder repository
        srs=4326,
        path_to_techno_economic_data_yaml=None, # Use default data
        default_regions_fp=location_shape_path,
    )
    modelManager.technoEconomicData_setup()
    ted = modelManager.ted
    shutil.rmtree(model_base_folder)

    return ted

def check_vars_and_types(expected_vars, conversion_ted):
    for technology in conversion_ted:
        for expected_var in expected_vars.keys():
            excepted_type = expected_vars[expected_var]
            #test if var is available
            assert expected_var in conversion_ted[technology].keys(), f"Could not find var {expected_var} of tech {technology}"
            
            #assert if right data type
            if isinstance(conversion_ted[technology][expected_var], dict) and not (excepted_type == dict):
                #dict with years found! iterate over vars
                to_be_checked_var = conversion_ted[technology][expected_var]
                for year in to_be_checked_var.keys():
                    to_be_checked_var_at_year = to_be_checked_var[year]
                    if excepted_type:
                        assert isinstance(to_be_checked_var_at_year, excepted_type), f"Wrong dtype for var {expected_var} of tech {technology}"
            else:
                #no dict found, can access var directly
                if excepted_type:
                    assert isinstance(conversion_ted[technology][expected_var], excepted_type), f"Wrong dtype for var {expected_var} of tech {technology}"


def test_sources(ted):

    expected_vars = {
        "investPerCapacity" : numbers.Number,
        "opexFix" : numbers.Number,
        "opexPerOperation" : numbers.Number, # not implemented
        "interestRate" : numbers.Number,
        "economicLifetime" : numbers.Number,
        "commodity" : str,
        "citation" : str
    }

    sources_ted = ted["sources"]
    #manipulate this, as these are calculated from somewhere else!
    sources_ted["geothermal_EGS"]["investPerCapacity"] = 1
    check_vars_and_types(expected_vars, sources_ted)


def test_conversions(ted):

    expected_vars = {
        "commodityConversionFactors" : dict,
        "physicalUnit" : str,
        "investPerCapacity" : numbers.Number,
        "opexFix" : numbers.Number,
        "opexPerOperation" : numbers.Number,
        "economicLifetime" : numbers.Number,
        "citation" : str
    }

    conversion_ted = ted["conversion"]
    check_vars_and_types(expected_vars, conversion_ted)


def test_storages(ted):

    expected_vars = {
        "chargeEfficiency" : numbers.Number,
        "dischargeEfficiency" : numbers.Number,
        "cyclicLifetime" : numbers.Number,
        "selfDischarge" : numbers.Number,
        "chargeRate" : numbers.Number,
        "dischargeRate" : numbers.Number,
        "investPerCapacity" : numbers.Number,
        "opexFix" : numbers.Number,
        "opexPerChargeOperation" : numbers.Number,
        "opexPerDischargeOperation" : numbers.Number,
        "interestRate" : numbers.Number,
        "economicLifetime" : numbers.Number,
        "commodity" : str,
        "citation" : str
    }

    storage_ted = ted["storage"]
    check_vars_and_types(expected_vars, storage_ted)

def test_transmissions(ted):

    expected_vars = {
        "commodity" : str,
        "losses" : numbers.Number,
        "economicLifetime" : numbers.Number,
        "investPerCapacity" : numbers.Number,
        "opexFix" : numbers.Number,
        "opexPerOperation" : numbers.Number,
        "interestRate" : numbers.Number,
        "citation" : str,
    }

    onshore_offshore_symmetric_vars = [
        #"interestRate",
        "commodity",
        #TODO: check these
        "economicLifetime",
        #"losses",
        #"opexPerOperation"
    ]

    transmission_ted = ted["transmission"]
    check_vars_and_types(expected_vars, transmission_ted)

    #additional onshore / offshore check
    all_transmission_vars = transmission_ted.keys()
    for tech in all_transmission_vars:
        if "_onshore" in tech or "_offshore" in tech:
            other_tech = tech.replace("_onshore", "_offshore").replace("_offshore", "_onshore")
            assert other_tech in all_transmission_vars
            #check if same vars match symetrically:
            for sym_var in onshore_offshore_symmetric_vars:
                transmission_ted[tech][sym_var] == transmission_ted[other_tech][sym_var]
        else:
            #nothing to do
            pass

def test_demand(ted):

    expected_vars = {
        "commodity" : str,
        #"citation" : str #not applicable, as no data are stored apart from technical data for fine
    }

    demand_ted = ted["demand"]
    check_vars_and_types(expected_vars, demand_ted)