import json
import numpy as np
import os
import warnings

# import third party packages
import yaml

# import from other modules
from .data import data_folder

# Put all utility functions in here! e.g. checks for input, etc.

#TODO remove below function when Singletons are implemented ISSUE112
def get_raw_regions(shore_type=None, country_list=[])->list:
    """_summary_

    Args:
        shore_type (str, optional): 
        'Onshore' or 'Offshore'. If None passed, both will be returned. Defaults to None:str.

    Returns:
        list: List with str default region codes for the input selection.
    """
    if not shore_type is None:
        assert isinstance(shore_type, str), f"shore_type must be a str."
        assert shore_type.upper() in ['ONSHORE','OFFSHORE'], f"shore_type must be either 'Onshore' or 'Offshore' (case-insensitive)."
        shore_type_list=[shore_type.upper()]
    else:
        shore_type_list=['ONSHORE','OFFSHORE']
    # load the raw default region code dict
    with open(os.path.join(data_folder, 'default_regions.json')) as json_file:
        all_default_regioncodes_dict = json.load(json_file)
    # check or set country list
    available_countries=[]
    for shore in shore_type_list:
        available_countries+=all_default_regioncodes_dict[shore].keys()
    if country_list == []:
        country_list=available_countries
    else:
        assert isinstance(country_list, list), f"country_list must be a list."
        assert all([country in available_countries for country in country_list]), f"All countries in country_list must be in: {', '.join(available_countries)}"
    # extract the raw gid1split codes for the given params
    all_default_regioncodes=[]
    for k1 in all_default_regioncodes_dict.keys():
        # skip if shore type is not desired
        if not k1.upper() in shore_type_list:
            continue
        for k2 in all_default_regioncodes_dict[k1].keys():
            # skip if country is not desired
            if not k2.upper() in country_list:
                continue
            # else add all region codes to all_default_regioncodes
            all_default_regioncodes+=all_default_regioncodes_dict[k1][k2]

    return all_default_regioncodes


def get_technology_dict(component_library):
    """Creates a dict with only the classes and technology names
    """
    technology_dict = {}
    for key, value in component_library.items():
        if "esM" not in key:
            technology_dict[key] = list(component_library[key].keys())
    return technology_dict

# select certain technologies
def select_technologies_from_dict(component_library, technology_selection):
    """Selects Subset of component library based on dict
    """ 
    component_library_selection = {}
    for key, values in technology_selection.items():
        component_library_selection[key] = dict()
        if isinstance(values, list):
            for value in values:
                component_library_selection[key][value] = component_library[key][value]
        else:
            component_library_selection[key][values] = component_library[key][values]
    return component_library_selection
