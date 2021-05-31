import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from FINE import utils
import FINE as fn



# TODO: declare private functions (and methods) with pre underscore

def generate_iteration_dicts(component_dict):
    """Creates iteration dictionaries that contain descriptions of all dataframes and series of the dictionaries esm_dict and component_dict.
    
    :param esm_dict: dictionary containing information about the esM instance
    :type esm_dict: dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
 
    :return: df_iteration_dict, series_iteration_dict
    """
    
    df_iteration_dict = {}

    for classname in component_dict:

        for component in component_dict[classname]:            

            for variable_description, data in component_dict[classname][component].items():
                description_tuple = (classname, component)

                if isinstance(data, pd.DataFrame):
                    if variable_description not in df_iteration_dict.keys():
                        df_iteration_dict[variable_description] = [description_tuple]
                    else:
                        df_iteration_dict[variable_description].append(description_tuple)

    series_iteration_dict = {} #NOTE: transmission components are series in component_dict (example index - cluster_0_cluster_2)

    for classname in component_dict:

        for component in component_dict[classname]:            

            for variable_description, data in component_dict[classname][component].items():
                description_tuple = (classname, component)

                if isinstance(data, pd.Series):
                    if variable_description not in series_iteration_dict.keys():
                        series_iteration_dict[variable_description] = [description_tuple]
                    else:
                        series_iteration_dict[variable_description].append(description_tuple)

    return df_iteration_dict, series_iteration_dict

def dimensional_data_to_xarray_dataset(esm_dict, component_dict):
    """Outputs all dimensional data to an xarray dataset.
    
    Note:  Here, "dimensional data" refers to data containing at least one of the dimensions of time and space.

    :param esm_dict: dictionary containing information about the esM instance
    :type esm_dict: dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
 
    :return: ds - xarray dataset containing all dimensional data of an esM instance
    """

    locations = list(esm_dict['locations'])

    df_iteration_dict, series_iteration_dict = generate_iteration_dicts(component_dict)

    # iterate over iteration dicts
    ds = xr.Dataset()

    # get all regional time series (space, time)
    for variable_description, description_tuple_list in df_iteration_dict.items():
        df_dict = {} # fn.dictionary of multiindex data frames that all contain all data for one variable
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.DataFrame):
                # print('data is pd.DataFrame')
                multi_index_dataframe = data.stack()
                if classname == 'LinearOptimalPowerFlow':
                    multi_index_dataframe.index.set_names("space", level=1, inplace=True)
                else:
                    multi_index_dataframe.index.set_names("space", level=1, inplace=True)
                    

                df_dict[df_description] = multi_index_dataframe # append or concat or so
                                        
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True) # ?

        ds_component = xr.Dataset()
        ds_component[f"ts_{variable_description}"] = df_variable.sort_index().to_xarray()

        ds = xr.merge([ds, ds_component])

    # get all 2d data (space, space)
    for variable_description, description_tuple_list in series_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series):

                if classname in ['Transmission', 'LinearOptimalPowerFlow']:    #NOTE: only ['Transmission', 'LinearOptimalPowerFlow'] are 2d classes 
                    # TODO: which one of transmission's components are 2d and which 1d or dimensionless

                    df = utils.transform1dSeriesto2dDataFrame(data, locations)
                    multi_index_dataframe = df.stack()
                    multi_index_dataframe.index.set_names(["space", "space_2"], inplace=True)

                    df_dict[df_description] = multi_index_dataframe

        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            ds_component = xr.Dataset()
            ds_component[f"2d_{variable_description}"] = df_variable.sort_index().to_xarray()  #NOTE: prefix 2d and 1d are added in this function

            ds = xr.merge([ds, ds_component])

    # get all 1d data (space)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
        for description_tuple in description_tuple_list:
            classname, component = description_tuple
    
            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series): # TODO: remove this line, as all data should be series (?)
                # if classname not in ['Transmission', 'LinearOptimalPowerFlow'] and len(data>= len(locations)):
                if classname not in ['Transmission', 'LinearOptimalPowerFlow']:
                    if len(data) >= len(locations): # TODO: this is a bugfix to remove '1d_locationalEligibility', do this properly
                        df_dict[df_description] = data.rename_axis("space")
            

        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            ds_component = xr.Dataset()
            ds_component[f"1d_{variable_description}"] = df_variable.sort_index().to_xarray()

            ds = xr.merge([ds, ds_component])

    return ds

 
def update_dicts_based_on_xarray_dataset(esm_dict, component_dict, xarray_dataset):
    """Replaces dimensional data and respective descriptions in component_dict and esm_dict with spatially aggregated data from xarray_dataset.

    Note:  Here, "dimensional data" refers to data containing at least one of the dimensions of time and space.

    :param esm_dict: dictionary containing information about the esM instance
    :type esm_dict: dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param xarray_dataset: dataset containing all "dimensional data" of an esM instance
    :type xarray_dataset: xarray.dataset

    :return: esm_dict, component_dict - updated dictionaries containing spatially aggregated data
    """
    
    df_iteration_dict, series_iteration_dict = generate_iteration_dicts(component_dict)

    # update esm_dict
    esm_dict['locations'] = set(str(value) for value in xarray_dataset.space.values)
    
    # update component_dict
    # set all regional time series (regions, time)
    for variable_description, description_tuple_list in df_iteration_dict.items():
        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"
            df = xarray_dataset[f"ts_{variable_description}"].sel(component=df_description).drop("component").to_dataframe().unstack(level=2)
            
            if len(df.columns) > 1:
                df.columns = df.columns.droplevel(0)

            component_dict[classname][component_description][variable_description] = df.sort_index()


    # set all 2d data (regions, regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"

            if classname in ['Transmission', 'LinearOptimalPowerFlow']:
                series = xarray_dataset[f"2d_{variable_description}"].sel(component=df_description
                                                            ).drop("component").to_dataframe().stack(level=0)

                series.index = series.index.droplevel(level=2).map('_'.join)

                component_dict[classname][component_description][variable_description] = series.sort_index()


    # set all 1d data (regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"

            if classname not in ['Transmission', 'LinearOptimalPowerFlow']:

                if variable_description not in ['commodityConversionFactors']: # TODO: this is a bugfix, properly implement this
                    # print(xarray_dataset[f"1d_{variable_description}"].sel(component=df_description))
                    series = xarray_dataset[f"1d_{variable_description}"].sel(component=df_description
                                                                            ).drop("component").to_dataframe().unstack(level=0)
                    series.index = series.index.droplevel(level=0)

                    component_dict[classname][component_description][variable_description] = series.sort_index()

    return esm_dict, component_dict



