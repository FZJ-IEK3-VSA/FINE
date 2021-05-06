import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import FINE as fn
from FINE import utils
from FINE.IOManagement import dictIO


# TODO: declare private functions (and methods) with pre underscore

def generate_iteration_dicts(component_dict):
    """Creates iteration dictionaries that contain descriptions of all 
    dataframes, series, and constants present in component_dict.
    
    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
 
    :return: df_iteration_dict, series_iteration_dict, constants_iteration_dict
    """
    
    df_iteration_dict, series_iteration_dict, constants_iteration_dict = {}, {}, {}

    for classname in component_dict:

        for component in component_dict[classname]:            

            for variable_description, data in component_dict[classname][component].items():
                description_tuple = (classname, component)
                
                def _append_to_iteration_dicts(_variable_description, _data):

                    if isinstance(_data, dict):
                        for key, value in _data.items():
                            nested_variable_description = f'{_variable_description}.{key}'

                            _append_to_iteration_dicts(nested_variable_description, value)

                    elif isinstance(_data, pd.DataFrame):
                        if _variable_description not in df_iteration_dict.keys():
                            df_iteration_dict[_variable_description] = [description_tuple]
                        else:
                            df_iteration_dict[_variable_description].append(description_tuple)
                    
                    #NOTE: transmission components are series in component_dict 
                    # (example index - cluster_0_cluster_2)
                    
                    elif isinstance(_data, pd.Series):       
                        if _variable_description not in series_iteration_dict.keys():
                            series_iteration_dict[_variable_description] = [description_tuple]
                        else:
                            series_iteration_dict[_variable_description].append(description_tuple)
                            
                    else:
                        if _variable_description not in constants_iteration_dict.keys():
                            constants_iteration_dict[_variable_description] = [description_tuple]
                        else:
                            constants_iteration_dict[_variable_description].append(description_tuple)

                _append_to_iteration_dicts(variable_description, data)


    return df_iteration_dict, series_iteration_dict, constants_iteration_dict



def convert_esM_instance_to_xarray_dataset(esM):  
    """Takes esM instance and converts it into an xarray dataset,
    which can be saved in netCDF format.  #TODO: update the docstring 
    
    :param esm_dict: dictionary containing information about the esM instance
    :type esm_dict: dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
 
    :return: ds - xarray dataset containing all dimensional data of an esM instance
    """
    
    #STEP 1. Get the esm and component dicts 
    esm_dict, component_dict = dictIO.exportToDict(esM)

    locations = list(esm_dict['locations'])

    #STEP 2. Get the iteration dicts 
    df_iteration_dict, series_iteration_dict, constants_iteration_dict = generate_iteration_dicts(component_dict)
    
    #STEP 3. Iterate through each iteration dicts and add the data to a xarray Dataset.
    # data comes from component_dict
    ds = xr.Dataset()

    #STEP 3a. Add all regional time series (dimensions - space, time)
    for variable_description, description_tuple_list in df_iteration_dict.items():
        df_dict = {} # fn.dictionary of multiindex data frames that all contain all data for one variable
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            if '.' in variable_description:
                nested_variable_description = variable_description.split(".")
                data = component_dict[classname][component][nested_variable_description[0]][nested_variable_description[1]]
            else:
                data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.DataFrame):
                
                multi_index_dataframe = data.stack()
                
                multi_index_dataframe.index.set_names("time", level=0, inplace=True)
                multi_index_dataframe.index.set_names("space", level=1, inplace=True)
                
                df_dict[df_description] = multi_index_dataframe # append or concat or so
                                        
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True) # ?

        ds_component = xr.Dataset()
        ds_component[f"ts_{variable_description}"] = df_variable.sort_index().to_xarray()

        ds = xr.merge([ds, ds_component])

    #STEP 3b. Add all 2d data (dimensions - space, space)
    for variable_description, description_tuple_list in series_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            if '.' in variable_description:
                nested_variable_description = variable_description.split(".")
                data = component_dict[classname][component][nested_variable_description[0]][nested_variable_description[1]]
            else:
                data = component_dict[classname][component][variable_description]
            # data = component_dict[classname][component][variable_description]

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

    #STEP 3c. Add all 1d data (dimension - space)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
        for description_tuple in description_tuple_list:
            classname, component = description_tuple
    
            df_description = f"{classname}, {component}"

            if '.' in variable_description:
                nested_variable_description = variable_description.split(".")
                data = component_dict[classname][component][nested_variable_description[0]][nested_variable_description[1]]
            else:
                data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series): #TODO: remove this line, as all data should be series (?)
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

    #STEP 3d. Add all constants 
    for variable_description, description_tuple_list in constants_iteration_dict.items():
        
        df_dict = {} 
        for description_tuple in description_tuple_list:
            classname, component = description_tuple
            df_description = f"{classname}, {component}"

            if '.' in variable_description:
                nested_variable_description = variable_description.split(".")
                data = component_dict[classname][component][nested_variable_description[0]][nested_variable_description[1]]
            else:
                data = component_dict[classname][component][variable_description]

            df_dict[df_description] = data
            
        if len(df_dict) > 0:
            df_variable = pd.Series(df_dict)       #TODO: optimize the code, maybe directly take series instead of dict 
            df_variable.index.set_names("component", inplace=True)
            
            ds_component = xr.Dataset()
            ds_component[f"0d_{variable_description}"] = xr.DataArray.from_series(df_variable)
            
            ds = xr.merge([ds, ds_component])
            
    #STEP 4. Add the data present in esm_dict as xarray attributes (these are dimensionless data). 
    # Exclude locations because this comes from the xarray dimensions.  
    esm_dict.pop('locations', None)
    ds.attrs = esm_dict

    return ds



# def update_dicts_based_on_xarray_dataset(esm_dict, component_dict, xarray_dataset):
#     """Replaces dimensional data and respective descriptions in component_dict and esm_dict with spatially aggregated data from xarray_dataset.

#     Note:  Here, "dimensional data" refers to data containing at least one of the dimensions of time and space.

#     :param esm_dict: dictionary containing information about the esM instance
#     :type esm_dict: dict

#     :param component_dict: dictionary containing information about the esM instance's components
#     :type component_dict: dict

#     :param xarray_dataset: dataset containing all "dimensional data" of an esM instance
#     :type xarray_dataset: xarray.dataset

#     :return: esm_dict, component_dict - updated dictionaries containing spatially aggregated data
#     """
    
#     df_iteration_dict, series_iteration_dict = generate_iteration_dicts(component_dict)

#     # update esm_dict
#     esm_dict['locations'] = set(str(value) for value in xarray_dataset.space.values)
    
#     # update component_dict
#     # set all regional time series (regions, time)
#     for variable_description, description_tuple_list in df_iteration_dict.items():
#         for description_tuple in description_tuple_list:
#             classname, component_description = description_tuple

#             df_description = f"{classname}, {component_description}"
#             df = xarray_dataset[f"ts_{variable_description}"].sel(component=df_description).drop("component").to_dataframe().unstack(level=2)
            
#             if len(df.columns) > 1:
#                 df.columns = df.columns.droplevel(0)

#             component_dict[classname][component_description][variable_description] = df.sort_index()


#     # set all 2d data (regions, regions)
#     for variable_description, description_tuple_list in series_iteration_dict.items():

#         for description_tuple in description_tuple_list:
#             classname, component_description = description_tuple

#             df_description = f"{classname}, {component_description}"

#             if classname in ['Transmission', 'LinearOptimalPowerFlow']:
#                 series = xarray_dataset[f"2d_{variable_description}"].sel(component=df_description
#                                                             ).drop("component").to_dataframe().stack(level=0)

#                 series.index = series.index.droplevel(level=2).map('_'.join)

#                 component_dict[classname][component_description][variable_description] = series.sort_index()


#     # set all 1d data (regions)
#     for variable_description, description_tuple_list in series_iteration_dict.items():

#         for description_tuple in description_tuple_list:
#             classname, component_description = description_tuple

#             df_description = f"{classname}, {component_description}"

#             if classname not in ['Transmission', 'LinearOptimalPowerFlow']:

#                 if variable_description not in ['commodityConversionFactors']: # TODO: this is a bugfix, properly implement this
#                     # print(xarray_dataset[f"1d_{variable_description}"].sel(component=df_description))
#                     series = xarray_dataset[f"1d_{variable_description}"].sel(component=df_description
#                                                                             ).drop("component").to_dataframe().unstack(level=0)
#                     series.index = series.index.droplevel(level=0)

#                     component_dict[classname][component_description][variable_description] = series.sort_index()

#     return esm_dict, component_dict

def convert_xarray_dataset_to_esM_instance(xarray_dataset):
    
    esm_dict = xarray_dataset.attrs

    esm_dict['locations'] = set(str(value) for value in xarray_dataset.space.values)

    component_dict = {}
    for component in xarray_dataset.component.values:
        sub_xarray = xarray_dataset.sel(component=component)
        
        for variable in sub_xarray.data_vars:
                
            # if not xr.ufuncs.isnan(sub_xarray[variable].values).all():
            if not pd.isnull(sub_xarray[variable].values).all():
                # numpy.isnan(myarray).any()
                
                # set all regional time series (regions, time)minimal_test_esM
                if variable[:3]== 'ts_':
                    
                    # print(f'{variable} is chosen')
                    # TODO: check unstack level and variable name 'level_1' for time
                    # df = sub_xarray[variable].drop("component").to_dataframe().unstack(level=2)
                    df = sub_xarray[variable].drop("component").to_dataframe().unstack(level=1)
                    if len(df.columns) > 1:
                        df.columns = df.columns.droplevel(0)
                    
                    [class_name, comp_name] = component.split(', ')
                    if class_name not in component_dict.keys():
                        component_dict.update({class_name: {}})
                    if comp_name not in component_dict.get(class_name).keys():
                        component_dict.get(class_name).update({comp_name: {}})
                    if "." in variable:
                        [var_name, nested_var_name] = variable.split(".")
                        if var_name[3:] not in component_dict.get(class_name).get(comp_name).keys():
                            component_dict.get(class_name).get(comp_name).update({var_name[3:]: {}})
                        component_dict.get(class_name).get(comp_name).get(var_name[3:]).update({nested_var_name: df.sort_index()})
                    else:
                        component_dict.get(class_name).get(comp_name).update({variable[3:]: df.sort_index()})

                # set all 2d data (regions, regions)
                elif variable[:3]== '2d_':
                    
                    df = sub_xarray[variable].drop("component").to_dataframe().unstack(level=1)
                    if len(df.columns) > 1:
                        df.columns = df.columns.droplevel(0)
                    df.sort_index(axis=0, inplace=True)
                    df.sort_index(axis=1, inplace=True)
                    
                    #series = sub_xarray[variable].drop("component").to_dataframe().stack(level=0)
                    #series.index = series.index.droplevel(level=2).map('_'.join)

                    [class_name, comp_name] = component.split(', ')
                    if class_name not in component_dict.keys():
                        component_dict.update({class_name: {}})
                    if comp_name not in component_dict.get(class_name).keys():
                        component_dict.get(class_name).update({comp_name: {}})
                    if "." in variable:
                        [var_name, nested_var_name] = variable.split(".")
                        if var_name[3:] not in component_dict.get(class_name).get(comp_name).keys():
                            component_dict.get(class_name).get(comp_name).update({var_name[3:]: {}})
                        component_dict.get(class_name).get(comp_name).get(var_name[3:]).update({nested_var_name: df})
                    else:
                        component_dict.get(class_name).get(comp_name).update({variable[3:]: df})

                # # set all 1d data (regions)
                elif variable[:3]== '1d_':
                    # print(f'{variable} is chosen')
                    series = sub_xarray[variable].drop("component").to_dataframe().unstack(level=0)
                    series.index = series.index.droplevel(level=0)
                    
                    [class_name, comp_name] = component.split(', ')
                    if class_name not in component_dict.keys():
                        component_dict.update({class_name: {}})
                    if comp_name not in component_dict.get(class_name).keys():
                        component_dict.get(class_name).update({comp_name: {}})
                    if "." in variable:
                        [var_name, nested_var_name] = variable.split(".")
                        if var_name[3:] not in component_dict.get(class_name).get(comp_name).keys():
                            component_dict.get(class_name).get(comp_name).update({var_name[3:]: {}})
                        component_dict.get(class_name).get(comp_name).get(var_name[3:]).update({nested_var_name: series.sort_index()})
                    else:
                        component_dict.get(class_name).get(comp_name).update({variable[3:]: series.sort_index()})

                # # set all 0d data 
                elif variable[:3]== '0d_':
                    # print(f'{variable} is chosen')
                    var_value = sub_xarray[variable].values
                    # series = sub_xarray[variable].drop("component").to_dataframe().unstack(level=0)
                    # series.index = series.index.droplevel(level=0)
                    
                    [class_name, comp_name] = component.split(', ')
                    if class_name not in component_dict.keys():
                        component_dict.update({class_name: {}})
                    if comp_name not in component_dict.get(class_name).keys():
                        component_dict.get(class_name).update({comp_name: {}})
                    if "." in variable:
                        [var_name, nested_var_name] = variable.split(".")
                        if var_name[3:] not in component_dict.get(class_name).get(comp_name).keys():
                            component_dict.get(class_name).get(comp_name).update({var_name[3:]: {}})
                        component_dict.get(class_name).get(comp_name).get(var_name[3:]).update({nested_var_name: var_value.item()})
                    else:
                        component_dict.get(class_name).get(comp_name).update({variable[3:]: var_value.item()})

    # Create esm from esm_dict and component_dict
    return dictIO.importFromDict(esm_dict, component_dict)
