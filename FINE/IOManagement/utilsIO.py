import warnings

import math
import numpy as np
import pandas as pd
import xarray as xr 

def transform1dSeriesto2dDataFrame(series, locations, separator="_"):
    values = np.zeros((len(locations), len(locations)))

    df = pd.DataFrame(values, columns=locations, index=locations)

    for row in series.iteritems():
        
        try:
            id_1, id_2 = row[0].split(separator) 
        except:
            warnings.warn(f'More than one {separator} found in series index. \
            Therefore, {separator} is not used to split the index')

            row_center_id = math.ceil(len(row[0])/2)
            id_1, id_2 = row[0][:row_center_id-1], row[0][row_center_id:]

        df.loc[id_1, id_2] = row[1]

    return df

class PowerDict(dict):  
    '''
    Dictionary with additional functions. 
    Helps in creating nested dictionaries on the fly.
    '''
    def __init__(self, parent=None, key=None):
        self.parent = parent
        self.key = key

    def __missing__(self, key): 
        '''
        Creation of subdictionaries on fly
        '''
        self[key] = PowerDict(self, key)
        return self[key]

    def append(self, item):
        '''
        Additional append function for lists in dict
        '''
        self.parent[self.key] = [item]

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        try:
            val.parent = self
            val.key = key
        except AttributeError:
            pass

def generateIterationDicts(component_dict):
    """Creates iteration dictionaries that contain descriptions of all 
    dataframes, series, and constants present in component_dict.
    
    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
 
    :return: df_iteration_dict, series_iteration_dict, constants_iteration_dict
    """
    
    df_iteration_dict, series_iteration_dict, constants_iteration_dict = {}, {}, {}

    # Loop through every class-component-variable combination 
    for classname in component_dict:

        for component in component_dict[classname]:            

            for variable_description, data in component_dict[classname][component].items():
                description_tuple = (classname, component)
                
                #private function to check if the current variable is a dict, df, series or constant. 
                # If its a dict (in the case of commodityConversionFactors), this is unpacked and the 
                # the function is run on each value in dict 
                def _append_to_iteration_dicts(_variable_description, _data):

                    if isinstance(_data, dict):
                        for key, value in _data.items():
                            nested_variable_description = f'{_variable_description}.{key}' #NOTE: a . is introduced in the variable here  

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


def addDFVariablesToXarray(xr_ds, component_dict, df_iteration_dict):
    """Adds all variables whose data is contained in a pd.DataFrame to xarray dataset. 
    These variables are normally regional time series (dimensions - space, time) 
    
    :param xr_ds: xarray dataset to which the DF variables should be added 
    :type xr_ds: xr.Dataset

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param df_iteration_dict: dictionary with:
        keys - DF variable names
        values - tuple of component class and component name
    :type df_iteration_dict: dict

    :return: xr_ds
    """

    for variable_description, description_tuple_list in df_iteration_dict.items():
        df_dict = {} 

        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"
            
            # If a . is present in variable name, then the data would be 
            # another level further in the component_dict 
            if '.' in variable_description:      
                [var_name, subvar_name] = variable_description.split('.')
                data = component_dict[classname][component][var_name][subvar_name]
            else:
                data = component_dict[classname][component][variable_description]

            multi_index_dataframe = data.stack()
            multi_index_dataframe.index.set_names("time", level=0, inplace=True)
            multi_index_dataframe.index.set_names("space", level=1, inplace=True)
            
            df_dict[df_description] = multi_index_dataframe 
                                        
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True) 

        ds_component = xr.Dataset()
        ds_component[f"ts_{variable_description}"] = df_variable.sort_index().to_xarray() 

        xr_ds = xr.merge([xr_ds, ds_component])
    
    return xr_ds

def addSeriesVariablesToXarray(xr_ds, component_dict, series_iteration_dict, locations):
    """Adds all variables whose data is contained in a pd.Series to xarray dataset. 
    These variables can be either:
        - 2d (dimensions - space, space). Series indices in this case are packed like loc1_loc2
        or
        - 1d (dimension - space)
        or 
        - time series (dimension - time). This situation is unique to single node esM model 
    
    :param xr_ds: xarray dataset to which the series variables should be added 
    :type xr_ds: xr.Dataset

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param series_iteration_dict: dictionary with:
        keys - series variable names
        values - tuple of component class and component name
    :type series_iteration_dict: dict

    :param locations: sorted esM locations 
    :type locations: list 

    :return: xr_ds
    """

    for variable_description, description_tuple_list in series_iteration_dict.items():
        space_space_dict = {} 
        space_dict = {}
        time_dict = {}
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            # If a . is present in variable name, then the data would be 
            # another level further in the component_dict 
            if '.' in variable_description:
                [var_name, subvar_name] = variable_description.split('.')
                data = component_dict[classname][component][var_name][subvar_name]
            else:
                data = component_dict[classname][component][variable_description]

            # Only ['Transmission', 'LinearOptimalPowerFlow'] are 2d classes.
            # So, if classname is one of these, append the data to df_space_space_dict 
            if classname in ['Transmission', 'LinearOptimalPowerFlow']: 

                df = transform1dSeriesto2dDataFrame(data, locations)
                multi_index_dataframe = df.stack()
                multi_index_dataframe.index.set_names(["space", "space_2"], inplace=True)

                space_space_dict[df_description] = multi_index_dataframe

            else:
                # If the data indices correspond to esM locations, then the 
                # data is appended to df_space_dict, else df_time_dict
                if sorted(locations) == sorted(data.index.values):
                    space_dict[df_description] = data.rename_axis("space")
                else:
                    time_dict[df_description] = data.rename_axis("time")
                    time_dict[df_description] = pd.concat({locations[0]: time_dict[df_description]}, names=['space'])
                    time_dict[df_description] = time_dict[df_description].reorder_levels(["time", "space"])

        # If the dicts are populated with at least one item, 
        # process them further and merge with xr_ds 
        if len(space_space_dict) > 0:
            df_variable = pd.concat(space_space_dict)
            df_variable.index.set_names("component", level=0, inplace=True) 
            ds_component = xr.Dataset()
            ds_component[f"2d_{variable_description}"] = df_variable.sort_index().to_xarray()  

            xr_ds = xr.merge([xr_ds, ds_component])
        
        if len(space_dict) > 0:
            df_variable = pd.concat(space_dict)
            df_variable.index.set_names("component", level=0, inplace=True) 
            ds_component = xr.Dataset()
            ds_component[f"1d_{variable_description}"] = df_variable.sort_index().to_xarray()
            
            xr_ds = xr.merge([xr_ds, ds_component])

        if len(time_dict) > 0:
            df_variable = pd.concat(time_dict)
            df_variable.index.set_names("component", level=0, inplace=True) 
            ds_component = xr.Dataset()
            ds_component[f"ts_{variable_description}"] = df_variable.sort_index().to_xarray()
            
            xr_ds = xr.merge([xr_ds, ds_component])

    return xr_ds

def addConstantsToXarray(xr_ds, component_dict, constants_iteration_dict):
    """Adds all variables whose data is just a constant value, to xarray dataset. 
    
    :param xr_ds: xarray dataset to which the constant value variables should be added 
    :type xr_ds: xr.Dataset

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param constants_iteration_dict: dictionary with:
        keys - constant value variable names
        values - tuple of component class and component name
    :type constants_iteration_dict: dict

    :return: xr_ds
    """

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
            

        df_variable = pd.Series(df_dict)      
        df_variable.index.set_names("component", inplace=True)
        
        ds_component = xr.Dataset()
        ds_component[f"0d_{variable_description}"] = xr.DataArray.from_series(df_variable)
        
        xr_ds = xr.merge([xr_ds, ds_component])

    return xr_ds
    

        
    
            

            

            
            
        