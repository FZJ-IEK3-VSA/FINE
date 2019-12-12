import xarray as xr
import FINE.IOManagement.dictIO as dictio
from FINE import utils
import FINE as fn
import numpy as np
import pandas as pd

def dimensional_data_to_xarray(esM):

    """Outputs all dimensional data, hence data containing at least one of the dimensions of time and space, to an xarray dataset"""

    esm_dict, component_dict = dictio.exportToDict(esM)

    locations = list(esm_dict['locations'])
    locations.sort() # TODO: should not be necessary any more

    # create iteration dict that indicates all iterators

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

    series_iteration_dict = {}

    for classname in component_dict:

        for component in component_dict[classname]:            

            for variable_description, data in component_dict[classname][component].items():
                description_tuple = (classname, component)

                if isinstance(data, pd.Series):
                    if variable_description not in series_iteration_dict.keys():
                        series_iteration_dict[variable_description] = [description_tuple]
                    else:
                        series_iteration_dict[variable_description].append(description_tuple)

    # iterate over iteration dict

    ds = xr.Dataset()

    for variable_description, description_tuple_list in df_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.DataFrame):
                multi_index_dataframe = data.stack()
                multi_index_dataframe.index.set_names("location", level=2, inplace=True)

                df_dict[df_description] = multi_index_dataframe # append or concat or so
                                        
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True) # ?

        da_component = df_variable.to_xarray()
        ds[variable_description] = da_component

    for variable_description, description_tuple_list in series_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series):

                if classname == 'Transmission':
                    # TODO: which one of transmission's components are 2d and which 1d or dimensionless

                    df = utils.transform1dSeriesto2dDataFrame(data, locations)
                    multi_index_dataframe = df.stack()
                    multi_index_dataframe.index.set_names(["location", "location_2"], inplace=True)

                    df_dict[df_description] = multi_index_dataframe

                # else:
                #     df_dict[df_description] = data.rename_axis("location")

        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            da_component = df_variable.to_xarray()
            ds[variable_description] = da_component

    for variable_description, description_tuple_list in series_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series):

                if classname != 'Transmission':
                    df_dict[df_description] = data.rename_axis("location")

        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            da_component = df_variable.to_xarray()
            ds[variable_description] = da_component

    return ds
