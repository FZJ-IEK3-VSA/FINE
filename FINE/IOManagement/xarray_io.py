import xarray as xr
import FINE.IOManagement.dictIO as dictio
from FINE import utils
import FINE as fn
import numpy as np
import pandas as pd

def generate_iteration_dicts(esm_dict, component_dict):
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

    return df_iteration_dict, series_iteration_dict

def dimensional_data_to_xarray(esM):
    """Outputs all dimensional data, hence data containing at least one of the dimensions of time and space, to an xarray dataset"""

    esm_dict, component_dict = dictio.exportToDict(esM)

    locations = list(esm_dict['locations'])
    locations.sort() # TODO: should not be necessary any more

    df_iteration_dict, series_iteration_dict = generate_iteration_dicts(esm_dict, component_dict)

    # iterate over iteration dict

    ds = xr.Dataset()

    # get all regional time series (regions, time)
    for variable_description, description_tuple_list in df_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"
            # print(df_description)

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.DataFrame):
                # print('data is pd.DataFrame')
                multi_index_dataframe = data.stack()
                multi_index_dataframe.index.set_names("space", level=2, inplace=True)

                df_dict[df_description] = multi_index_dataframe # append or concat or so
                                        
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True) # ?

        ds_component = xr.Dataset()
        ds_component[variable_description] = df_variable.to_xarray()

        ds = xr.merge([ds, ds_component])

    # get all 2d data (regions, regions)
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
                    multi_index_dataframe.index.set_names(["space", "space_2"], inplace=True)

                    df_dict[df_description] = multi_index_dataframe

                # TODO: shouldn't this case be uncommented?
                # else:
                #     df_dict[df_description] = data.rename_axis("space")


        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            ds_component = xr.Dataset()
            ds_component[variable_description] = df_variable.to_xarray()

            ds = xr.merge([ds, ds_component])

    # get all 1d data (regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
        for description_tuple in description_tuple_list:
            classname, component = description_tuple
    
            df_description = f"{classname}, {component}"
            # print(df_description)

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series):
                # print('data is pd.Series')

                if classname != 'Transmission':
                    df_dict[df_description] = data.rename_axis("space")

        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            ds_component = xr.Dataset()
            ds_component[variable_description] = df_variable.to_xarray()

            ds = xr.merge([ds, ds_component])

    return ds

def update_dicts_based_on_xarray_dataset(esm_dict, component_dict, xarray_dataset):
    """Replaces dimensional data (using aggregated data from xarray_dataset) and respective description in component_dict and esm_dict"""
    df_iteration_dict, series_iteration_dict = generate_iteration_dicts(esm_dict, component_dict)

    # update esm_dict
    esm_dict['locations'] = set(str(value) for value in xarray_dataset.space.values)
    
    # update component_dict
    # set all regional time series (regions, time)
    for variable_description, description_tuple_list in df_iteration_dict.items():
        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"
            try:
                df = xarray_dataset[variable_description].sel(component=df_description).drop("component").to_dataframe().unstack(level=0)            

                component_dict[classname][component_description][variable_description] = df

            except:
                print(f"'{variable_description}' for '{df_description}' not in xarray_dataset")
                # TODO: these data should not be missing, should they? check to_dict function @Leander

    # set all 2d data (regions, regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():
    
        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"

            try:
                if classname == 'Transmission':
                    df = xarray_dataset[variable_description].sel(component=df_description).drop("component").to_dataframe().unstack(level=0)            

                    component_dict[classname][component_description][variable_description] = df
                
                # else:  # TODO: shouldn't this case be uncommented?
                #     df_dict[df_description] = data.rename_axis("space")

            except:
                print(f"'{variable_description}' for '{df_description}' not in xarray_dataset")
                # TODO: these data should not be missing, should they? check to_dict function @Leander

    # set all 1d data (regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"

            if classname != 'Transmission':
                df = xarray_dataset[variable_description].sel(component=df_description).drop("component").to_dataframe().unstack(level=0)            

                component_dict[classname][component_description][variable_description] = df
                # TODO: correctly unstack and rename the dataframe

    return esm_dict, component_dict