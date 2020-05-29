import xarray as xr
from FINE import utils
import FINE as fn
import numpy as np
import pandas as pd
import os
import warnings

try:
    import spagat.manager as spm
    import spagat.representation as spr
except ImportError:
    warnings.warn('The Spagat python package could not be imported.')

# TODO: declare private functions (and methods) with pre underscore

def generate_iteration_dicts(esm_dict, component_dict):
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

    df_iteration_dict, series_iteration_dict = generate_iteration_dicts(esm_dict, component_dict)

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
                    multi_index_dataframe.index.set_names("space", level=2, inplace=True)
                    

                df_dict[df_description] = multi_index_dataframe # append or concat or so
                                        
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True) # ?

        ds_component = xr.Dataset()
        ds_component[variable_description] = df_variable.sort_index().to_xarray()

        ds = xr.merge([ds, ds_component])

    # get all 2d data (space, space)
    for variable_description, description_tuple_list in series_iteration_dict.items():
        df_dict = {} # dictionary of multiindex data frames that all contain all data for one variable
    
        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}, {component}"

            data = component_dict[classname][component][variable_description]

            if isinstance(data, pd.Series):

                if classname in ['Transmission', 'LinearOptimalPowerFlow']:
                    # TODO: which one of transmission's components are 2d and which 1d or dimensionless

                    df = utils.transform1dSeriesto2dDataFrame(data, locations)
                    multi_index_dataframe = df.stack()
                    multi_index_dataframe.index.set_names(["space", "space_2"], inplace=True)

                    df_dict[df_description] = multi_index_dataframe

        if len(df_dict) > 0:
            df_variable = pd.concat(df_dict)
            df_variable.index.set_names("component", level=0, inplace=True) # ?

            ds_component = xr.Dataset()
            ds_component[f"2d_{variable_description}"] = df_variable.sort_index().to_xarray()

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
    
    df_iteration_dict, series_iteration_dict = generate_iteration_dicts(esm_dict, component_dict)

    # update esm_dict
    esm_dict['locations'] = set(str(value) for value in xarray_dataset.space.values)
    
    # update component_dict
    # set all regional time series (regions, time)
    for variable_description, description_tuple_list in df_iteration_dict.items():
        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"
            df = xarray_dataset[variable_description].sel(component=df_description).drop("component").to_dataframe().unstack(level=2)
            
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

def spatial_aggregation(esM, numberOfRegions, gdfRegions=None, aggregation_function_dict=None, clusterMethod="centroid-based", aggregatedShapefileFolderPath=None, **kwargs):
    """Clusters the spatial data of all components considered in the EnergySystemModel instance and returns a new esM instance with the aggregated data.        
    
    Additional keyword arguments for the SpatialAggregation instance can be added (facilitated by kwargs). 
    
    Please refer to the SPAGAT package documentation for more information.

    :param esM: energy system model instance 
        |br| * the default value is None
    :type esM: energySystemModelInstance

    :param numberOfRegions: states the number of regions into which the spatial data
        should be clustered.
        Note: Please refer to the SPAGAT package documentation of the parameter numberOfRegions for more
        information.
        |br| * the default value is None
    :type numberOfRegions: strictly positive integer, None

    **Default arguments:**

    :param gdfRegions: geodataframe containing the shapes of the regions of the energy system model instance
        |br| * the default value is None
    :type gdfRegions: geopandas.dataframe

    :param aggregatedShapefileFolderPath: indicate the path to the folder were the input and aggregated shapefiles shall be located 
        |br| * the default value is None
    :type aggregatedShapefileFolderPath: string

    :param clusterMethod: states the method which is used in the SPAGAT package for clustering the spatial
        data. Options are for example 'centroid-based'.
        Note: Please refer to the SPAGAT package documentation of the parameter clusterMethod for more information.
        |br| * the default value is 'centroid-based'
    :type clusterMethod: string

    :return: esM_aggregated - esM instance with spatially aggregated data and xarray dataset containing all spatially resolved data
    """

    # initialize spagat_manager
    spagat_manager = spm.SpagatManager()
    esm_dict, component_dict = fn.dictIO.exportToDict(esM)
    spagat_manager.sds.xr_dataset = dimensional_data_to_xarray_dataset(esm_dict, component_dict)

    if gdfRegions is not None:
        spagat_manager.sds.add_objects(description='gpd_geometries',
                                       dimension_list=['space'],
                                       object_list=gdfRegions.geometry)
        spr.add_region_centroids(spagat_manager.sds, spatial_dim='space')

    # spatial clustering 
    spagat_manager.grouping(dimension_description='space')

    # representation of the clustered regions
    if aggregation_function_dict is not None:
        spagat_manager.aggregation_function_dict = aggregation_function_dict
    else:
        spagat_manager.aggregation_function_dict = {'operationRateMax': ('mean', None), # ('weighted mean', 'capacityMax')
                                                'operationRateFix': ('sum', None),
                                                'locationalEligibility': ('bool', None), 
                                                'capacityMax': ('sum', None),
                                                'investPerCapacity': ('sum', None), # ?
                                                'investIfBuilt': ('sum', None), # ? 
                                                'opexPerOperation': ('sum', None), # ?
                                                'opexPerCapacity': ('sum', None), # ?
                                                'opexIfBuilt': ('sum', None), # ?
                                                'interestRate': ('mean', None), # ?
                                                'economicLifetime': ('mean', None), # ?
                                                'capacityFix': ('sum', None),
                                                'losses': ('mean', None), # ?
                                                'distances': ('mean', None), # weighted mean ?
                                                'commodityCost': ('mean', None), # ?
                                                'commodityRevenue': ('mean', None), # ?
                                                'opexPerChargeOperation': ('mean', None),
                                                'opexPerDischargeOperation': ('mean', None),
                                            }

    spagat_manager.aggregation_function_dict = {f"{dimension}_{key}": value 
                                                for key, value in spagat_manager.aggregation_function_dict.items()
                                            for dimension in ["1d", "2d"]}

    spagat_manager.representation(number_of_regions=numberOfRegions)

    # create aggregated esM instance

    esmDict, compDict = update_dicts_based_on_xarray_dataset(esm_dict, component_dict, 
                                                             xarray_dataset=spagat_manager.sds_out.xr_dataset)
    
    esM_aggregated = fn.dictIO.importFromDict(esmDict, compDict)

    if aggregatedShapefileFolderPath is not None:

        # create region shape file
        aggregated_regions_FilePath = os.path.join(aggregatedShapefileFolderPath, 'aggregated_regions.shp')

        spagat_manager.sds_out.save_sds_regions(aggregated_regions_FilePath)

        # df_aggregated = spagat_manager.sds_out.xr_dataset.gpd_geometries.to_dataframe(
        #     ).reset_index(level=0).rename(columns={'space':'index', 'gpd_geometries': 'geometry'})

        # gdf_aggregated = gpd.GeoDataFrame(df_aggregated)
        # gdf_aggregated.crs = {'init' :'epsg:3035'}


        # create grid shape file
        aggregated_grid_FilePath = os.path.join(aggregatedShapefileFolderPath, 'aggregated_grid.shp')
        # spr.create_grid_shapefile(spagat_manager.sds_out, filename=aggregated_grid_FilePath)

    return esM_aggregated

