import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from FINE import utils
import FINE as fn
try:
    import FINE.spagat.dataset as spd
    import FINE.spagat.grouping as spg
    import FINE.spagat.representation as spr 
except ImportError:
    warnings.warn('The Spagat python package could not be imported.')


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

def spatial_aggregation(esM, 
                        gdfRegions, 
                        grouping_mode = 'string_based', # options -'string_based', distance_based, all_variable_based
                        nRegionsForRepresentation = 2,
                        aggregatedResultsPath=None,
                        **kwargs): 
    #STEP 1. Obtain xr dataset from esM 
    sds = spd.SpagatDataset()
    esm_dict, comp_dict = fn.dictIO.exportToDict(esM)
    sds.xr_dataset = dimensional_data_to_xarray_dataset(esm_dict, comp_dict)
    
    #STEP 2. Add shapefile information to sds
    sds.add_objects(description='gpd_geometries',
                    dimension_list=['space'],
                    object_list=gdfRegions.geometry)
    spr.add_region_centroids(sds, spatial_dim='space')
    
    #STEP 3. Spatial grouping
    if grouping_mode == 'string_based':
        print('Performing string-based clustering on the regions')
        locations = sds.xr_dataset.space.values
        aggregation_dict = spg.string_based_clustering(locations)

    elif grouping_mode == 'distance_based':
        agg_mode = kwargs.get('agg_mode', 'sklearn_hierarchical')  #TODO: some of the parameters and their default values are repeating in
        print(f'Performing distance-based clustering on the regions. Clustering mode: {agg_mode}')

        dimension_description = kwargs.get('dimension_description', 'space') # 'all_variable_based'. Maybe make it common 
        ax_illustration = kwargs.get('ax_illustration', None) 
        save_path = kwargs.get('save_path', None) 
        fig_name = kwargs.get('fig_name', None)
        verbose = kwargs.get('verbose', False)

        aggregation_dict = spg.distance_based_clustering(sds, 
                                                        agg_mode, 
                                                        dimension_description, 
                                                        ax_illustration, 
                                                        save_path,
                                                        fig_name, 
                                                        verbose)

    elif grouping_mode == 'all_variable_based':
        agg_mode = kwargs.get('agg_mode', 'sklearn_hierarchical') 
        print(f'Performing all variable-based clustering on the regions. Clustering mode: {agg_mode}')

        dimension_description = kwargs.get('dimension_description', 'space') 
        ax_illustration = kwargs.get('ax_illustration', None) 
        save_path = kwargs.get('save_path', None) 
        fig_name = kwargs.get('fig_name', None)
        verbose = kwargs.get('verbose', False)
        weighting = kwargs.get('weighting', None)

        aggregation_dict = spg.all_variable_based_clustering(sds, 
                                                            agg_mode,
                                                            dimension_description,
                                                            ax_illustration, 
                                                            save_path, 
                                                            fig_name,  
                                                            verbose,
                                                            weighting)
    
    #STEP 4. Representation of the new regions
    if grouping_mode == 'string_based':
        sub_to_sup_region_id_dict = aggregation_dict #INFO: Not a nested dict for different #regions
    else:
        sub_to_sup_region_id_dict = aggregation_dict[nRegionsForRepresentation]
    
    if 'aggregation_function_dict' not in kwargs:
        aggregation_function_dict = None
    else: 
        print('aggregation_function_dict found in kwargs')
        aggregation_function_dict = kwargs.get('aggregation_function_dict')
        if aggregation_function_dict != None:
            aggregation_function_dict = {f"{dimension}_{key}": value      #INFO: xarray dataset has prefix 1d_ and 2d_. Therefore, in order to match that,the prefix is added here for each variable  
                                            for key, value in aggregation_function_dict.items()
                                                for dimension in ["1d", "2d"]}
    
    spatial_dim = kwargs.get('spatial_dim', 'space')
    time_dim = kwargs.get('time_dim', 'TimeStep')
    
    aggregated_sds = spr.aggregate_based_on_sub_to_sup_region_id_dict(sds,
                                                                sub_to_sup_region_id_dict,
                                                                aggregation_function_dict,
                                                                spatial_dim,        #TODO: check how useful these parameters would be, 
                                                                time_dim)       # if you decide to keep them, make it uniform, 
                                                                                            # ex.: in grouping functions, spatial_dim is called dimension_description
    
    #STEP 5. Obtain aggregated esM
    new_esm_dict, new_comp_dict = update_dicts_based_on_xarray_dataset(esm_dict, comp_dict, 
                                                                  xarray_dataset=aggregated_sds.xr_dataset)
    
    aggregated_esM = fn.dictIO.importFromDict(new_esm_dict, new_comp_dict)
    
    #STEP 6. Save shapefiles and aggregated data if user chooses
    if aggregatedResultsPath is not None:   #TODO: test if they are saved as intented 
        sds_region_filename = kwargs.get('sds_region_filename', 'sds_regions.shp') 
        sds_xr_dataset_filename = kwargs.get('sds_xr_dataset_filename', 'sds_xr_dataset.nc4')
        
        aggregated_sds.save_sds(aggregatedResultsPath,
                    sds_region_filename,
                    sds_xr_dataset_filename)
        

    return aggregated_esM

#TODO: maybe copy and adapt the docstring and then delete this function 
# def spatial_aggregation(esM, numberOfRegions, gdfRegions=None, aggregation_function_dict=None, clusterMethod="centroid-based", aggregatedShapefileFolderPath=None, **kwargs): #FIXME: **kwargs are not passed to any function in spagat manager. 
#     """Clusters the spatial data of all components considered in the EnergySystemModel instance and returns a new esM instance with the aggregated data.        
    
#     Additional keyword arguments for the SpatialAggregation instance can be added (facilitated by kwargs). 
    
#     Please refer to the SPAGAT package documentation for more information.

#     :param esM: energy system model instance 
#         |br| * the default value is None
#     :type esM: energySystemModelInstance

#     :param numberOfRegions: states the number of regions into which the spatial data
#         should be clustered.
#         Note: Please refer to the SPAGAT package documentation of the parameter numberOfRegions for more
#         information.
#         |br| * the default value is None
#     :type numberOfRegions: strictly positive integer, None

#     **Default arguments:**

#     :param gdfRegions: geodataframe containing the shapes of the regions of the energy system model instance
#         |br| * the default value is None
#     :type gdfRegions: geopandas.dataframe

#     :param aggregatedShapefileFolderPath: indicate the path to the folder were the input and aggregated shapefiles shall be located 
#         |br| * the default value is None
#     :type aggregatedShapefileFolderPath: string

#     :param clusterMethod: states the method which is used in the SPAGAT package for clustering the spatial
#         data. Options are for example 'centroid-based'.
#         Note: Please refer to the SPAGAT package documentation of the parameter clusterMethod for more information.
#         |br| * the default value is 'centroid-based'
#     :type clusterMethod: string

#     :return: esM_aggregated - esM instance with spatially aggregated data and xarray dataset containing all spatially resolved data
#     """

#     # initialize spagat_manager
#     spagat_manager = spm.SpagatManager()
#     esm_dict, component_dict = fn.dictIO.exportToDict(esM)
#     spagat_manager.sds.xr_dataset = dimensional_data_to_xarray_dataset(esm_dict, component_dict)

#     if gdfRegions is not None:
#         spagat_manager.sds.add_objects(description='gpd_geometries',
#                                        dimension_list=['space'],
#                                        object_list=gdfRegions.geometry)
#         spr.add_region_centroids(spagat_manager.sds, spatial_dim='space')

#     # spatial clustering 
#     spagat_manager.grouping(dimension_description='space')    #FIXME: clusterMethod is not used here, nor is there an option in manager.grouping()
#                                                               # also, centroid-based in not a method for grouping at all -> distance_based_clustering is 
#                                                               #TODO: fix the grouping function in grouping.manager and use it here more apporpriately
#     # representation of the clustered regions
#     if aggregation_function_dict is not None:
#         spagat_manager.aggregation_function_dict = aggregation_function_dict
#     else:
#         spagat_manager.aggregation_function_dict = {'operationRateMax': ('mean', None), # ('weighted mean', 'capacityMax')
#                                                 'operationRateFix': ('sum', None),
#                                                 'locationalEligibility': ('bool', None), 
#                                                 'capacityMax': ('sum', None),
#                                                 'investPerCapacity': ('sum', None), # ?
#                                                 'investIfBuilt': ('sum', None), # ? 
#                                                 'opexPerOperation': ('sum', None), # ?
#                                                 'opexPerCapacity': ('sum', None), # ?
#                                                 'opexIfBuilt': ('sum', None), # ?
#                                                 'interestRate': ('mean', None), # ?
#                                                 'economicLifetime': ('mean', None), # ?
#                                                 'capacityFix': ('sum', None),
#                                                 'losses': ('mean', None), # ?
#                                                 'distances': ('mean', None), # weighted mean ?
#                                                 'commodityCost': ('mean', None), # ?
#                                                 'commodityRevenue': ('mean', None), # ?
#                                                 'opexPerChargeOperation': ('mean', None),
#                                                 'opexPerDischargeOperation': ('mean', None),
#                                             }

#     spagat_manager.aggregation_function_dict = {f"{dimension}_{key}": value    #TODO: find out why this is required. 
#                                                 for key, value in spagat_manager.aggregation_function_dict.items()
#                                             for dimension in ["1d", "2d"]}

#     spagat_manager.representation(number_of_regions=numberOfRegions)

#     # create aggregated esM instance

#     esmDict, compDict = update_dicts_based_on_xarray_dataset(esm_dict, component_dict, 
#                                                              xarray_dataset=spagat_manager.sds_out.xr_dataset)
    
#     esM_aggregated = fn.dictIO.importFromDict(esmDict, compDict)

#     if aggregatedShapefileFolderPath is not None:

#         # create region shape file
#         aggregated_regions_FilePath = os.path.join(aggregatedShapefileFolderPath, 'aggregated_regions.shp')

#         spagat_manager.sds_out.save_sds_regions(aggregated_regions_FilePath)

#         # df_aggregated = spagat_manager.sds_out.xr_dataset.gpd_geometries.to_dataframe(
#         #     ).reset_index(level=0).rename(columns={'space':'index', 'gpd_geometries': 'geometry'})

#         # gdf_aggregated = gpd.GeoDataFrame(df_aggregated)
#         # gdf_aggregated.crs = {'init' :'epsg:3035'}


#         # create grid shape file
#         aggregated_grid_FilePath = os.path.join(aggregatedShapefileFolderPath, 'aggregated_grid.shp')
#         # spr.create_grid_shapefile(spagat_manager.sds_out, filename=aggregated_grid_FilePath)

#     return esM_aggregated

