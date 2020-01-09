import xarray as xr
import FINE.IOManagement.dictIO as dictio
from FINE import utils
import FINE as fn
import numpy as np
import pandas as pd
import os

import spagat.manager as spm
import spagat.representation as spr
import metis_utils.io_tools as ito
import geopandas as gpd

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
            ds_component[f"2d_{variable_description}"] = df_variable.to_xarray()

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
            ds_component[f"1d_{variable_description}"] = df_variable.to_xarray()

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
            # try:
            df = xarray_dataset[variable_description].sel(component=df_description).drop("component").to_dataframe().unstack(level=0)

            # df.index = df.index.droplevel(level=3)
            
            if len(df.columns) > 1:
                df.columns = df.columns.droplevel(0)

            component_dict[classname][component_description][variable_description] = df.sort_index()

            # except:
            #     print(f"'{variable_description}' for '{df_description}' not in xarray_dataset")
            #     # TODO: these data should not be missing, should they? check to_dict function @Leander


    # set all 2d data (regions, regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"

    #         try:
            if classname == 'Transmission':
                series = xarray_dataset[f"2d_{variable_description}"].sel(component=df_description
                                                            ).drop("component").to_dataframe().stack(level=0)

                series.index = series.index.droplevel(level=2).map('_'.join)

                # print(df_description, variable_description)
                # print(component_dict[classname][component_description][variable_description].head())
                component_dict[classname][component_description][variable_description] = series.sort_index()
                # print(component_dict[classname][component_description][variable_description].head())
                # print()
                # print()
                # print()

            # else:  # TODO: shouldn't this case be uncommented?
            #     df_dict[df_description] = data.rename_axis("space")

    #         except:
    #             print(f"'{variable_description}' for '{df_description}' not in xarray_dataset")
    #             # TODO: these data should not be missing, should they? check to_dict function @Leander


    # set all 1d data (regions)
    for variable_description, description_tuple_list in series_iteration_dict.items():

        # print()
        # print(variable_description)

        for description_tuple in description_tuple_list:
            classname, component_description = description_tuple

            df_description = f"{classname}, {component_description}"

            if classname != 'Transmission':
                # print(xarray_dataset[f"1d_{variable_description}"].sel(component=df_description))
                series = xarray_dataset[f"1d_{variable_description}"].sel(component=df_description
                                                                          ).drop("component").to_dataframe().unstack(level=0)
                series.index = series.index.droplevel(level=0)

                # print(series)

                # print("   ", df_description)
                # print(component_dict[classname][component_description][variable_description].head())

                component_dict[classname][component_description][variable_description] = series.sort_index()
                # print(component_dict[classname][component_description][variable_description].head())
                
                # print()
                # print()
                # print()

    return esm_dict, component_dict

def spatial_aggregation(esM, n_regions, aggregation_function_dict=None,
                        locFilePath=None, aggregatedShapefileFolderPath=None):

    # initialize spagat_manager
    spagat_manager = spm.SpagatManager()
    spagat_manager.sds.xr_dataset = dimensional_data_to_xarray(esM)

    locFilePath = os.path.join('examples/Multi-regional Energy System Workflow', 'InputData', 'SpatialData','ShapeFiles', 'clusteredRegions.shp')
    if locFilePath is not None:
        gdf_regions = gpd.read_file(locFilePath)
        spagat_manager.sds.add_objects(description='gpd_geometries',
                        dimension_list=['space'],
                        object_list=gdf_regions.geometry)
        spr.add_region_centroids(spagat_manager.sds, spatial_dim='space')

    # spatial clustering 
    spagat_manager.grouping(dimension_description='space')

    # representation of the clustered regions
    if aggregation_function_dict is None:
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
    else:
        spagat_manager.aggregation_function_dict = aggregation_function_dict

    spagat_manager.aggregation_function_dict = {f"{dimension}_{key}": value 
                                                for key, value in spagat_manager.aggregation_function_dict.items()
                                            for dimension in ["1d", "2d"]}

    spagat_manager.representation(number_of_regions=n_regions)

    # create aggregated esM instance

    esmDict, compDict = dictio.exportToDict(esM)

    esmDict, compDict = update_dicts_based_on_xarray_dataset(esmDict, compDict, 
                                                                  xarray_dataset=spagat_manager.sds_out.xr_dataset)
    
    esM_aggregated = dictio.importFromDict(esmDict, compDict)

    if aggregatedShapefileFolderPath is not None:

        aggregated_grid_FilePath = os.path.join(aggregatedShapefileFolderPath, 'aggregated_grid.shp')
        spr.create_grid_shapefile(spagat_manager.sds_out, filename=aggregated_grid_FilePath)

        aggregated_regions_FilePath = os.path.join(aggregatedShapefileFolderPath, 'aggregated_regions.shp')

        df_aggregated = spagat_manager.sds_out.xr_dataset.gpd_geometries.to_dataframe(
            ).reset_index(level=0).rename(columns={'space':'index', 'gpd_geometries': 'geometry'})

        gdf_aggregated = gpd.GeoDataFrame(df_aggregated)
        gdf_aggregated.crs = {'init' :'epsg:3035'}
        gdf_aggregated.to_file(aggregated_regions_FilePath)

    return esM_aggregated
