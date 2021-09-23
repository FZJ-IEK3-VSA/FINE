
import os 
import logging

import xarray as xr 
import geopandas as gpd

import FINE.spagat.utils as spu
import FINE.spagat.grouping as spg
import FINE.spagat.representation as spr 

from FINE.IOManagement import xarrayIO_spagat as xrIO

logger_spagat = logging.getLogger('spatial_aggregation')

def perform_spatial_aggregation(xr_dataset,
                            shapefile, 
                            grouping_mode='parameter_based', 
                            nRegionsForRepresentation=2,
                            aggregatedResultsPath=None,
                            **kwargs):
    """Performs spatial grouping of regions (by calling the functions in grouping.py) 
    and then representation of the data within each region group (by calling functions 
    in representation.py).

    Parameters
    ----------
    xr_dataset : str/xr.Dataset 
        Either the path to the dataset or the read-in xr.Dataset
        - Dimensions in this data - 'component', 'space', 'space_2', and 'time' 
    shapefile : str/GeoDataFrame
        Either the path to the shapefile or the read-in shapefile
    grouping_mode : {'parameter_based', 'string_based', 'distance_based'}, optional
        Defines how to spatially group the regions. Refer to grouping.py for more 
        information.
    nRegionsForRepresentation : strictly positive int, optional (default=2)
        Indicates the number of regions chosen for representation of data. 
        If 'distance_based' or 'parameter_based' is chosen for `grouping_mode`, grouping 
        is performed for 1 to number of regions initially present in the `xr_dataset`. 
        Here, the number of groups finally chosen for representation of data is to be 
        specified. This parameter is irrelevant if `grouping_mode` is 'string_based'.
    aggregatedResultsPath : str, optional (default=None)
        Indicates path to which the aggregated results should be saved. 
        If None, results are not saved. 
    
    Additional keyword arguments can be added passed via kwargs.

    Returns
    -------
    aggregated_xr_dataset : The xarray dataset holding aggregated data
    """

    #STEP 1. Read and check shapefile 
    if isinstance(shapefile, str): 
        if not os.path.isfile(shapefile):
            raise FileNotFoundError("The shapefile path specified is not valid")
        else:
            shapefile = gpd.read_file(shapefile)
            
    elif not isinstance(shapefile, gpd.geodataframe.GeoDataFrame):
        raise TypeError("shapefile must either be a path to a shapefile or a geopandas dataframe")

    n_geometries = len(shapefile.index)
    if n_geometries < 2: 
        raise ValueError("Atleast two regions must be present in shapefile and data \
            in order to perform spatial aggregation")
    
    if n_geometries < nRegionsForRepresentation:
        raise ValueError(f"{n_geometries} regions cannot be reduced to {nRegionsForRepresentation} \
            regions. Please provide a valid number for nRegionsForRepresentation")

    #STEP 2. Read xr_dataset
    if isinstance(xr_dataset, str): 
        try:
            xr_dataset = xr.open_dataset(xr_dataset)
        except:
            raise FileNotFoundError("The xr_dataset path specified is not valid")

    #STEP 3. Add shapefile information to xr_dataset
    xr_dataset = spu.add_objects_to_xarray(xr_dataset,
                                        description='gpd_geometries',
                                        dimension_list=['space'],
                                        object_list=shapefile.geometry)

    xr_dataset = spu.add_region_centroids_to_xarray(xr_dataset) 
    xr_dataset = spu.add_centroid_distances_to_xarray(xr_dataset)
    
    #STEP 4. Spatial grouping
    if grouping_mode == 'string_based':

        logger_spagat.info('Performing string-based grouping on the regions')
        
        locations = xr_dataset.space.values
        aggregation_dict = spg.perform_string_based_grouping(locations)

    elif grouping_mode == 'distance_based':

        save_path = kwargs.get('save_path', None) 
        fig_name = kwargs.get('fig_name', None)
        verbose = kwargs.get('verbose', False)
            
        logger_spagat.info(f'Performing distance-based grouping on the regions')

        aggregation_dict = spg.perform_distance_based_grouping(xr_dataset,
                                                            save_path,
                                                            fig_name, 
                                                            verbose)

    elif grouping_mode == 'parameter_based':

        linkage = kwargs.get('linkage', 'complete') 
        weights = kwargs.get('weights', None) 

        logger_spagat.info(f'Performing parameter-based grouping on the regions.')

        aggregation_dict = spg.perform_parameter_based_grouping(xr_dataset, 
                                                                linkage,
                                                                weights)

    else:
        raise ValueError(f'The grouping mode {grouping_mode} is not valid. Please choosen one of \
        the valid grouping mode among: string_based, distance_based, parameter_based')

    
    #STEP 5. Representation of the new regions
    if grouping_mode == 'string_based':
        sub_to_sup_region_id_dict = aggregation_dict #INFO: Not a nested dict for different #regions
    else:
        sub_to_sup_region_id_dict = aggregation_dict.get(nRegionsForRepresentation)
    
    if 'aggregation_function_dict' not in kwargs:
        aggregation_function_dict = None
    else: 
        logger_spagat.info('aggregation_function_dict found in kwargs')
        aggregation_function_dict = kwargs.get('aggregation_function_dict')
    
    aggregated_xr_dataset = spr.aggregate_based_on_sub_to_sup_region_id_dict(xr_dataset,
                                                                sub_to_sup_region_id_dict,
                                                                aggregation_function_dict) 
    
    #STEP 6. Save shapefiles and aggregated xarray dataset if user chooses
    if aggregatedResultsPath is not None:   
        # get file names 
        shp_name = kwargs.get('shp_name', 'aggregated_regions') 
        aggregated_xr_filename = kwargs.get('aggregated_xr_filename', 'aggregated_xr_dataset.nc4')
        
        # save shapefiles 
        spu.save_shapefile_from_xarray(aggregated_xr_dataset,
                                        aggregatedResultsPath,
                                        shp_name)

        # remove geometry related data vars from aggregated xarray dataset as these cannot be saved 
        aggregated_xr_dataset = aggregated_xr_dataset.drop_vars(['gpd_geometries', 'gpd_centroids', 'centroid_distances'])

        # save aggregated xarray dataset 
        file_name_with_path = os.path.join(aggregatedResultsPath, aggregated_xr_filename)
        xrIO.saveNetcdfFile(aggregated_xr_dataset, file_name_with_path)

    return aggregated_xr_dataset