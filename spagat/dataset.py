import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString
import os 

import geokit as gk
import metis_utils.io_tools as ito
import metis_utils.time_tools as tto

logger_dataset = logging.getLogger('spagat_dataset')


class SpagatDataSet:
    # TODO: maybe inherit xr.Dataset => SpagatDataSet(xr.Dataset())
    """The SpagatDataset (SDS) contains all the data that is necessary for Energy System Optimization.
    Therefore, it contains:
    - capacity factors time series (raster or per region)
    - demand time series (raster or per region)
    - grid data (incidence, capacity, ...)
    - region data (shapefiles, ...)
    - ...

    Idea:
    - Spagat initializes with a pre-aggregation sds
    - the aggregated data is written to a post-aggregation sds
    - both files are saved as netcdf files (if possible)
    - maybe it makes sense to also keep interim-aggregation sds,
      when iterative procedure is implemented
    """

    # dimensions:
    # space: region ids
    # time: time steps
    # technology: technology-dependent

    # data:
    # 1d:
    # - region shapes
    # - region centroids, area, ...
    #
    # 2d:
    # - time series for each region
    # - grid incidence, capacity, ...
    # -

    def __init__(self):
        self.xr_dataset = xr.Dataset()
        # self.xr_ds_res = None
        # TODO: add regions etc. already here?

    def add_objects(self, description, dimension_list, object_list):

        # TODO: understandtype of pd.Series(object_list).values and transform object_list without pandas to it

        self.xr_dataset[description] = (dimension_list, pd.Series(object_list).values)

        # self.xr_dataset[description] = (('region_ids', 'region_ids_2'), grid_data)

    def add_region_data(self, region_ids):
        """Add the region_ids as coordinates to the dataset"""
        self.xr_dataset.coords['region_ids'] = region_ids
        self.xr_dataset.coords['region_ids_2'] = region_ids

    def read_dataset(self, sds_folder,
                     sds_regions_filename='sds_regions.shp', sds_xr_dataset_filename='sds_xr_dataset.nc4'):
        '''Reads in both shapefile as well as xarray dataset from a folder to the sds'''

        #gets the complete paths
        sds_xr_dataset_path = sds_folder_path / ds_xr_dataset_filename
        sds_regions_path = sds_folder_path / sds_regions_filename

        self.xr_dataset = xr.open_dataset(sds_xr_dataset_path)

        gdf_regions = gpd.read_file(sds_regions_path)
        self.add_objects(description='gpd_geometries',
                         dimension_list=['region_ids'],
                         object_list=gdf_regions.geometry)

    def save_sds_regions(self, shape_output_path, crs=3035):
        """Save regions and geometries from xr_array to shapefile"""

        df = self.xr_dataset.region_ids.to_dataframe()
        geometries = self.xr_dataset.gpd_geometries.values

        ito.create_gdf(df=df, geometries=geometries, crs=crs, filepath=shape_output_path)

    def save_data(self, sds_output_path):

        drop_list = [variable for variable in ['gpd_geometries', 'gpd_centroids', 'gk_geometries']
                     if hasattr(self.xr_dataset, variable)]

        if len(drop_list) > 0:
            self.xr_dataset.drop(drop_list).to_netcdf(sds_output_path)
        else:
            self.xr_dataset.to_netcdf(sds_output_path)

    @tto.timer
    def save_sds(self, sds_folder_path,
                 sds_region_filename='sds_regions.shp', sds_xr_dataset_filename='sds_xr_dataset.nc4'):
        ito.create_dir(sds_folder_path)

        # save geometries
        shape_output_path =  os.path.join(sds_folder_path, sds_region_filename)  
        self.save_sds_regions(shape_output_path)

        # TODO: maybe also save grid files here

        # save data
        sds_output_path = os.path.join(sds_folder_path, sds_xr_dataset_filename)  
        self.save_data(sds_output_path)
