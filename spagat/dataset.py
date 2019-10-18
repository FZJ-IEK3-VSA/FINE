import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString

import geokit as gk
import tsa_lib.io_tools as ito
import tsa_lib.time_tools as tto

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

    def add_region_data(self, regions):
        """Add the region_ids as coordinates to the dataset"""
        self.xr_dataset.coords['regions'] = regions
        self.xr_dataset.coords['regions_2'] = regions

    def read_sds(self, sds_folder,
                 sds_regions_filename='sds_regions.shp', sds_xr_dataset_filename='sds_xr_dataset.nc4'):
        '''Reads in both shapefile as well as xarray dataset from a folder to the sds'''
        self.xr_dataset = xr.open_dataset(sds_folder / sds_xr_dataset_filename)

        # TODO: remove the following quickfix
        self.xr_dataset = self.xr_dataset.rename({'region_ids': 'regions'})
        self.xr_dataset = self.xr_dataset.rename({'region_ids_2': 'regions_2'})

        gdf_regions = gpd.read_file(sds_folder / sds_regions_filename)
        self.add_objects(description='gpd_geometries',
                         dimension_list=['regions'],
                         object_list=gdf_regions.geometry)

    def save_sds_regions(self, shape_output_path, crs=3035):
        """Save regions and geometries from xr_array to shapefile"""

        df = self.xr_dataset.regions.to_dataframe()
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
    def save_sds(self, sds_folder,
                 sds_region_filename='sds_regions.shp', sds_xr_dataset_filename='sds_xr_dataset.nc4'):
        ito.create_dir(sds_folder)

        # save geometries
        shape_output_path = sds_folder / sds_region_filename
        self.save_sds_regions(shape_output_path)

        # TODO: maybe also save grid files here

        # save data
        sds_output_path = sds_folder / sds_xr_dataset_filename
        self.save_data(sds_output_path)
