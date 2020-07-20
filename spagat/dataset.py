"""Implementation of the SpagatDataset that contains all spatial data.

"""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString
import os

import spagat.utils as spu
import pathlib

from typing import Dict, List

logger_dataset = logging.getLogger("spagat_dataset")


class SpagatDataset:
    """
    The SpagatDataset (SDS) contains all the spatially-resolved data that is necessary for Energy System Optimization.

    """

    def __init__(self):
        """Initializes the SpagatDataset (sds)."""

        self.xr_dataset = xr.Dataset()

    def add_objects(
        self, description: str, dimension_list: List[str], object_list: List[object]
    ) -> None:
        """Adds a list of arbitrary (?) objects to the sds.
        
        Parameters
        ----------
        description
            description of the objects
        dimension_list
            list of all sds dimensions the objects live in
        object_list
            list of objects that will be added to the sds
            
        """

        # TODO: understandtype of pd.Series(object_list).values and transform object_list without pandas to it

        self.xr_dataset[description] = (dimension_list, pd.Series(object_list).values)

    def add_region_data(self, space: List[object], spatial_dim: str = "space") -> None:
        """Add space coordinates to the dataset
        
        Parameters
        ----------
        space
            coordinates of the space, for example region names
        spatial_dim
            name of the spatial dimension, e.g. 'space' or 'region_ids'

        """
        self.xr_dataset.coords[f"{spatial_dim}"] = space
        self.xr_dataset.coords[f"{spatial_dim}_2"] = space

    def read_dataset(
        self,
        sds_folder_path: str,
        sds_regions_filename: str = "sds_regions.shp",
        sds_xr_dataset_filename: str = "sds_xr_dataset.nc4",
    ) -> None:
        """Reads in both shapefile as well as xarray dataset from a folder to the sds
        
        Parameters
        ----------
        sds_folder_path
            folder that contains the sds data
        sds_regions_filename
            filename for the region shapefile
        sds_xr_dataset_filename
            filename of the netcdf file containing all information apart from the region shapes
        """

        # gets the complete paths
        sds_xr_dataset_path = os.path.join(sds_folder_path, sds_xr_dataset_filename)
        sds_regions_path = os.path.join(sds_folder_path, sds_regions_filename)

        self.xr_dataset = xr.open_dataset(sds_xr_dataset_path)

        gdf_regions = gpd.read_file(sds_regions_path)
        self.add_objects(
            description="gpd_geometries",
            dimension_list=["space"],
            object_list=gdf_regions.geometry,
        )

    def save_sds_regions(self, shape_output_path : str, crs : int = 3035):
        """Save regions and geometries from xr_array to shapefile
        
        Parameters
        ----------
        shape_output_path
            path to which the shapefile shall be saved
        
        crs
            coordinate reference system in which to save the shapefiles
        """

        df = self.xr_dataset.space.to_dataframe()
        geometries = self.xr_dataset.gpd_geometries.values

        spu.create_gdf(
            df=df, geometries=geometries, crs=crs, filepath=shape_output_path
        )

    def save_data(self, sds_output_path : str) -> None:
        """Save all data of the dataset apart from the shapes.

        Parameters
        ----------
        sds_output_path
            folder to which to save the sds data
                
        """

        drop_list = [
            variable
            for variable in ["gpd_geometries", "gpd_centroids", "gk_geometries"]
            if hasattr(self.xr_dataset, variable)
        ]

        if len(drop_list) > 0:
            self.xr_dataset.drop(drop_list).to_netcdf(sds_output_path)
        else:
            self.xr_dataset.to_netcdf(sds_output_path)

    @spu.timer
    def save_sds(
        self,
        sds_folder_path: str,
        sds_region_filename: str ="sds_regions.shp",
        sds_xr_dataset_filename: str ="sds_xr_dataset.nc4",
    ) -> None:
        """Save all data of the sds in a netcdf and a shapefile.

        Parameters
        ----------
        sds_folder_path
            folder to which to save the sds data       
        sds_region_filename
            filename to which to save the shapefile       
        sds_xr_dataset_filename
            filename to which to save the sds data       
        """
        spu.create_dir(sds_folder_path)

        # save geometries
        shape_output_path = os.path.join(sds_folder_path, sds_region_filename)
        self.save_sds_regions(shape_output_path)

        # TODO: maybe also save grid files here

        # save data
        sds_output_path = os.path.join(sds_folder_path, sds_xr_dataset_filename)
        self.save_data(sds_output_path)
