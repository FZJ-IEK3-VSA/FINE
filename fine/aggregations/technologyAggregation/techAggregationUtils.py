"""
Functions to assist technology aggregation algorithm.
"""
import warnings
import os
import numpy as np
from affine import Affine
import xarray as xr

try:
    from rasterio import features
except ImportError:
    warnings.warn(
        "The package rasterio is not installed. Spatial aggregation cannot be used without it."
    )

try:
    import geopandas as gpd
except ImportError:
    warnings.warn(
        "The package geopandas is not installed. Spatial aggregation cannot be used without it."
    )


def rasterize_geometry(geometry, coords, latitude="y", longitude="x"):
    """
    Given a geometry and geolocations, it masks the geolocations
    such that all the geolocations within the geometry are indicated
    by a 1 and rest are NAs.

    :param geometry: The geometry to be used
    :type geometry: polygon/multiploygon

    :param coords: Holds latitudes and longitudes
    :type coords: Dict-like

    **Default arguments:**

    :param latitude: The description of latitude in `coords`
        |br| * the default value is 'y'
    :type latitude: str

    :param longitude: The description of longitude in `coords`
        |br| * the default value is 'x'
    :type longitude: str

    :returns: raster - A 2d matrix of size latitudes * longitudes
        If a latitude-longitude pair falls within the `geometry` then
        the value at this point in the matrix is 1, otherwise NA
    :rtype: np.ndarray
    """

    # STEP 1. Get the affine transformation
    lat = np.asarray(coords[latitude])
    lon = np.asarray(coords[longitude])

    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    transform = trans * scale

    # STEP 2. Get the raster mask
    out_shape = (len(lat), len(lon))

    raster = features.rasterize(
        [geometry], out_shape=out_shape, fill=np.nan, transform=transform, dtype=float
    )

    return raster


def rasterize_xr_ds(
    gridded_RE_ds,
    CRS_attr,
    shp_file,
    index_col="region_ids",
    geometry_col="geometry",
    longitude="x",
    latitude="y",
):
    """
    For each geometry in the specified `shp_file`, a binary mask
    is added to the `gridded_RE_ds`, so that subsetting the data
    for each region is possible.

    :param gridded_RE_ds: Either the path to the dataset or the read-in xr.Dataset
        2 mandatory dimensions in this data - `latitude` and `longitude`
    :type gridded_RE_ds: str/xr.Dataset

    :param CRS_attr: The attribute in `gridded_RE_ds` that holds its
        Coordinate Reference System (CRS) information
    :type CRS_attr: str

    :param shp_file: Either the path to the shapefile or the read-in shapefile
        that should be added to `gridded_RE_ds`
    :type shp_file: str/GeoDataFrame

    **Default arguments:**

    :param index_col: The column in `shp_file` that needs to be taken as location-index in `gridded_RE_ds`
        |br| * the default value is 'region_ids'
    :type index_col: str

    :param geometry_col: The column in `shp_file` that holds geometries
        |br| * the default value is 'geometry'
    :type geometry_col: str

    :param longitude: The dimension name in `gridded_RE_ds` that corresponds to longitude
        |br| * the default value is 'x'
    :type longitude: str

    :param latitude: The dimension name in `gridded_RE_ds` that corresponds to latitude
        |br| * the default value is 'y'
    :type latitude: str

    :returns: rasterized_RE_ds - dataset with

        - Additional dimension with name `index_col`
        - Additional variable with name 'rasters' and values as rasters
          corresponding to each geometry in `shp_file`

    :rtype: xr.Dataset
    """

    # STEP 1. Read in the files
    ## gridded_RE_ds
    if isinstance(gridded_RE_ds, str):
        try:
            gridded_RE_ds = xr.open_dataset(gridded_RE_ds)
        except:
            raise FileNotFoundError("The gridded_RE_ds path specified is not valid")

    elif not isinstance(gridded_RE_ds, xr.Dataset):
        raise TypeError(
            "gridded_RE_ds must either be a path to a netcdf file or xarray dataset"
        )

    ## shp_file
    if isinstance(shp_file, str):
        if not os.path.isfile(shp_file):
            raise FileNotFoundError("The shp_file path specified is not valid")
        else:
            shp_file = gpd.read_file(shp_file)

    elif not isinstance(shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError(
            "shp_file must either be a path to a shapefile or a geopandas dataframe"
        )

    # STEP 2. Match the CRS of shapefile to that of the dataset
    shp_file = shp_file.to_crs({"init": gridded_RE_ds.attrs[CRS_attr]})

    # STEP 3. rasterize each geometry and add it to new data_var "rasters"

    region_geometries = shp_file[geometry_col]
    region_indices = shp_file[index_col]

    rasterized_RE_ds = gridded_RE_ds.expand_dims({"region_ids": region_indices})

    coords = rasterized_RE_ds.coords

    rasterized_RE_ds["rasters"] = (
        ["region_ids", latitude, longitude],
        [
            rasterize_geometry(geometry, coords, longitude=longitude, latitude=latitude)
            for geometry in region_geometries
        ],
    )

    return rasterized_RE_ds
