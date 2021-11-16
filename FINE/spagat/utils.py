"""Functions to assist spatial aggregation 
"""
import warnings
import time
import numpy as np
import pandas as pd
import xarray as xr

try:
    import geopandas as gpd
except ImportError:
    warnings.warn(
        "The package geopandas is not installed. Spatial aggregation cannot be used without it."
    )


def timer(func):
    """Wrapper around a function to track the time taken by
    the function.

    Parameters
    ----------
    func : Function

    Notes
    -----
    Usage : as a decorator before a function -> @spu.timer
    """

    def f(*args, **kwargs):
        before = time.perf_counter()
        rv = func(*args, **kwargs)
        after = time.perf_counter()
        print(
            "elapsed time for {.__name__}: {:.2f} minutes".format(
                func, (after - before) / 60
            )
        )
        return rv

    return f


def create_gdf(df, geometries, crs=3035, file_path=None, files_name="xr_regions"):
    """Creates a geodataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe which would, among other things, have the region ids/names
    geometries : List[Geometries]
        List of geometries/shapes
    crs : int, optional (default=3035)
        The coordinate reference system in which to create the geodataframe
    file_path : str, optional (default=None)
        If default None, files (called shapefiles) are not saved
    files_name : str, optional (default='xr_regions')
        The name of the saved files
        yes it is plural! -> many files with same name but
        different extensions '.cpg', '.dbf', '.prj', '.shp', '.shx'

    Returns
    -------
    gdf : gpd.GeoDataFrame
        A geodataframe that is created
    """

    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=f"epsg:{crs}")

    if file_path is not None:
        gdf.reset_index(
            drop=True, inplace=True
        ) 
        gdf.to_file(file_path, layer=f"{files_name}")

    return gdf


def create_geom_xarray(shapefile, geom_col_name="geometry", geom_id_col_name="index"):
    """Creates an xr.Dataset with geometry info from the `shapefile`.

    Parameters
    ----------
    shapefile : GeoDataFrame
        The shapefile to be converted
    geom_col_name : str, optional (default="geometry")
        The geomtry column name in `shapefile`
    geom_id_col_name : str, optional (default="index")
        The colum in `shapefile` consisting geom ids

    Returns
    -------
    xr_ds : The xarray dataset holding "geometries", "centroids", "centroid_distances"
    """

    # geometries and their IDs
    geometries = shapefile[geom_col_name]
    geom_ids = shapefile[geom_id_col_name]

    geometries_da = xr.DataArray(
        geometries,
        coords=[geom_ids],
        dims=["space"],
    )

    # centroids
    centroids = pd.Series([geom.centroid for geom in geometries_da.values])
    centroids_da = xr.DataArray(
        centroids,
        coords=[geom_ids],
        dims=["space"],
    )

    # centroid distances
    centroid_dist_da = xr.DataArray(
        np.zeros((len(geom_ids), len(geom_ids))),
        coords=[geom_ids, geom_ids],
        dims=["space", "space_2"],
    )

    for region_id_1 in centroids_da["space"]:
        for region_id_2 in centroids_da["space"]:
            centroid_1 = centroids_da.sel(space=region_id_1).item(0)
            centroid_2 = centroids_da.sel(space=region_id_2).item(0)
            centroid_dist_da.loc[dict(space=region_id_1, space_2=region_id_2)] = (
                centroid_1.distance(centroid_2) / 1e3
            )  # distances in km

    xr_ds = xr.Dataset(
        {
            "geometries": geometries_da,
            "centroids": centroids_da,
            "centroid_distances": centroid_dist_da,
        }
    )
    return xr_ds


def save_shapefile_from_xarray(
    geom_xr, save_path, shp_name="aggregated_regions", crs: int = 3035
):
    """Extracts regions and their geometries from `xarray_dataset`
    and saves to a shapefile.

    Parameters
    ----------
    geom_xr : xr.Dataset
        The xarray dataset holding the geom info
    save_path : str
        path to folder in which to save the shapefile
    shp_name : str, optional (default='aggregated_regions')
        name to be given to the saved files
    crs : int, optional (default=3035)
        coordinate reference system (crs) in which to save the shapefiles
    """

    df = geom_xr.space.to_dataframe()
    geometries = geom_xr.values

    create_gdf(
        df=df, geometries=geometries, crs=crs, file_path=save_path, files_name=shp_name
    )
