import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import LineString


def plt_savefig(fig=None, save_name="test", path=None):
    """Save a figure in .png format.

    Parameters
    ----------
    fig : Figure, optional (default=None)
        If None, it get the current figure and saves it
    save_name : str, optional (default="test")
        Name of the figure
    path : str, optional (default=None)
        The path to which to save the figure.
        If default None, it is saved in the current working directory.
    """
    if fig is None:
        fig = plt.gcf()

    if path is not None:
        save_name = os.path.join(path, save_name)

    plt.savefig(f"{save_name}.png", format="png", bbox_inches="tight", dpi=200)


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
        )  # NOTE: pandas different versions behave differently here!
        gdf.to_file(file_path, layer=f"{files_name}")

    return gdf


def create_geom_xarray(shapefile, 
                        geom_col_name='geometry', 
                        geom_id_col_name='index'):
    #TODO: doc string 
    
    # geometries and their IDs
    geometries = shapefile[geom_col_name]
    geom_ids = shapefile[geom_id_col_name]

    geometries_da = xr.DataArray(
                            geometries,
                            coords=[geom_ids],
                            dims=["space"],
                        )

    #centroids 
    centroids = pd.Series(
        [geom.centroid for geom in geometries_da.values]
    )
    centroids_da = xr.DataArray(
                            centroids,
                            coords=[geom_ids],
                            dims=["space"],
                        )

    # centroid distances
    centroid_dist_da = xr.DataArray(
        np.zeros((len(geom_ids), len(geom_ids))), 
        coords=[geom_ids, geom_ids], 
        dims=["space", "space_2"]
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
    xarray_dataset, save_path, shp_name="aggregated_regions", crs: int = 3035
):
    """Extracts regions and their geometries from `xarray_dataset`
    and saves to a shapefile.

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        the xarray dataset from which regions and their geometries
        are to be obtained
    save_path : str
        path to folder in which to save the shapefile
    shp_name : str, optional (default='aggregated_regions')
        name to be given to the saved files
    crs
        coordinate reference system (crs) in which to save the shapefiles
    """

    df = xarray_dataset.space.to_dataframe()
    geometries = xarray_dataset.gpd_geometries.values

    create_gdf(
        df=df, geometries=geometries, crs=crs, file_path=save_path, files_name=shp_name
    )

