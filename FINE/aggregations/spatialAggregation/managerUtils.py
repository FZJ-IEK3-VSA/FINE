"""
Functions to assist spatial aggregation 
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr

try:
    import geopandas as gpd
except ImportError:
    warnings.warn(
        "The package geopandas is not installed. Spatial aggregation cannot be used without it."
    )


def create_gdf(df, geometries, crs=3035, file_path=None, files_name="xr_regions"):
    """
    Creates a geodataframe.

    :param df: The dataframe which would, among other things, have the region ids/names
    :type df: pd.DataFrame

    :param geometries: List of geometries/shapes
    :type geometries: List[Geometries]

    **Default arguments:**

    :param crs: The coordinate reference system in which to create the geodataframe
        |br| * the default value is 3035
    :type crs: int

    :param file_path: If default None, files (called shapefiles) are not saved
        |br| * the default value is None
    :type file_path: str

    :param files_name: The name of the saved files
        yes it is plural! -> many files with same name but
        different extensions '.cpg', '.dbf', '.prj', '.shp', '.shx'
        |br| * the default value is 'xr_regions'
    :type files_name: str

    :returns: gdf - A geodataframe that is created
    :rtype: gpd.GeoDataFrame
    """

    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs=f"epsg:{crs}")

    if file_path is not None:
        gdf.reset_index(drop=True, inplace=True)
        gdf.to_file(file_path, layer=f"{files_name}")

    return gdf


def create_geom_xarray(
    shapefile, geom_col_name="geometry", geom_id_col_name="index", add_centroids=True
):
    """
    Creates an xr.Dataset with geometry info from the `shapefile`.

    :param shapefile: The shapefile to be converted
    :type shapefile: gpd.GeoDataFrame

    **Default arguments:**

    :param geom_col_name: The geomtry column name in `shapefile`
        |br| * the default value is 'geometry'
    :type geom_col_name: str

    :param geom_id_col_name: The colum in `shapefile` consisting geom ids
        |br| * the default value is 'index'
    :type geom_id_col_name: str

    :param add_centroids: Indicates whether region centroids and centroid distances should be
        added to the resulting geom xarray
        |br| * the default value is True
    :type add_centroids: bool

    :returns: xr_ds - The xarray dataset holding 'geometries', 'centroids', 'centroid_distances'
    :rtype: xr.Dataset
    """

    # geometries and their IDs
    geometries = shapefile[geom_col_name]
    geom_ids = shapefile[geom_id_col_name]

    geometries_da = xr.DataArray(
        geometries,
        coords=[geom_ids],
        dims=["space"],
    )

    xr_ds = xr.Dataset({"geometries": geometries_da})

    if add_centroids:
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

        xr_ds.update(
            {
                "centroids": centroids_da,
                "centroid_distances": centroid_dist_da,
            }
        )

    return xr_ds


def save_shapefile_from_xarray(
    geom_xr, save_path, shp_name="aggregated_regions", crs: int = 3035
):
    """
    Extracts regions and their geometries from `xarray_dataset`
    and saves to a shapefile.

    :param geom_xr: The xarray dataset holding the geom info
    :type geom_xr: xr.Dataset

    :param save_path: path to folder in which to save the shapefile
    :type save_path: str

    **Default arguments:**

    :param shp_name: name to be given to the saved files
        |br| * the default value is 'aggregated_regions'
    :type shp_name: str

    :param crs: coordinate reference system (crs) in which to save the shapefiles
        |br| * the default value is 3035
    :type crs: int
    """

    df = geom_xr.space.to_dataframe()
    geometries = geom_xr.values

    create_gdf(
        df=df, geometries=geometries, crs=crs, file_path=save_path, files_name=shp_name
    )
