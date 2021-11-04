import os
import logging

import geopandas as gpd

import FINE.spagat.utils as spu
import FINE.spagat.grouping as spg
import FINE.spagat.representation as spr
import FINE.IOManagement.xarrayIO as xrIO

logger_spagat = logging.getLogger("spatial_aggregation")


def perform_spatial_aggregation(
    xr_datasets,
    shapefile,
    grouping_mode="parameter_based",
    n_groups=3,
    aggregatedResultsPath=None,
    **kwargs,
):
    """Performs spatial grouping of regions (by calling the functions in grouping.py)
    and then representation of the data within each region group (by calling functions
    in representation.py).

    Parameters
    ----------
    xr_dataset : str/Dict[str, xr.Dataset]
        Either the path to .netCDF file or the read-in dictionary of xarray datasets 
        - Dimensions in this data - 'time', 'space', 'space_2'
    shapefile : str/GeoDataFrame
        Either the path to the shapefile or the read-in shapefile
    grouping_mode : {'parameter_based', 'string_based', 'distance_based'}, optional
        Defines how to spatially group the regions. Refer to grouping.py for more
        information.
    n_groups : strictly positive int, optional (default=3)
        The number of region groups to be formed from the original region set.
        This parameter is irrelevant if `grouping_mode` is 'string_based'.
    aggregatedResultsPath : str, optional (default=None)
        Indicates path to which the aggregated results should be saved.
        If None, results are not saved.

    Additional keyword arguments can be added passed via kwargs.

    Returns
    -------
    aggregated_xr_dataset : The xarray dataset holding aggregated data
    """

    # STEP 1. Read and check shapefile
    if isinstance(shapefile, str):
        if not os.path.isfile(shapefile):
            raise FileNotFoundError("The shapefile path specified is not valid")
        else:
            shapefile = gpd.read_file(shapefile)

    elif not isinstance(shapefile, gpd.geodataframe.GeoDataFrame):
        raise TypeError(
            "shapefile must either be a path to a shapefile or a geopandas dataframe"
        )

    n_geometries = len(shapefile.index)
    if n_geometries < 2:
        raise ValueError(
            "Atleast two regions must be present in shapefile and data \
            in order to perform spatial aggregation"
        )

    if n_geometries < n_groups:
        raise ValueError(
            f"{n_geometries} regions cannot be reduced to {n_groups} \
            regions. Please provide a valid number for n_groups"
        )

    # STEP 2. Read xr_dataset
    if isinstance(xr_datasets, str):
        try:
            xr_datasets = xrIO.readNetCDFToDatasets(filePath=xr_datasets)
        except:
            raise FileNotFoundError("The xr_dataset path specified is not valid")

    # STEP 3. Add geometries to xr_dataset
    geom_col_name = kwargs.get("geom_col_name", 'geometry')
    geom_id_col_name = kwargs.get("geom_id_col_name", 'index')

    geom_xr = spu.create_geom_xarray(shapefile, geom_col_name, geom_id_col_name)

    xr_datasets['Geometry'] = geom_xr

    # STEP 4. Spatial grouping
    if grouping_mode == "string_based":

        separator = kwargs.get("separator", None)
        position = kwargs.get("position", None)

        locations = geom_xr.space.values

        logger_spagat.info("Performing string-based grouping on the regions")

        aggregation_dict = spg.perform_string_based_grouping(
            locations, separator, position
        )

    elif grouping_mode == "distance_based":

        logger_spagat.info(f"Performing distance-based grouping on the regions")

        aggregation_dict = spg.perform_distance_based_grouping(geom_xr, n_groups)

    elif grouping_mode == "parameter_based":

        weights = kwargs.get("weights", None)
        aggregation_method = kwargs.get("aggregation_method", "kmedoids_contiguity")
        solver = kwargs.get("solver", "gurobi")

        logger_spagat.info(f"Performing parameter-based grouping on the regions.")

        aggregation_dict = spg.perform_parameter_based_grouping(
            xr_datasets,
            n_groups=n_groups,
            aggregation_method=aggregation_method,
            weights=weights,
            solver=solver,
        )

    else:
        raise ValueError(
            f"The grouping mode {grouping_mode} is not valid. Please choose one of \
        the valid grouping mode among: string_based, distance_based, parameter_based"
        )

    # STEP 5. Representation of the new regions
    aggregation_function_dict = kwargs.get("aggregation_function_dict", None)

    aggregated_xr_dataset = spr.aggregate_based_on_sub_to_sup_region_id_dict(
        xr_datasets, aggregation_dict, aggregation_function_dict
    )

    # STEP 6. Save shapefiles and aggregated xarray dataset if user chooses
    if aggregatedResultsPath is not None:
        # get file names
        shp_name = kwargs.get("shp_name", "aggregated_regions")
        aggregated_xr_filename = kwargs.get(
            "aggregated_xr_filename", "aggregated_xr_dataset.nc"
        )

        crs = kwargs.get("crs", 3035)

        # save shapefiles
        spu.save_shapefile_from_xarray(
            aggregated_xr_dataset['Geometry'], aggregatedResultsPath, shp_name, crs=crs
        )

        # remove geometry related data vars from aggregated xarray dataset as these cannot be saved
        aggregated_xr_dataset.pop('Geometry')

        # save aggregated xarray dataset
        file_name_with_path = os.path.join(
            aggregatedResultsPath, aggregated_xr_filename
        )
        xrIO.writeDatasetsToNetCDF(aggregated_xr_dataset, file_name_with_path)

    return aggregated_xr_dataset
