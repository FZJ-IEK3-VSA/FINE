"""Manager function that calls spatial grouping and aggregation algorithm. 
"""
import os
import logging
import warnings
import difflib
from FINE.aggregations.spatialAggregation import grouping
from FINE.aggregations.spatialAggregation import aggregation
from FINE.aggregations.spatialAggregation import managerUtils as manUtils
from FINE.IOManagement.standardIO import timer
from FINE.IOManagement import xarrayIO as xrIO, utilsIO

try:
    import geopandas as gpd
except ImportError:
    warnings.warn(
        "The package geopandas is not installed. Spatial aggregation cannot be used without it."
    )

logger_spagat = logging.getLogger("spatial_aggregation")


@timer
def perform_spatial_aggregation(
    xr_datasets,
    shapefile,
    grouping_mode="parameter_based",
    n_groups=3,
    distance_threshold=None,
    aggregatedResultsPath=None,
    **kwargs,
):
    """
    Performs spatial grouping of regions (by calling the functions in grouping.py)
    and then representation of the data within each region group (by calling functions
    in representation.py).

    :param xr_datasets: Either the path to .netCDF file or the read-in xarray datasets\n
        * Dimensions in the datasets: 'time', 'space', 'space_2'
    :type xr_datasets: str/Dict[str, xr.Dataset]

    :param shapefile: Either the path to the shapefile or the read-in shapefile
    :type shapefile: str/GeoDataFrame

    **Default arguments:**

    :param grouping_mode: Defines how to spatially group the regions. Refer to grouping.py for more
        information.
        |br| * the default value is 'parameter_based'
    :type grouping_mode: str, one of {'parameter_based', 'string_based', 'distance_based'}

    :param n_groups: The number of region groups to be formed from the original region set.
        This parameter is irrelevant if `grouping_mode` is 'string_based'.
        |br| * the default value is 3
    :type n_groups: strictly positive int

    :param distance_threshold: The distance threshold at or above which regions will not be aggregated into one.
        |br| * the default value is None. If not None, n_groups must be None
    :type distance_threshold: float

    :param aggregatedResultsPath: Indicates path to which the aggregated results should be saved.
        If None, results are not saved.
        |br| * the default value is None
    :type aggregatedResultsPath: str

    **Additional keyword arguments that can be passed via kwargs:**

    :param geom_col_name: The geomtry column name in `shapefile`
        |br| * the default value is 'geometry'
    :type geom_col_name: str

    :param geom_id_col_name: The colum in `shapefile` consisting geom IDs
        |br| * the default value is 'index'
    :type geom_id_col_name: str

    :param geom_id_col_name: The colum in `shapefile` consisting geom IDs
        |br| * the default value is 'index'
    :type geom_id_col_name: str

    :param separator: Relevant only if `grouping_mode` is 'string_based'.
        The character or string in the region IDs that defines where the ID should be split.\n
        E.g.: region IDs -> ['01_es', '02_es'] and separator='_', then IDs are split at _ and the
        last part ('es') is taken as the group ID

        |br| * the default value is None
    :type separator: str

    :param position: Relevant only if `grouping_mode` is 'string_based'.
        Used to define the position(s) of the region IDs where the split should happen.
        An int i would mean the part from 0 to i is taken as the group ID. A tuple (i,j) would mean
        the part i to j is taken at the group ID.

        .. note:: either `separator` or `position` must be passed in order to perform string_based_grouping

        |br| * the default value is None
    :type position: int/tuple

    :param weights: Relevant only if `grouping_mode` is 'parameter_based'.
        Through the `weights` dictionary, one can assign weights to variable-component pairs. When calculating
        distance corresonding to each variable-component pair, these specified weights are
        considered, otherwise taken as 1.

        It must be in one of the formats:

        * If you want to specify weights for particular variables and particular corresponding components:\n
            { 'components' : Dict[<component_name>, <weight>}], 'variables' : List[<variable_name>] }

        * If you want to specify weights for particular variables, but all corresponding components:\n
            { 'components' : {'all' : <weight>}, 'variables' : List[<variable_name>] }

        * If you want to specify weights for all variables, but particular corresponding components:\n
            { 'components' : Dict[<component_name>, <weight>}], 'variables' : 'all' }

        <weight> can be of type int/float
        |br| * the default value is None
    :type weights: Dict

    :param aggregation_method: Relevant only if `grouping_mode` is 'parameter_based'.
        The clustering method that should be used to group the regions. Options:

            * 'kmedoids_contiguity':
                kmedoids clustering with added contiguity constraint.
                Refer to TSAM docs for more info: https://github.com/FZJ-IEK3-VSA/tsam/blob/master/tsam/utils/k_medoids_contiguity.py
            * 'hierarchical':
                sklearn's agglomerative clustering with complete linkage, with a connetivity matrix to ensure contiguity.
                Refer to Sklearn docs for more info: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

        |br| * the default value is 'kmedoids_contiguity'
    :type aggregation_method: str, one of {'kmedoids_contiguity', 'hierarchical'}

    :param skip_regions: The region IDs to be skipped while aggregating regions

        .. note:: currently only implemented for `grouping_mode` 'distance_based'

        |br| * the default value is None
    :type skip_regions: List[str]

    :param enforced_groups: The groups that should be enforced when aggregating regions.

        .. note:: currently only implemented for `grouping_mode` 'distance_based'

        |br| * the default value is None
    :type enforced_groups: Dict[str, List[str]]

    :param solver: Relevant only if `grouping_mode` is 'parameter_based' and `aggregation_method` is 'kmedoids_contiguity'.
        The optimization solver to be chosen.
        |br| * the default value is 'gurobi'
    :type solver: str

    :param solver: Relevant only if `grouping_mode` is 'parameter_based' and `aggregation_method` is 'kmedoids_contiguity'.
        The optimization solver to be chosen.
        |br| * the default value is 'gurobi'
    :type solver: str

    :param aggregation_function_dict: - Contains information regarding the mode of aggregation for each individual variable.

        * Possibilities: mean, weighted mean, sum, bool (boolean OR).
        * Format of the dictionary:

            {<variable_name>: (<mode_of_aggregation>, <weights>),
            <variable_name>: (<mode_of_aggregation>, None)}

          <weights> is required only if <mode_of_aggregation> is
          'weighted mean'. The name of the variable that should act as weights should be provided. Can be None otherwise.

        .. note::
            A default dictionary is considered with the following corresponding modes. If `aggregation_function_dict` is
            passed, this default dictionary is updated. The default dicitionary:

            {\n
            "operationRateMax": ("weighted mean", "capacityMax"),\n
            "operationRateFix": ("sum", None),\n
            "locationalEligibility": ("bool", None),\n
            "capacityMax": ("sum", None),\n
            "investPerCapacity": ("mean", None),\n
            "investIfBuilt": ("bool", None),\n
            "opexPerOperation": ("mean", None),\n
            "opexPerCapacity": ("mean", None),\n
            "opexIfBuilt": ("bool", None),\n
            "interestRate": ("mean", None),\n
            "economicLifetime": ("mean", None),\n
            "capacityFix": ("sum", None),\n
            "losses": ("mean", None),\n
            "distances": ("mean", None),\n
            "commodityCost": ("mean", None),\n
            "commodityRevenue": ("mean", None),\n
            "opexPerChargeOperation": ("mean", None),\n
            "opexPerDischargeOperation": ("mean", None),\n
            "QPcostScale": ("sum", None),\n
            "technicalLifetime": ("mean", None),\n
            "balanceLimit": ("sum", None)\n
            "pathwayBalanceLimit": ("sum", None)\n
            }

        |br| * the default value is None
    :type aggregation_function_dict: Dict[str, Tuple(str, None/str)]

    :param aggregated_shp_name: Name to be given to the saved shapefiles after aggregation
        |br| * the default value is 'aggregated_regions'
    :type aggregated_shp_name: str

    :param crs: Coordinate reference system (crs) in which to save the shapefiles
        |br| * the default value is 3035
    :type crs: int

    :param aggregated_xr_filename: Name to be given to the saved netCDF file containing aggregated esM data
        |br| * the default value is 'aggregated_xr_dataset.nc'
    :type aggregated_xr_filename: str

    :returns: aggregated_xr_dataset - The xarray datasets holding aggregated data
    :rtype: Dict[str, xr.Dataset]
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
            "At least two regions must be present in shapefile and data \
            in order to perform spatial aggregation"
        )

    if n_groups is not None:
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
    geom_col_name = kwargs.get("geom_col_name", "geometry")
    geom_id_col_name = kwargs.get("geom_id_col_name", "index")

    if grouping_mode == "string_based":
        add_centroids = False
    else:
        add_centroids = True

    geom_xr = manUtils.create_geom_xarray(
        shapefile, geom_col_name, geom_id_col_name, add_centroids
    )

    xr_datasets["Geometry"] = geom_xr

    # STEP 4. Spatial grouping
    if grouping_mode == "string_based":
        separator = kwargs.get("separator", None)
        position = kwargs.get("position", None)

        locations = geom_xr.space.values

        logger_spagat.info("Performing string-based grouping on the regions")

        aggregation_dict = grouping.perform_string_based_grouping(
            locations, separator, position
        )

    elif grouping_mode == "distance_based":
        skip_regions = kwargs.get("skip_regions", None)
        enforced_groups = kwargs.get("enforced_groups", None)

        logger_spagat.info(f"Performing distance-based grouping on the regions")

        aggregation_dict = grouping.perform_distance_based_grouping(
            geom_xr, n_groups, skip_regions, enforced_groups, distance_threshold
        )

    elif grouping_mode == "parameter_based":
        weights = kwargs.get("weights", None)
        aggregation_method = kwargs.get("aggregation_method", "kmedoids_contiguity")
        solver = kwargs.get("solver", "gurobi")

        logger_spagat.info(f"Performing parameter-based grouping on the regions.")

        aggregation_dict = grouping.perform_parameter_based_grouping(
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
    ## prepare aggregation_funtion_dict
    aggregation_function_dict_default = {
        "operationRateMax": ("weighted mean", "capacityMax"),
        "operationRateFix": ("sum", None),
        "locationalEligibility": ("bool", None),
        "capacityMax": ("sum", None),
        "investPerCapacity": ("mean", None),
        "investIfBuilt": ("bool", None),
        "opexPerOperation": ("mean", None),
        "opexPerCapacity": ("mean", None),
        "opexIfBuilt": ("bool", None),
        "interestRate": ("mean", None),
        "economicLifetime": ("mean", None),
        "capacityFix": ("sum", None),
        "losses": ("mean", None),
        "distances": ("mean", None),
        "commodityCost": ("mean", None),
        "commodityRevenue": ("mean", None),
        "opexPerChargeOperation": ("mean", None),
        "opexPerDischargeOperation": ("mean", None),
        "QPcostScale": ("sum", None),
        "technicalLifetime": ("mean", None),
        "balanceLimit": ("sum", None),
        "pathwayBalanceLimit": ("sum", None),
    }

    ### if the user has passed some values, update the dict
    aggregation_function_dict = kwargs.get("aggregation_function_dict", None)
    if aggregation_function_dict != None:
        aggregation_function_dict_default.update(aggregation_function_dict)

    aggregated_xr_dataset = aggregation.aggregate_based_on_sub_to_sup_region_id_dict(
        xr_datasets, aggregation_dict, aggregation_function_dict_default
    )

    # STEP 6. Save shapefiles and aggregated xarray dataset if user chooses
    if aggregatedResultsPath is not None:
        # get file names
        aggregated_shp_name = kwargs.get("aggregated_shp_name", "aggregated_regions")
        aggregated_xr_filename = kwargs.get(
            "aggregated_xr_filename", "aggregated_xr_dataset.nc"
        )

        crs = kwargs.get("crs", 3035)

        # save shapefiles
        manUtils.save_shapefile_from_xarray(
            aggregated_xr_dataset["Geometry"],
            aggregatedResultsPath,
            aggregated_shp_name,
            crs=crs,
        )

        # remove geometry related data vars from aggregated xarray dataset as these cannot be saved
        aggregated_xr_dataset.pop("Geometry")

        # save aggregated xarray dataset
        file_name_with_path = os.path.join(
            aggregatedResultsPath, aggregated_xr_filename
        )
        xrIO.writeDatasetsToNetCDF(
            aggregated_xr_dataset, file_name_with_path, removeExisting=True
        )

    logger_spagat.info(
        f"Spatial aggregation completed, resulting in {n_groups} regions"
    )

    return aggregated_xr_dataset
