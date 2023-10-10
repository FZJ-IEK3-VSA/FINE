"""Functions to assist spatial grouping algorithms. 
"""
import warnings
import numpy as np
from scipy.cluster import hierarchy
from FINE.IOManagement.utilsIO import PowerDict

try:
    import geopandas as gpd
except ImportError:
    warnings.warn(
        "The package geopandas is not installed. Spatial aggregation cannot be used without it."
    )


def get_normalized_array(array):
    """
    Normalizes the given matrix to [0,1].

    :param matrix: Matrix to be normalized
    :type matrix: np.ndarray

    :returns: Normalized matrix
    :rtype: np.ndarray
    """

    norm_min, norm_max = 0, 1

    if np.max(array) == np.min(array):
        return np.ones(array.shape)

    return ((array - np.min(array)) / (np.max(array) - np.min(array))) * (
        norm_max - norm_min
    ) + norm_min


def preprocess_time_series(vars_dict):
    """
    Preprocesses time series variables.

    :param vars_dict: For each key (variable name), the corresponding value is a dictionary. This dictionary
                    consists of each component name and the corresponding xr.DataArray.
                    - Dimensions of xr.DataArray - 'time', 'space'
    :type vars_dict: Dict[str, Dict[str, xr.DataArray]]

    :returns: processed_ts_dict - For each key (variable name), the corresponding value is a dictionary. This dictionary
            consists of each component name and the corresponding nomalized data matrix
            - Size of each matrix: n_timesteps * n_regions
    :rtype: Dict[str, Dict[str, np.ndarray]]
    """

    processed_ts_dict = {}

    for var_name, var_dict in vars_dict.items():
        processed_ts_dict.update({var_name: {}})

        # For each component, Normalize the corresponding matrix. Add to resulting dict
        for comp_name, da in var_dict.items():
            norm_comp_matrix = get_normalized_array(da.values)

            processed_ts_dict.get(var_name).update({comp_name: norm_comp_matrix})

    return processed_ts_dict


def preprocess_1d_variables(vars_dict):
    """
    Preprocesses 1-dimensional variables.

    :param vars_dict: For each key (variable name), the corresponding value is a dictionary. This dictionary
        consists of each component name and the corresponding xr.DataArray.
        - Dimensions of xr.DataArray - 'space'
    :type vars_dict: Dict[str, Dict[str, xr.DataArray]]

    :returns: processed_1d_dict - For each key (variable name), the corresponding value is a dictionary. This dictionary
        consists of each component name and the corresponding normalized data array
        - Size of each array: n_regions
    :rtype: Dict[str, Dict[str, np.ndarray]]
    """

    processed_1d_dict = {}

    for var_name, var_dict in vars_dict.items():
        processed_1d_dict.update({var_name: {}})

        # For each component, normalize the corresponding matrix. Add to resulting dict
        for comp_name, da in var_dict.items():
            norm_comp_array = get_normalized_array(da.values)

            processed_1d_dict.get(var_name).update({comp_name: norm_comp_array})

    return processed_1d_dict


def preprocess_2d_variables(vars_dict):
    """
    Preprocesses 2-dimensional variables.

    :param vars_dict: For each key (variable name), the corresponding value is a dictionary. This dictionary consists of
        each component name and the corresponding xr.DataArray.
        - Dimensions of xr.DataArray - 'space','space_2'
    :type vars_dict: Dict[str, Dict[str, np.ndarray]]

    :returns: processed_2d_dict - For each key (variable name), the corresponding value is a dictionary. This dictionary consists of
        each component name and the corresponding data normalized (between [0, 1]),
        converted to vector form, and translated to distance meaning.
        - Size of each data array: n_regions
    :rtype: Dict[str, Dict[str, np.ndarray]]

    .. note::
        For each variable-component pair:
        - a normalised matrix of n_regions * n_regions is obtained
        - The matrix is flattened to obtain it's vector form:
                        [[0.  0.1 0.2]
                        [0.1 0.  1. ]       -->  [0.1 0.2 1. ]   (only the elements from upper or lower triangle
                        [0.2 1.  0. ]]                            as the other is always redundant in a dist matrix )
        - Translate the matrix from connectivity (similarity) to distance (dissimilarity) : (1- connectivity vector)
    """

    processed_2d_dict = {}

    for var_name, var_dict in vars_dict.items():
        processed_2d_dict.update({var_name: {}})

        # For each component...
        for comp_name, da in var_dict.items():
            ## Normalize the data
            norm_comp_matrix = get_normalized_array(da.values)

            ## Obtain the vector form of this symmetric connectivity matrix
            norm_comp_vector = hierarchy.distance.squareform(
                norm_comp_matrix, checks=False
            )

            ## Convert the value of connectivity (similarity) to distance (dissimilarity)
            norm_comp_vector = 1 - norm_comp_vector

            ## Add to resulting dict
            processed_2d_dict.get(var_name).update({comp_name: norm_comp_vector})

    return processed_2d_dict


def preprocess_dataset(xarray_dataset):
    """
    Preprocesses xarray dataset.

    :param xarray_dataset: the xarray dataset that needs to be preprocessed
    :type xarray_dataset: xr.Dataset

    :returns: dict_ts, dict_1d, dict_2d - Dictionaries obtained from
                preprocess_time_series(),
                preprocess_1d_variables(),
            and preprocess_2d_variables(), respectively
    :rtype: Dict
    """

    # STEP 0. Traverse all variables in the dataset, and put them in separate categories
    # NOTE: vars_ts, vars_1d, vars_2d -> dicts of variables and their corresponding dataArrays
    vars_ts = PowerDict()
    vars_1d = PowerDict()
    vars_2d = PowerDict()

    for comp_class, comp_dict in xarray_dataset.items():
        for comp, comp_ds in comp_dict.items():
            for varname, da in comp_ds.data_vars.items():
                ## Time series
                if varname[:3] == "ts_":
                    vars_ts[varname][comp] = da

                ## 1d variables
                elif varname[:3] == "1d_":
                    vars_1d[varname][comp] = da

                ## 2d variables
                elif varname[:3] == "2d_":
                    vars_2d[varname][comp] = da

    # STEP 1. Preprocess Time Series
    processed_ts_dict = preprocess_time_series(vars_ts)

    # STEP 2. Preprocess 1d Variables
    processed_1d_dict = preprocess_1d_variables(vars_1d)

    # STEP 3. Preprocess 2d Variables
    processed_2d_dict = preprocess_2d_variables(vars_2d)

    return processed_ts_dict, processed_1d_dict, processed_2d_dict


def get_custom_distance(
    processed_ts_dict,
    processed_1d_dict,
    processed_2d_dict,
    n_regions,
    region_index_x,
    region_index_y,
    weights=None,
):
    """
    Calculates and returns a customized distance between two regions.
    This distance is based on residual sum of squares, and is defined for
    two regions 'm' and 'n' as:
        D(m, n) = D_ts(m, n) + D_1d(m, n) + D_2d(m, n)

        where,
            D_ts(m, n) is cumulative distance of all time series variables:
                Sum of square of the difference between the values
                    - summed over all time stpes
                    - summed over all time series variables

            D_1d(m, n) is cumulative distance of all 1d variables:
                Sum of square of the difference between the values
                    - summed over all 1d variables

            D_2d(m, n) is cumulative distance of all 2d variables:
                Sum of square of (1 - value)
                    - summed over all 2d variables

                (2d values define how strong the connection is between
                two regions. They are converted to distance meaning by
                subtracting in from 1).

    :param processed_ts_dict, processed_1d_dict, processed_2d_dict: Dictionaries obtained as a result of preprocess_dataset()
    :type processed_ts_dict, processed_1d_dict, processed_2d_dict: Dict

    :param n_regions: Total number of regions in the given data
    :type n_regions: int

    :param region_index_x, region_index_y: Indicate the two regions between which the custom distance is to be calculated
        range of these indices - [0, n_regions)
    :type region_index_x, region_index_y: int

    **Default arguments:**

    :param weights: weights for each variable-component pair
        |br| * the default value is None.
    :type weights: Dict

    :returns: Custom distance value
    :rtype: float
    """

    # STEP 1. Check if weights are specified correctly
    if weights != None:
        if "components" not in weights.keys():
            raise ValueError(
                "weights dictionary must contain a 'components' dictionary within it"
            )

        if not set(weights.keys()).issubset({"components", "variables"}):
            raise ValueError(
                "Something is wrong with weights dictionary. Please refer to the its template in the doc string"
            )

        if "variables" in weights.keys():
            var_weights = weights.get("variables")
            if isinstance(var_weights, str):
                if var_weights != "all":
                    warnings.warn(
                        "Unrecognised string for variable weights. All variables will be weighted"
                    )
                    weights["variables"] = "all"

        else:
            warnings.warn(
                "variable list not found in weights dictionary. All variables will be weighted"
            )
            weights.update({"variables": "all"})

    def _get_var_comp_weight(var_name, comp_name):
        """Private function to get weight corresponding to a variable-component pair"""

        wgt = 1

        if weights != None:
            [var_category, var] = var_name.split(
                "_"
            )  # strip the category and take only var

            var_weights = weights.get("variables")
            comp_weights = weights.get("components")

            if (var_weights == "all") or (var in var_weights):
                if comp_weights.get("all") != None:
                    wgt = comp_weights.get("all")
                elif comp_weights.get(comp_name) != None:
                    wgt = comp_weights.get(comp_name)

        return wgt

    # STEP 2. Find distance for each variable category separately

    # STEP 3a. Distance of Time Series category
    distance_ts = 0
    for var_name, var_dict in processed_ts_dict.items():
        for comp_name, data_matrix in var_dict.items():
            # (i) Get weight
            var_comp_weight = _get_var_comp_weight(var_name, comp_name)

            # (ii) Extract data corresponding to the variable-component pair in both regions
            region_x_data = data_matrix[:, region_index_x]
            region_y_data = data_matrix[:, region_index_y]

            # (ii) Calculate distance
            # INFO: ts_region_x and ts_region_y are vectors,
            # subtract the vectors, square each element and add all elements. And multiply with its weight
            distance_ts += (
                sum(np.power((region_x_data - region_y_data), 2)) * var_comp_weight
            )

    # STEP 3b. Distance of 1d Variables category
    distance_1d = 0
    for var_name, var_dict in processed_1d_dict.items():
        for comp_name, data_array in var_dict.items():
            # (i) Get weight
            var_comp_weight = _get_var_comp_weight(var_name, comp_name)

            # (ii) Extract data corresponding to the variable in both regions
            region_x_data = data_array[region_index_x]
            region_y_data = data_array[region_index_y]

            # (iii) Calculate distance
            distance_1d += pow(region_x_data - region_y_data, 2) * var_comp_weight

    # STEP 3c. Distance of 2d Variables category
    distance_2d = 0

    # STEP 3c (i). Since processed_2d_dict is a condensed matrix, we have to get dist. corresponding to the two given regions
    region_index_x_y = (
        region_index_x * (n_regions - region_index_x)
        + (region_index_y - region_index_x)
        - 1
    )

    for var_name, var_dict in processed_2d_dict.items():
        for comp_name, data_array in var_dict.items():
            # (i) Get weight
            var_comp_weight = _get_var_comp_weight(var_name, comp_name)

            # (ii) Extract data corresponding to the variable in both regions
            dist = data_array[region_index_x_y]

            if not np.isnan(
                dist
            ):  # INFO: if the regions are not connected the value will be na
                # Calculate the distance
                distance_2d += pow(dist, 2) * var_comp_weight

    # STEP 4. Add all three distances
    return distance_ts + distance_1d + distance_2d


def get_custom_distance_matrix(
    processed_ts_dict, processed_1d_dict, processed_2d_dict, n_regions, weights=None
):
    """
    For every region combination, calculates the custom distance by calling get_custom_distance().

    :param processed_ts_dict, processed_1d_dict, processed_2d_dict: Dictionaries obtained as a result of preprocess_dataset()
    :type processed_ts_dict, processed_1d_dict, processed_2d_dict: Dict

    :param n_regions: Total number of regions in the given data
    :type n_regions: int

    **Default arguments:**

    :param weights: weights for each variable-component pair
        |br| * the default value is None.
    :type weights: Dict

    :returns: distMatrix - A n_regions by n_regions hollow, symmetric distance matrix
    :rtype: np.ndarray
    """

    distMatrix = np.zeros((n_regions, n_regions))

    # STEP 1. For every region pair, calculate the distance
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            distMatrix[i, j] = get_custom_distance(
                processed_ts_dict,
                processed_1d_dict,
                processed_2d_dict,
                n_regions,
                i,
                j,
                weights,
            )

    # STEP 2. Only upper triangle has values, reflect these values in lower triangle to make it a hollow, symmetric matrix
    distMatrix += distMatrix.T - np.diag(distMatrix.diagonal())

    return distMatrix


def get_connectivity_matrix(xarray_datasets):
    """
    Generates connectiviy matrix for the given `xarray_datasets`.

    :param xarray_datasets: The dictionary of xarray datasets for which connectiviy matrix needs
        to be generated
    :type xarray_datasets: Dict[str, xr.Dataset]

    :returns: connectivity_matrix - A n_regions by n_regions symmetric matrix
    :rtype: np.ndarray

    .. note::
        The `connectivity_matrix` indicates if two regions are connected or not.
        - In this matrix, if two regions are connected, it is indicated as 1 and 0 otherwise.
        - A given region pair if connected if:
            - Their borders touch at least at one point
            - In case of islands, its nearest mainland region, or
            - If the regions are connected via a transmission line or pipeline
    """

    geom_xr = xarray_datasets.get("Geometry")
    input_xr = xarray_datasets.get("Input")

    n_regions = len(geom_xr["space"].values)

    connectivity_matrix = np.zeros((n_regions, n_regions))

    # STEP 1: Check for contiguous neighbors
    geometries = gpd.GeoSeries(
        geom_xr["geometries"].values
    )  # NOTE: disjoint seems to work only on geopandas or geoseries object
    for ix, geom in enumerate(geometries):
        neighbors = geometries[~geometries.disjoint(geom)].index.tolist()
        connectivity_matrix[ix, neighbors] = 1

    # STEP 2: Find nearest neighbor for island regions
    for row in range(len(connectivity_matrix)):
        if (
            np.count_nonzero(connectivity_matrix[row, :] == 1) == 1
        ):  # if a region is connected only to itself
            # get the nearest neighbor based on regions centroids
            centroid_distances = geom_xr["centroid_distances"].values[row, :]
            nearest_neighbor_idx = np.argmin(
                centroid_distances[np.nonzero(centroid_distances)]
            )

            # make the connection between the regions (both ways to keep it symmetric)
            (
                connectivity_matrix[row, nearest_neighbor_idx],
                connectivity_matrix[nearest_neighbor_idx, row],
            ) = (1, 1)

    # STEP 3: Additionally, check if there are transmission between regions that are not yet connected in the
    # connectivity matrix
    for comp_class, comp_dict in input_xr.items():
        for comp, comp_ds in comp_dict.items():
            for varname, da in comp_ds.data_vars.items():
                if varname[:3] == "2d_":
                    connectivity_matrix[
                        da.values > 0
                    ] = 1  # if a pos, non-zero value exits, make a connection!

    return connectivity_matrix


def get_region_list(geom_xr, skip_regions, enforced_group):
    """
    Generates a modified region list that is to be used during region grouping.

    :param geom_xr: The xarray dataset holding the geom info
    :type geom_xr: xr.Dataset

    param skip_regions: The region IDs to be skipped while aggregating regions
        |br| * the default value is None
    :type skip_regions: List

        * Ex.: ['02_reg']
               ['02_reg', '03_reg]

    :param enforced_group: A region group
    |br| * the default value is None
    :type enforced_group: List

        * Ex.: ['01_es', '02_es', '03_es']

    :returns: connectivity_matrix - A n_regions by n_regions symmetric matrix
    :rtype: np.ndarray
    """

    if (skip_regions is not None) & (enforced_group is None):
        assert isinstance(
            skip_regions, list
        ), "A list containing the region ID's to be skipped should be provided."

        # get all regions
        regions_list = geom_xr["space"].values

        # remove regions that should be skipped
        regions_list = np.array(list(set(regions_list) - set(skip_regions)))

        # create skipped regions dict
        skipped_dict = {reg: [reg] for reg in skip_regions}

    elif (skip_regions is None) & (enforced_group is not None):
        assert isinstance(
            enforced_group, list
        ), "A dictionary containing the super-regions as keys and sub-regions values should be provided."

        # get subset of regions
        regions_list = np.array(list(enforced_group))

        # create an empty skipped regions dict
        skipped_dict = {}

    elif (skip_regions is not None) & (enforced_group is not None):
        assert isinstance(
            skip_regions, list
        ), "A list containing the region ID's to be skipped should be provided."
        assert isinstance(
            enforced_group, list
        ), "A dictionary containing the super-regions as keys and sub-regions values should be provided."

        # get region subset based on enfored_group
        skip_regions, enforced_group = list(map(set, [skip_regions, enforced_group]))
        regions_list = enforced_group - skip_regions
        regions_list = np.array(list(regions_list))

        # create skipped regions dict
        skipped_dict = {reg: [reg] for reg in skip_regions}

    else:
        # get all regions
        regions_list = geom_xr["space"].values
        skipped_dict = {}

    return regions_list, skipped_dict
