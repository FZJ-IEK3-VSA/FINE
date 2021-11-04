import warnings

import numpy as np
import geopandas as gpd
from scipy.cluster import hierarchy

from FINE.IOManagement.utilsIO import PowerDict


def get_normalized_array(array):
    """Normalize the given matrix to [0,1].

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be normalized

    Returns
    -------
    np.ndarray
        Normalized matrix
    """

    norm_min, norm_max = 0, 1

    if np.max(array) == np.min(array):
        warnings.warn(
            "The minimum and maximum values are the same in the array to be normalized. Setting all values to 1"
        )
        return np.ones(array.shape)

    return ((array - np.min(array)) / (np.max(array) - np.min(array))) * (
        norm_max - norm_min
    ) + norm_min


def preprocess_time_series(vars_dict):
    """Preprocess time series variables.

    Parameters
    ----------
    vars_dict : Dict[str, Dict[str, xr.DataArray]] 
        For each key (variable name), the corresponding value is a dictionary. This dictionary
        consists of each component name and the corresponding xr.DataArray.
        - Dimensions of xr.DataArray - 'time', 'space'

    Returns
    -------
    processed_ts_dict : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary
        consists of each component name and the corresponding nomalized data matrix
        - Size of each matrix: n_timesteps * n_regions 
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
    """Preprocess 1-dimensional variables.

    Parameters
    ----------
    vars_dict : Dict[str, Dict[str, xr.DataArray]] 
        For each key (variable name), the corresponding value is a dictionary. This dictionary
        consists of each component name and the corresponding xr.DataArray.
        - Dimensions of xr.DataArray - 'space'

    Returns
    -------
    processed_1d_dict : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary
        consists of each component name and the corresponding normalized data array
        - Size of each array: n_regions
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
    """Preprocess 2-dimensional variables.

    Parameters
    ----------
    vars_dict : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary consists of
        each component name and the corresponding xr.DataArray.
        - Dimensions of xr.DataArray - 'space','space_2'

    Returns
    -------
    processed_2d_dict : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary consists of
        each component name and the corresponding data normalized (between [0, 1]),
        converted to vector form, and translated to distance meaning.
        - Size of each data array: n_regions

    Notes
    -----
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
    """Preprocess xarray dataset.

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        the xarray dataset that needs to be preprocessed

    Returns
        dict_ts, dict_1d, dict_2d : Dict
            Dictionaries obtained from
                preprocess_time_series(),
                preprocess_1d_variables(),
            and preprocess_2d_variables(), respectively
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
    """Calculates and returns a customized distance between two regions.
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


    Parameters
    ----------
    processed_ts_dict, processed_1d_dict, processed_2d_dict : Dict
        Dictionaries obtained as a result of preprocess_dataset()
    n_regions : int
        Total number of regions in the given data
    region_index_x, region_index_y : int
        Indicate the two regions between which the custom distance is to be calculated
        range of these indices - [0, n_regions)
    weights : Dict
        weights for each variable-component pair

    Returns
    -------
    float
        Custom distance value
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

    """For every region combination, calculates the custom distance by calling get_custom_distance().

    Parameters
    ----------
    processed_ts_dict, processed_1d_dict, processed_2d_dict : Dict
        Dictionaries obtained as a result of preprocess_dataset()
    n_regions : int
        Total number of regions in the given data
        range of these indices - [0, n_regions)
    weights : Dict
        weights for each variable-component pair

    Returns
    -------
    distMatrix : np.ndarray
        A n_regions by n_regions hollow, symmetric distance matrix
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
    """Generates connectiviy matrix for the given `xarray_datasets`.

    Parameters
    ----------
    xarray_datasets : Dict[str, xr.Dataset]
        The dictionary of xarray datasets for which connectiviy matrix needs
        to be generated

    Returns
    -------
    connectivity_matrix : np.ndarray
        A n_regions by n_regions symmetric matrix

    Notes
    -----
    The `connectivity_matrix` indicates if two regions are connected or not.
    - In this matrix, if two regions are connected, it is indicated as 1 and 0 otherwise.
    - A given region pair if connected if:
        - Their borders touch at least at one point
        - In case of islands, its nearest mainland region, or
        - If the regions are connected via a transmission line or pipeline
    """

    geom_xr = xarray_datasets.get('Geometry')
    input_xr = xarray_datasets.get('Input')

    n_regions = len(geom_xr["space"].values)

    connectivity_matrix = np.zeros((n_regions, n_regions))

    # STEP 1: Check for contiguous neighbors
    geometries = gpd.GeoSeries(geom_xr['geometries'].values)  # NOTE: disjoint seems to work only on geopandas or geoseries object
    for ix, geom in enumerate(geometries):
        neighbors = geometries[~geometries.disjoint(geom)].index.tolist()
        connectivity_matrix[ix, neighbors] = 1

    # STEP 2: Find nearest neighbor for island regions
    for row in range(len(connectivity_matrix)):
        if (
            np.count_nonzero(connectivity_matrix[row, :] == 1) == 1
        ):  # if a region is connected only to itself

            # get the nearest neighbor based on regions centroids
            centroid_distances = geom_xr['centroid_distances'].values[row, :]
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
                    connectivity_matrix[da.values > 0] = 1  # if a pos, non-zero value exits, make a connection!

    return connectivity_matrix
