"""
Grouping algorithms determine how to reduce the number of input regions to 
fewer regions while minimizing information loss.
"""

import logging
import numpy as np
import sklearn.cluster as skc
from tsam.utils.k_medoids_contiguity import k_medoids_contiguity
from FINE.aggregations.spatialAggregation import groupingUtils as gprUtils
from FINE.IOManagement.standardIO import timer
from functools import partial

logger_grouping = logging.getLogger("spatial_grouping")


def perform_string_based_grouping(regions, separator=None, position=None):
    """
    Groups regions based on their names/ids.

    :param regions: List or array of region names.\n
        Ex.: ['01_es', '02_es', '01_de', '02_de', '03_de']
    :type regions: List[str]/np.array(str)

    **Default arguments:**

    :param separator: The character or string in the region IDs that defines where the ID should be split

        * Ex.: '_' would split the above IDs at _ and take the last part ('es', 'de') as the group ID

        |br| * the default value is None
    :type separator: str

    :param position: Used to define the position(s) of the region IDs where the split should happen.
        An int i would mean the part from 0 to i is taken as the group ID. A tuple (i,j) would mean
        the part i to j is taken at the group ID.
        |br| * the default value is None
    :type position: int/tuple

    :returns: sub_to_sup_region_id_dict - Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'es' : ['01_es', '02_es'] , 'de' : ['01_de', '02_de', '03_de']}

    :rtype: Dict[str, List[str]]
    """

    sub_to_sup_region_id_dict = {}

    if isinstance(position, int):
        position = (0, position)

    if separator != None and position == None:
        for region in regions:
            sup_region = region.split(separator)[1]

            if sup_region not in sub_to_sup_region_id_dict.keys():
                sub_to_sup_region_id_dict[sup_region] = [region]
            else:
                sub_to_sup_region_id_dict[sup_region].append(region)

    elif separator == None and position != None:
        for region in regions:
            sup_region = region[position[0] : position[1]]

            if sup_region not in sub_to_sup_region_id_dict.keys():
                sub_to_sup_region_id_dict[sup_region] = [region]
            else:
                sub_to_sup_region_id_dict[sup_region].append(region)

    else:
        raise ValueError("Please input either separator or position")

    return sub_to_sup_region_id_dict


# %%
@timer
def perform_distance_based_grouping(
    geom_xr,
    n_groups=3,
    skip_regions=None,
    enforced_groups=None,
    distance_threshold=None,
):
    """
    Groups regions based on the regions' centroid distances, using sklearn's hierarchical clustering.

    :param geom_xr: The xarray dataset holding the geom info
    :type geom_xr: xr.Dataset

    **Default arguments:**

    :param n_groups: The number of region groups to be formed from the original region set
        |br| * the default value is 3
    :type n_groups: strictly positive int

    :param distance_threshold: The distance threshold at or above which regions will not be aggregated into one.
        |br| * the default value is None. If not None, n_groups must be None
    :type distance_threshold: float

    :param skip_regions: The region id's to be skipped while aggregating regions
        |br| * the default value is None
    :type skip_regions: List

        * Ex.: ['02_reg']
               ['02_reg', '03_reg]

    :param enforced_groups: The groups that should be enforced when aggregating regions.
        |br| * the default value is None
    :type enforced_groups: Dict

        * Ex.: {'es': ['01_es', '02_es', '03_es']
                'de': ['01_de', '02_de', '03_de', '04_de']}

    :returns: aggregation_dict - A nested dictionary containing results of spatial grouping at various levels/number of groups

        * Ex.: {3: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']},\n
            2: {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg': ['03_reg']},\n
            1: {'01_reg_02_reg_03_reg': ['01_reg','02_reg','03_reg']}}

    :rtype: Dict[int, Dict[str, List[str]]]
    """

    def _perform_distance_based_grouping(
        geom_xr,
        n_groups=3,
        skip_regions=None,
        enforced_group=None,
        distance_threshold=distance_threshold,
    ):
        regions_list, skipped_dict = gprUtils.get_region_list(
            geom_xr, skip_regions, enforced_group
        )

        # reduce ds
        geom_xr = geom_xr.sel(space=regions_list)
        centroids = geom_xr["centroids"].values

        centroids_x_y_points = (
            np.asarray([[point.x, point.y] for point in centroids]) / 1000
        )  # km

        # STEP 1. Compute hierarchical clustering
        model = skc.AgglomerativeClustering(
            n_clusters=n_groups, distance_threshold=distance_threshold
        ).fit(centroids_x_y_points)

        # STEP 2. Create a regions dictionary for the aggregated regions
        aggregation_dict = {}
        for label in np.unique(model.labels_):
            # Group the regions of this regions label
            sub_regions_list = sorted(regions_list[model.labels_ == label])
            sup_region_id = "_".join(sub_regions_list)
            aggregation_dict[sup_region_id] = sub_regions_list.copy()

        aggregation_dict.update(skipped_dict)

        return aggregation_dict

    if enforced_groups is None:
        aggregation_dict = _perform_distance_based_grouping(
            geom_xr=geom_xr,
            skip_regions=skip_regions,
            n_groups=n_groups,
            distance_threshold=distance_threshold,
        )
        return aggregation_dict

    else:
        aggregation_dict = {}
        for key, group in enforced_groups.items():
            print("Grouping: ", key)
            output = _perform_distance_based_grouping(
                geom_xr=geom_xr,
                skip_regions=skip_regions,
                enforced_group=group,
                n_groups=n_groups,
                distance_threshold=distance_threshold,
            )
            aggregation_dict.update(output)

        return aggregation_dict


# %%


# TODO: If there is a demand for it, implement skip_regions, and enforce_group
# (just like in perform_distance_based_grouping) here as well
@timer
def perform_parameter_based_grouping(
    xarray_datasets,
    n_groups=3,
    aggregation_method="kmedoids_contiguity",
    weights=None,
    solver="gurobi",
):
    """
    Groups regions based on the Energy System Model instance's data.
    This data may consist of

        a. regional time series variables such as operationRateMax of PVs
        b. regional values such as capacityMax of PVs
        c. connection values such as distances of DC Cables
        d. values constant across all regions such as CommodityConversionFactors

    All variables that vary across regions (a,b, and c) belonging to different
    ESM components are considered while determining similarity between regions.

    :param xarray_datasets: The dictionary of xarray datasets holding esM's info
    :type xarray_datasets: Dict[str, xr.Dataset]

    **Default arguments:**

    :param n_groups: The number of region groups to be formed from the original region set
        |br| * the default value is 3
    :type n_groups: strictly positive int

    :param aggregation_method: The clustering method that should be used to group the regions.
        Options:

            * 'kmedoids_contiguity':\n
                kmedoids clustering with added contiguity constraint\n
                Refer to TSAM docs for more info: https://github.com/FZJ-IEK3-VSA/tsam/blob/master/tsam/utils/k_medoids_contiguity.py
            * 'hierarchical': \n
                sklearn's agglomerative clustering with complete linkage, with a connetivity matrix to ensure contiguity\n
                Refer to Sklearn docs for more info: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

        |br| * the default value is 'kmedoids_contiguity'
    :type aggregation_method: str

    :param weights: Through the `weights` dictionary, one can assign weights to variable-component pairs. When calculating
        distance corresonding to each variable-component pair, these specified weights are
        considered, otherwise taken as 1. It must be in one of the formats:

        * If you want to specify weights for particular variables and particular corresponding components:\n
            { 'components' : Dict[<component_name>, <weight>}], 'variables' : List[<variable_name>] }\n
        * If you want to specify weights for particular variables, but all corresponding components:\n
            { 'components' : {'all' : <weight>}, 'variables' : List[<variable_name>] }\n
        * If you want to specify weights for all variables, but particular corresponding components:\n
            { 'components' : Dict[<component_name>, <weight>}], 'variables' : 'all' }

        <weight> can be of type int/float
        |br| * the default value is None
    :type weights: Dict

    :param solver: The optimization solver to be chosen.
        Relevant only if `aggregation_method` is 'kmedoids_contiguity'
        |br| * the default value is 'gurobi'
    :type solver: str

    :returns: aggregation_dict - A nested dictionary containing results of spatial grouping at various levels/number of groups

        * Ex.: {3: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']},\n
            2: {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg': ['03_reg']},\n
            1: {'01_reg_02_reg_03_reg': ['01_reg','02_reg','03_reg']}}

    :rtype: Dict[int, Dict[str, List[str]]]
    """

    # Original region list
    regions_list = xarray_datasets.get("Geometry")["space"].values
    n_regions = len(regions_list)

    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

    # STEP 1. Preprocess the whole dataset
    (
        processed_ts_dict,
        processed_1d_dict,
        processed_2d_dict,
    ) = gprUtils.preprocess_dataset(xarray_datasets.get("Input"))

    # STEP 2. Calculate the overall distance between each region pair (uses custom distance)
    precomputed_dist_matrix = gprUtils.get_custom_distance_matrix(
        processed_ts_dict, processed_1d_dict, processed_2d_dict, n_regions, weights
    )

    # STEP 3.  Obtain and check the connectivity matrix - indicates if a region pair is contiguous or not.
    connectivity_matrix = gprUtils.get_connectivity_matrix(xarray_datasets)

    # STEP 4. Cluster the regions
    if aggregation_method == "hierarchical":
        model = skc.AgglomerativeClustering(
            n_clusters=n_groups,
            affinity="precomputed",
            linkage="complete",
            connectivity=connectivity_matrix,
        ).fit(precomputed_dist_matrix)

        aggregation_dict = {}
        for label in range(n_groups):
            # Group the regions of this regions label
            sub_regions_list = list(regions_list[model.labels_ == label])
            sup_region_id = "_".join(sub_regions_list)
            aggregation_dict[sup_region_id] = sub_regions_list.copy()

    elif aggregation_method == "kmedoids_contiguity":
        r_y, r_x, r_obj = k_medoids_contiguity(
            precomputed_dist_matrix, n_groups, connectivity_matrix, solver=solver
        )
        labels_raw = r_x.argmax(axis=0)

        # Aggregated regions dict
        aggregation_dict = {}
        for label in np.unique(labels_raw):
            # Group the regions of this regions label
            sub_regions_list = list(regions_list[labels_raw == label])
            sup_region_id = "_".join(sub_regions_list)
            aggregation_dict[sup_region_id] = sub_regions_list.copy()

    else:
        raise ValueError(
            f"The aggregation method {aggregation_method} is not valid. Please choose either \
        kmedoids_contiguity or hierarchical"
        )

    return aggregation_dict
