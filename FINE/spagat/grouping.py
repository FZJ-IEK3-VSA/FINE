"""Grouping algorithms determine how to reduce the number of input regions to 
fewer regions while minimizing information loss.
"""
import logging

import math 
import numpy as np

import sklearn.cluster as skc
import libpysal.weights as wgt
from spopt.region.skater import SpanningForest

import FINE.spagat.utils as spu
import FINE.spagat.grouping_utils as gu

logger_grouping = logging.getLogger("spatial_grouping")


def perform_string_based_grouping(regions):
    """Groups regions based on their names/ids. Looks for a match in ids after 
    a '_'. For example: '01_es', '02_es' both have 'es' after the '_'. 
    Therefore, the regions appear in the same group.

    Parameters
    ----------
    regions : List[str] or np.array(str)
        List or array of region names 
        Ex.: ['01_es', '02_es', '01_de', '02_de', '03_de']

    Returns
    -------
    sub_to_sup_region_id_dict : Dict[str, List[str]]
        Dictionary new regions' ids and their corresponding group of regions 
        Ex. {'es' : ['01_es', '02_es'] , 
             'de' : ['01_de', '02_de', '03_de']}
    """
    #FEATURE REQUEST: this is implemented spefically for the e-id: '01_es' -> generalize this!
    nation_set = set([region_id.split("_")[1] for region_id in regions])

    sub_to_sup_region_id_dict = {}

    for nation in nation_set:
        sub_to_sup_region_id_dict[nation] = [
            region_id for region_id in regions if region_id.split("_")[1] == nation
        ]

    return sub_to_sup_region_id_dict


@spu.timer
def perform_distance_based_grouping(xarray_dataset, 
                                    n_groups = 3):
    """Groups regions based on the regions' centroid distances,  #TODO: update docstirng 
    using sklearn's hierarchical clustering.

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        The xarray dataset holding the esM's info 
    save_path :  str, optional (default=None)
        The path to save the figures. 
        If default None, figure is not saved
    fig_name : str, optional (default=None)
        Name of the saved figure. 
        Valid only if `save_path` is not None. 
        If default None, figure saves under the name 
        'sklearn_hierarchical_dendrogram' 
    verbose : bool, optional (default=False)
        If True, the grouping results are printed. Supressed if False 

    Returns
    -------
    aggregation_dict : Dict[int, Dict[str, List[str]]] 
        A nested dictionary containing results of spatial grouping at various levels/number of groups
        - Ex. {3: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']},
               2: {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg': ['03_reg']},
               1: {'01_reg_02_reg_03_reg': ['01_reg','02_reg','03_reg']}}
    """
    centroids = np.asarray([[point.item().x, point.item().y] for point in xarray_dataset.gpd_centroids])/1000  # km
    regions_list = xarray_dataset['space'].values

    #STEP 1. Compute hierarchical clustering
    model = skc.AgglomerativeClustering(n_clusters=n_groups).fit(centroids)

    #STEP 2. Create a regions dictionary for the aggregated regions
    aggregation_dict = {}
    for label in range(n_groups):
        # Group the regions of this regions label
        sub_regions_list = list(regions_list[model.labels_== label])
        sup_region_id = '_'.join(sub_regions_list)
        aggregation_dict[sup_region_id] = sub_regions_list.copy()
    

    return aggregation_dict



@spu.timer
def perform_parameter_based_grouping(xarray_dataset,
                                    n_groups = 3, 
                                    aggregation_method = 'skater', 
                                    weights=None):
    """Groups regions based on the Energy System Model instance's data. #TODO: update the doc string 
    This data may consist of -
        a. regional time series variables such as operationRateMax of PVs
        b. regional values such as capacityMax of PVs
        c. connection values such as distances of DCCables 
        d. values constant across all regions such as CommodityConversionFactors 

    All variables that vary across regions (a,b, and c) belonging to different 
    ESM components are considered while determining similarity between regions. 
    Sklearn's agglomerative hierarchical clustering is used to cluster the 
    regions.

    Parameters
    ----------
    xarray_dataset : xr.Dataset
        The xarray dataset holding the esM's info 
    linkage : str, optional (default='complete')
        The linkage criterion to be used with agglomerative hierarchical clustering. 
        Can be 'complete', 'single', etc. Refer to Sklearn's documentation for more info.
    weights : Dict 
        Through the `weights` dictionary, one can assign weights to variable-component pairs. 
        It must be in one of the formats:
        - If you want to specify weights for particular variables and particular corresponding components:
            { 'components' : Dict[<component_name>, <weight>}], 'variables' : List[<variable_name>] }
        - If you want to specify weights for particular variables, but all corresponding components:
            { 'components' : {'all' : <weight>}, 'variables' : List[<variable_name>] }
        - If you want to specify weights for all variables, but particular corresponding components:
            { 'components' : Dict[<component_name>, <weight>}], 'variables' : 'all' }
        
        <weight> can be of type int/float 

        When calculating distance corresonding to each variable-component pair, these specified weights are 
        considered, otherwise taken as 1.  

    Returns
    -------
    aggregation_dict : Dict[int, Dict[str, List[str]]]
        A nested dictionary containing results of spatial grouping at various levels/number of groups
        - Ex. {3: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']},
               2: {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg': ['03_reg']},
               1: {'01_reg_02_reg_03_reg': ['01_reg','02_reg','03_reg']}}
    
    Notes
    -----
    * While clustering/grouping regions, it is important to make sure that the regions 
      are spatially contiguous. Sklearn's agglomerative hierarchical clustering method is 
      capable of taking care of this if additional connectivity matrix is input. 
      This matrix should indicate which region pairs are connected (or contiguous).

    Overall steps involved:
        - Preprocessing data -> preprocessDataset()
        - Custom distance -> selfDistanceMatrix()
        - Clustering method -> sklearn's agglomerative hierarchical clustering with specified 
                                `linkage`
        - Spatial contiguity -> Connectivity matrix is passed to the clustering method. 
                                generateConnectivityMatrix() to obtain Connectivity matrix.
        - Accuracy indicators -> (a) Cophenetic correlation coefficients are printed
                                 (b) Inconsistencies are printed. 
                                 (c) Silhouette scores are printed. 
    """

    # Original region list
    regions_list = xarray_dataset['space'].values
    n_regions = len(regions_list)

    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}  
    
    #STEP 1. Preprocess the whole dataset 
    processed_ts_dict, processed_1d_dict, processed_2d_dict = gu.preprocess_dataset(xarray_dataset) 

    #STEP 2. Calculate the overall distance between each region pair (uses custom distance)
    precomputed_dist_matrix = gu.get_custom_distance_matrix(processed_ts_dict, 
                                                            processed_1d_dict, 
                                                            processed_2d_dict, 
                                                            n_regions, 
                                                            weights)
    
    #STEP 3.  Obtain and check the connectivity matrix - indicates if a region pair is contiguous or not. 
    connectivity_matrix = gu.get_connectivity_matrix(xarray_dataset)

    if aggregation_method == 'hierarchical':
               
        model = skc.AgglomerativeClustering(n_clusters=n_groups,
                                            affinity='precomputed',
                                            linkage='complete',
                                            connectivity=connectivity_matrix).fit(precomputed_dist_matrix)

        #STEP 4c. Aggregated regions dict
        aggregation_dict = {}
        for label in range(n_groups):
            # Group the regions of this regions label
            sub_regions_list = list(regions_list[model.labels_== label])
            sup_region_id = '_'.join(sub_regions_list)
            aggregation_dict[sup_region_id] = sub_regions_list.copy()


    elif aggregation_method == 'skater':

        # get the connectivity in the format required by skater 
        n_rows, n_cols = connectivity_matrix.shape
        custom_neighbors = {}
        for i in range(n_rows):
            neighbors = []
            for j in range(n_cols):
                if (i!=j and connectivity_matrix[i,j] == 1):
                    neighbors.append(j)
            custom_neighbors.update({i:neighbors})

        contiguity_weights = wgt.W(custom_neighbors)   
        
        # start skater algorithm 
        n = 0
        final_n_clusters = 0 

        print(f'Original quorum per region group: {math.floor(n_regions/n_groups)}')

        while final_n_clusters != n_groups:
            
            min_reg_per_group = math.floor(n_regions/(n_groups+n))
            
            model = SpanningForest(dissimilarity='precomputed', reduction=np.max).fit(n_groups,
                                                                                        W=contiguity_weights,
                                                                                        data=precomputed_dist_matrix,
                                                                                        quorum=min_reg_per_group,
                                                                                        trace=False,
                                                                                        islands="increase")
            final_n_clusters = len(np.unique(model.current_labels_))
            n += 1

        print(f'Final quorum per region: {min_reg_per_group}')

        #STEP 4c. Aggregated regions dict
        aggregation_dict = {}
        for label in range(n_groups):
            # Group the regions of this regions label
            sub_regions_list = list(regions_list[model.current_labels_ == label])
            sup_region_id = '_'.join(sub_regions_list)
            aggregation_dict[sup_region_id] = sub_regions_list.copy()


    else: 
        raise ValueError(f'The aggregation method {aggregation_method} is not valid. Please choose either \
        skater or hierarchical')

        
    return aggregation_dict



