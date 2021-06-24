"""Grouping algorithms determine how to reduce the number of input regions to 
fewer regions while minimizing information loss.
"""
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from sklearn import metrics
from scipy.cluster import vq
import sklearn.cluster as skc

import FINE.spagat.utils as spu
import FINE.spagat.grouping_utils as gu

logger_grouping = logging.getLogger("spagat_grouping")


def perform_string_based_grouping(regions):
    """Groups regions based on their names/ids.

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
                                save_path = None, 
                                fig_name=None, 
                                verbose=False):
    """Groups regions based on the regions' centroid distances, 
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
    n_regions = len(regions_list)
    
    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}


    #STEP 1. Clustering for each number of regions from 1 to total number present in the original resolution
    for i in range(1, n_regions):

        #STEP 1a. Compute hierarchical clustering
        model = skc.AgglomerativeClustering(n_clusters=i).fit(centroids)
        regions_label_list = model.labels_

        #STEP 1b. Create a regions dictionary for the aggregated regions
        regions_dict = {}
        for label in range(i):
            # Group the regions of this regions label
            sub_regions_list = list(regions_list[regions_label_list == label])
            sup_region_id = '_'.join(sub_regions_list)
            regions_dict[sup_region_id] = sub_regions_list.copy()
        
        if verbose:
            print(i)
            print('\t', 'lables:', regions_label_list)
            for sup_region_id, sub_regions_list in regions_dict.items():
                print('\t', sup_region_id, ': ', sub_regions_list)

        #STEP 1c. Append the dict to main dict 
        aggregation_dict[i] = regions_dict.copy()

    #STEP 2. Create linkage matrix 
    #STEP 2a. Obtain clustering tree
    clustering_tree = skc.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(centroids)

    #STEP 2b. Create the counts of samples under each node
    counts = np.zeros(clustering_tree.children_.shape[0])
    n_samples = len(clustering_tree.labels_)
    for i, merge in enumerate(clustering_tree.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    #STEP 2c. Obtain linkage matrix    
    linkage_matrix = np.column_stack([clustering_tree.children_, clustering_tree.distances_, counts]).astype(float)   
    
    #STEP 3. Create and save dendrogram (from linkage matrix) if user specifies 
    if fig_name is None: fig_name = 'sklearn_hierarchical_dendrogram'

    if save_path is not None:
        fig, ax = plt.subplots(figsize=(25, 12))

        R = hierarchy.dendrogram(linkage_matrix, 
                                orientation="top",
                                labels=xarray_dataset['space'].values, 
                                ax=ax, 
                                leaf_font_size=14
                                )

        spu.plt_savefig(save_name=fig_name, path=save_path)
    
    #STEP 3. Obtain distance matrix
    distance_matrix = hierarchy.distance.pdist(centroids)

    #STEP 4. Evaluation 
    print('Statistics on this hiearchical clustering:')

    #STEP 4a. Cophenetic correlation coefficients
    cophenetic_correlation_coefficient = hierarchy.cophenet(linkage_matrix, distance_matrix)[0]
    print('The cophenetic correlation coefficient is ', cophenetic_correlation_coefficient)

    #STEP 4b. Inconsistency coefficients (in a plot)
    fig, ax = plt.subplots(figsize=(18,7))
    inconsistency = hierarchy.inconsistent(linkage_matrix)
    ax.plot(range(1,len(linkage_matrix)+1),list(inconsistency[:,3]),'go-')
    ax.set_title('Inconsistency Coefficients: indicate where to cut the hierarchy', fontsize=14)
    ax.set_xlabel('Linkage height', fontsize=12)
    ax.set_ylabel('Inconsistencies', fontsize=12)

    plt.xticks(np.arange(1, len(linkage_matrix)+1, 1))
    plt.show(block=False)

    
    return aggregation_dict



@spu.timer
def perform_parameter_based_grouping(xarray_dataset,
                                    linkage='complete',
                                    weights=None):
    """Groups regions based on the Energy System Model instance's data. 

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
    dict_ts, dict_1d, dict_2d = gu.preprocess_dataset(xarray_dataset) 

    #STEP 2. Calculate the overall distance between each region pair (uses custom distance)
    precomputed_dist_matrix = gu.get_custom_distance_matrix(dict_ts, dict_1d, dict_2d, n_regions, weights)
    
    #STEP 3.  Obtain and check the connectivity matrix - indicates if a region pair is contiguous or not. 
    connectMatrix = gu.get_connectivity_matrix(xarray_dataset)

    silhouette_scores = []

    #STEP 4. Clustering for every number of regions from 1 to one less than n_regions 
    for i in range(1,n_regions):           #NOTE: each level in the hierarchy shows one merge. Looks like here it does not. 
                                            #Hence, for loop is used to perform clustering for every number of desired regions 
                                            #TODO: maybe investigate this?

        #STEP 2a. Hierarchical clustering with average linkage
        model = skc.AgglomerativeClustering(n_clusters=i,
                                            affinity='precomputed',
                                            linkage=linkage,
                                            connectivity=connectMatrix).fit(precomputed_dist_matrix)
        regions_label_list = model.labels_

        #STEP 2b. Silhouette Coefficient score 
        if i != 1:
            s = metrics.silhouette_score(precomputed_dist_matrix, regions_label_list, metric='precomputed')
            silhouette_scores.append(s)

        #STEP 2c. Aggregated regions dict
        regions_dict = {}
        for label in range(i):
            # Group the regions of this regions label
            sub_regions_list = list(regions_list[regions_label_list == label])
            sup_region_id = '_'.join(sub_regions_list)
            regions_dict[sup_region_id] = sub_regions_list.copy()

        aggregation_dict[i] = regions_dict.copy()

    #STEP 3. Plot the hierarchical tree dendrogram
    clustering_tree = skc.AgglomerativeClustering(distance_threshold=0, 
                                                    n_clusters=None, 
                                                    affinity='precomputed', 
                                                    linkage=linkage,
                                                    connectivity=connectMatrix).fit(precomputed_dist_matrix)

    #STEP 4. Cophenetic correlation coefficient                                              
    counts = np.zeros(clustering_tree.children_.shape[0])
    n_samples = len(clustering_tree.labels_)
    for i, merge in enumerate(clustering_tree.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
        
    linkage_matrix = np.column_stack([clustering_tree.children_, clustering_tree.distances_, counts]).astype(float)   
    
    distance_matrix = hierarchy.distance.squareform(precomputed_dist_matrix)
    print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(linkage_matrix, distance_matrix)[0])
    
    #STEP 5. Check for inconsistency                        #TODO: more info regarding this should be displayed. Otherwise, it is not very informative
    inconsistency = hierarchy.inconsistent(linkage_matrix)
    print('Inconsistencies:',list(inconsistency[:,3]))
    
    #STEP 6. Print Silhouette scores 
    print('Silhouette scores: ',silhouette_scores)


    return aggregation_dict



