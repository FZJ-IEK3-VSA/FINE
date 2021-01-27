"""Grouping algorithms determine how to reduce a number of input regions to 
fewer regions while minimizing information loss.
"""
import os
import logging

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from sklearn import metrics
from scipy.cluster import vq
import sklearn.cluster as skc
from typing import Dict, List

import FINE.spagat.utils as spu
import FINE.spagat.grouping_utils as gu

logger_grouping = logging.getLogger("spagat_grouping")


def string_based_clustering(regions):
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
    # TODO: this is implemented spefically for the e-id: '01_es' -> generalize this!
    nation_set = set([region_id.split("_")[1] for region_id in regions])

    sub_to_sup_region_id_dict = {}

    for nation in nation_set:
        sub_to_sup_region_id_dict[nation] = [
            region_id for region_id in regions if region_id.split("_")[1] == nation
        ]

    return sub_to_sup_region_id_dict


@spu.timer
def distance_based_clustering(sds, agg_mode = 'sklearn_hierarchical', 
                            dimension_description='space',
                            ax_illustration=None, 
                            save_path = None, 
                            fig_name=None, 
                            verbose=False):
    """Groups regions based on the regions' centroid distances. 

    Parameters
    ----------
    sds : Instance of SpagatDataset
        Refer to SpagatDataset class in dataset.py for more information 
    agg_mode : {'sklearn_hierarchical', 'sklearn_kmeans', 'sklearn_spectral', 'scipy_kmeans', 'scipy_hierarchical'}, optional 
        Specifies which python package and which clustering method to choose for grouping 
    dimension_description : str, optional (default='space')
        The name/description of the dimension in the sds data that corresponds to regions 
    ax_illustration : Axis
        Provide axis to an existing figure, to include the generated plots to the same figure 
    save_path :  str, optional (default=None)
        The path to which to save the figures. 
        If default None, figure is not saved
    fig_name : str, optional (default=None)
        Name of the saved figure. 
        Valid only if `save_path` is not None. 
        If default None, a default name is chosen based on the chosen `agg_mode`:
            - 'sklearn_hierarchical' -> 'sklearn_hierarchical_dendrogram'
            - 'sklearn_kmeans' -> 'sklearn_kmeans_distortion'
            - 'sklearn_spectral' -> NO FIGURE IS SAVED IN THIS MODE 
            - 'scipy_kmeans' -> 'scipy_kmeans_distortion'
            - 'scipy_hierarchical' -> 'scipy_hierarchical_dendrogram'
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
    #TODO: maybe scipy can be dropped ?

    centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000  # km
    regions_list = sds.xr_dataset[dimension_description].values
    n_regions = len(regions_list)
    
    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

    #================= SKLEARN - KMEANS =================#
    if agg_mode == 'sklearn_kmeans':
        rss = [] # RSS (distortion) for different k values, in sklearn: inertia / within-cluster sum-of-squares

        #STEP 1. Clustering for each number of regions from 1 to total number present in the original resolution
        for i in range(1, n_regions):

            #STEP 1a. Compute K-Means clustering
            kmeans = skc.KMeans(n_clusters=i).fit(centroids)
            regions_label_list = kmeans.predict(centroids)

            #STEP 1b. Save rss value
            rss.append(kmeans.inertia_)

            #STEP 1c. Create a regions dictionary for the aggregated regions
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

            #STEP 1d. Append the dict to main dict 
            aggregation_dict[i] = regions_dict.copy()

        #STEP 2. Plot rss values (check if there exists an inflection point) #TODO: indicate this inflection point on the diagram, del the text in brackets
        fig, ax = plt.subplots(figsize=(25, 12))
        ax.plot(range(1,n_regions),rss,'go-')
        ax.set_title('Within-cluster sum-of-squares')
        ax.set_xlabel('K (number_of_regions)')
        ax.set_ylabel('Distortion / Inertia')

        #STEP 3. Save fig if user specifies 
        if save_path is not None:           #NOTE: fig saved before show() to avoid saving blanks 
            if fig_name is None: fig_name = 'sklearn_kmeans_distortion'
            spu.plt_savefig(path=save_path, save_name=fig_name)

        plt.show(block=False)

    #================= SKLEARN - HIERARCHICAL =================#
    if agg_mode == 'sklearn_hierarchical':

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

        if ax_illustration is not None:
            R = hierarchy.dendrogram(linkage_matrix, 
                                    orientation="top",
                                    labels=sds.xr_dataset[dimension_description].values, 
                                    ax=ax_illustration, 
                                    leaf_font_size=14
                                    )

            if save_path is not None:
                spu.plt_savefig(save_name=fig_name, path=save_path)  #TODO: not at all sure if this will work. remove it or test it!

        elif save_path is not None:
            fig, ax = plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(linkage_matrix, 
                                    orientation="top",
                                    labels=sds.xr_dataset[dimension_description].values, 
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

    #================= SKLEARN - SPECTRAL =================#    
    if agg_mode == 'sklearn_spectral':

        #STEP 1. Clustering for each number of regions from 1 to total number present in the original resolution
        for i in range(1,n_regions):
            #STEP 1a. Compute spectral clustering 
            model = skc.SpectralClustering(n_clusters=i).fit(centroids)
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


    #================= SCIPY - KMEANS =================#  
    if agg_mode == 'scipy_kmeans':   
        #STEP 1. Normalize input observations (required to run kmeans)
        centroids_whitened = vq.whiten(centroids)

        rss = [] # RSS (distortion) for different k values - in vq.kmeans: average euclidean distance

        #STEP 1. Clustering for each number of regions from 1 to total number present in the original resolution
        for i in range(1, n_regions):
            #STEP 1a. Compute kmeans clustering
            aggregation_centroids, distortion = vq.kmeans(centroids_whitened, i)
            regions_label_list = vq.vq(centroids_whitened, aggregation_centroids)[0]

            #STEP 1b. Save rss value
            rss.append(distortion)
            
            #STEP 1c. Create a regions dictionary for the aggregated regions
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

            #STEP 1d. Append the dict to main dict 
            aggregation_dict[i] = regions_dict.copy()

        #STEP 2. Plot rss values (check if there exists an inflection point) #TODO: indicate this inflection point on the diagram, del the text in brackets
        fig, ax = spu.plt.subplots(figsize=(25, 12))
        ax.plot(range(1,n_regions),rss,'go-')
        ax.set_title('Impact of k on distortion')
        ax.set_xlabel('K (number_of_regions)')
        ax.set_ylabel('Distortion')

        #STEP 3. Save fig if user specifies
        if save_path is not None:
            if fig_name is None: fig_name = 'scipy_kmeans_distortion'
            spu.plt_savefig(path=save_path, save_name=fig_name) 

        plt.show(block=False)
        

    #================= SCIPY - HIERARCHICAL =================# 
    if agg_mode == 'scipy_hierarchical':
        #STEP 1. Compute hieracical clustering
        distance_matrix = hierarchy.distance.pdist(centroids)
        Z = hierarchy.linkage(distance_matrix, 'centroid') 
        
        #TODO: are the figures really required ? If yes, the function is misleading-> 
        # Inconsistency is plotted, dendrogram is saved. make this clearer

        #STEP 2. Evaluation 
        print('Statistics on this hiearchical clustering:')

        #STEP 2a. Cophenetic correlation coefficient
        cophenetic_correlation_coefficient = hierarchy.cophenet(Z, distance_matrix)[0]
        print('The cophenetic correlation coefficient is ', cophenetic_correlation_coefficient)
        
        #STEP 2b. Inconsistency coefficients (in a plot)
        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(Z)
        ax.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency Coefficients: indicate where to cut the hierarchy', fontsize=14) #TODO: Title should indicate where to cut, rather than just saying that it indicates so, 
        ax.set_xlabel('Linkage height', fontsize=12)
        ax.set_ylabel('Inconsistencies', fontsize=12)

        plt.xticks(np.arange(1, len(Z)+1, 1))
        plt.show(block=False)
        
        #STEP 3. Create and save dendrogram (from linkage matrix) if user specifies 
        if fig_name is None: fig_name = 'scipy_hierarchical_dendrogram'

        if ax_illustration is not None:   #TODO: is ax_illustration thing really necessary?
            R = hierarchy.dendrogram(
                Z,
                orientation="top",
                labels=sds.xr_dataset[dimension_description].values,
                ax=ax_illustration,
                leaf_font_size=14,
            )

            if save_path is not None:
                spu.plt_savefig(save_name=fig_name, path=save_path)

        elif save_path is not None:
            fig, ax = plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(
                Z,
                orientation="top",
                labels=sds.xr_dataset[dimension_description].values,
                ax=ax,
                leaf_font_size=14,
            )

            spu.plt_savefig(save_name=fig_name, path=save_path)

        #STEP 4. Create resuting aggregation dict for all levels of hierarchy
        regions_dict = {region_id: [region_id] for region_id in regions_list}
        regions_dict_complete = regions_dict.copy()

        ## Identify which regions are merged in each level of the hierarchy 
        for i in range(len(Z)):

            ### Identify the keys of the sub regions that will be merged
            key_list = list(regions_dict_complete.keys())
            key_1 = key_list[int(Z[i][0])]
            key_2 = key_list[int(Z[i][1])]

            ### Get the region_id_list_s of the sub regions
            value_list = list(regions_dict_complete.values())
            sub_region_id_list_1 = value_list[int(Z[i][0])]
            sub_region_id_list_2 = value_list[int(Z[i][1])]

            ### Add the new region to the dict by merging the two region_id_lists
            sup_region_id = f"{key_1}_{key_2}"

            sup_region_id_list = sub_region_id_list_1.copy()
            sup_region_id_list.extend(sub_region_id_list_2)

            regions_dict_complete[sup_region_id] = sup_region_id_list

            regions_dict[sup_region_id] = sup_region_id_list

            del regions_dict[key_1]
            del regions_dict[key_2]

            if verbose:                     #TODO: maybe remove verbose 
                print(i)
                print("\t", "keys:", key_1, key_2)
                print("\t", "list_1", sub_region_id_list_1)
                print("\t", "list_2", sub_region_id_list_2)
                print("\t", "sup_region_id", sup_region_id)
                print("\t", "sup_region_id_list", sup_region_id_list)

            aggregation_dict[n_regions - i - 1] = regions_dict.copy()

        
    return aggregation_dict



@spu.timer
def all_variable_based_clustering(sds, agg_mode='scipy_hierarchical', 
                                dimension_description='space',
                                ax_illustration=None, 
                                save_path=None, 
                                fig_name=None,  
                                verbose=False,
                                weighting=None):
    """Groups regions based on the Energy System Model instance's data. 

    Parameters
    ----------
    sds : Instance of SpagatDataset
        Refer to SpagatDataset class in dataset.py for more information 
    agg_mode : {'scipy_hierarchical', 'sklearn_hierarchical', 
                'spectral_with_precomputedAffinity', 'spectral_with_RBFaffinity'}, optional 
        Specifies which python package and which clustering method to choose for grouping 
    dimension_description : str, optional (default='space')
        The name/description of the dimension in the sds data that corresponds to regions 
    ax_illustration : Axis
        Provide axis to an existing figure, to include the generated plots to the same figure 
    save_path :  str, optional (default=None)
        The path to which to save the figure. 
        Valid only if `agg_mode` is 'scipy_hierarchical'
        If default None, figure is not saved
    fig_name : str, optional (default=None)
        Name of the saved figure. 
        Valid only if `save_path` is not None. 
        If default None, the figure is saved under the name 'scipy_hierarchical_dendrogram'    
    verbose : bool, optional (default=False)
        If True, the grouping results are printed. Supressed if False 
    weighting : List[int], optional (default=None)
        List containing weights for each part i.e., time series, 1d and 2d variables 
        If default None, each part has same weight of 1
        
    Returns
    -------
    aggregation_dict : Dict[int, Dict[str, List[str]]]
        A nested dictionary containing results of spatial grouping at various levels/number of groups
        - Ex. {3: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']},
               2: {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg': ['03_reg']},
               1: {'01_reg_02_reg_03_reg': ['01_reg','02_reg','03_reg']}}
    
    Notes
    -----
    * Various functions from grouping_utils module are employed here depending on the `agg_mode`. 
    * While clustering/grouping regions, it is important to make sure that the regions 
      are spatially contiguous. Different `agg_mode`s handle this problem differently. 

    Information regarding each `agg_mode` can be found below:
    1. 'scipy_hierarchical' 
        - Preprocessing data -> preprocessDataset() with handle_mode='toDissimilarity'
        - Custom distance -> selfDistanceMatrix()
        - Clustering method -> scipy's agglomerative hierarchical clustering with average linkage 
        - Spatial contiguity -> Clustering method internally handles spatial contiguity problem #TODO: verify this 
        - Figures -> (a) Inconsistencies are displayed in a plot. 
                         However, it is not saved!
                     (b) A dendrogram is directly saved if `save_path` is specified. 
                         However, it is not displayed! 
        - Additional accuracy indicators -> (a) Cophenetic correlation coefficients are printed 
                                           (b) Silhouette scores are computed and printed. 

    2. 'sklearn_hierarchical'
        - Preprocessing data -> preprocessDataset() with handle_mode='toDissimilarity'
        - Custom distance -> selfDistanceMatrix()
        - Clustering method -> sklearn's agglomerative hierarchical clustering with average linkage  
        - Spatial contiguity -> Connectivity matrix is passed to the clustering method. 
                                generateConnectivityMatrix() to obtain Connectivity matrix.
        - Accuracy indicators -> (a) Cophenetic correlation coefficients are printed
                                (b) Inconsistencies are printed. 
                                (c) Silhouette scores are printed. 

    3. 'spectral_with_precomputedAffinity'
        - Preprocessing data -> preprocessDataset() with handle_mode='toAffinity'
        - Clustering method -> sklearn's spectral clustering 
        - Affinity calculation -> - Clustering method requires affinity matrix. 
                                  - Constructed using RBF kernel, separately for all 
                                    3 parts i.e., time series, 1d and 2d variables. 
                                    This takes as it's input the results from Preprocessing step.
                                  - `weighting` is used to weight the 3 affinity parts and all parts 
                                    are then added.
        - Spatial contiguity -> Clustering method internally handles spatial contiguity problem 
        - Accuracy indicators -> (a) Modularites are computed and printed. 
                                 (b) Silhouette scores are computed and printed. 
        
    4. 'spectral_with_RBFaffinity'
        - Preprocessing data -> preprocessDataset() with handle_mode='toDissimilarity'
        - Custom distance -> selfDistanceMatrix()
        - Clustering method -> sklearn's spectral clustering
        - Affinity calculation -> Clustering method requires affinity matrix. 
                                  Constructed using RBF kernel. This takes as it's input 
                                  the scaled Custom distance. 
        - Spatial contiguity -> Clustering method internally handles spatial contiguity problem 
        - Accuracy indicators -> (a) Modularites are computed and printed. 
                                 (b) Silhouette scores are printed. 
    
    For more information, please refer to the Master Thesis of Shu Xing -> https://fz-juelich.sciebo.de/s/Cy7gIdoNjqShmzz
    """
    # Original region list
    regions_list = sds.xr_dataset[dimension_description].values
    n_regions = len(regions_list)

    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}  #INFO: for 3 regions looks like -> 
                                                                                          # {3: {'01_reg': ['01_reg'], 
                                                                                          # '02_reg': ['02_reg'], 
                                                                                          # '03_reg': ['03_reg']}}. 
                                                                                          # Notice that it is dict within dict
    #STEP 0. (Same for all agg_mode except spectral_with_precomputedAffinity)
    if agg_mode is not 'spectral_with_precomputedAffinity':
        #STEP 0a. Preprocess the whole dataset with handle_mode='toDissimilarity'
        dict_ts, dict_1d, dict_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity') 

        #STEP 0b. Calculate the overall distance between each region pair (uses custom distance)
        precomputed_dist_matrix = gu.selfDistanceMatrix(dict_ts, dict_1d, dict_2d, n_regions)
    else:
        #STEP 0a. Preprocess the whole dataset with handle_mode='toAffinity'
        feature_matrix_ts, feature_matrix_1d, adjacency_matrix_2d = gu.preprocessDataset(sds, handle_mode='toAffinity')

    #============================ SCIPY - HIERARCHICAL ============================#
    if agg_mode == 'scipy_hierarchical':

        #STEP 1. CLUSTERING

        #STEP 1a.  Hierarchical clustering with average linkage
        distance_matrix = hierarchy.distance.squareform(precomputed_dist_matrix)
        Z = hierarchy.linkage(distance_matrix, method='average')

        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(Z, distance_matrix)[0])
        
        ##STEP 1b.  Figure for inconsistency check 
        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(Z)
        ax.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency of each Link with the Links Below', fontsize=14)
        ax.set_xlabel('Number of disjoint clusters under this link', fontsize=12)
        ax.set_ylabel('Inconsistency Coefficients', fontsize=12)

        plt.xticks(range(1,len(Z)+1), np.arange(len(Z)+1,1, -1))
        plt.show(block=False)  #INFO: block = False lets the computation continute during tests
                               # TODO: maybe save both figures and don't show them

        
        #STEP 1c.  If specified, figure for resulting dendrogram
        if fig_name is None: fig_name = 'scipy_hierarchical_dendrogram'
        if ax_illustration is not None:                        #TODO: ax can be passed like axes[1] if the figure is a subplot of a plot. check how helpful this is. based on it maybe simplify or eliminate this
            R = hierarchy.dendrogram(Z, 
                                    orientation="top",
                                    labels=sds.xr_dataset[dimension_description].values, 
                                    ax=ax_illustration, 
                                    leaf_font_size=14
                                    )

            if save_path is not None:
                spu.plt_savefig(save_name=fig_name, path=save_path)

        elif save_path is not None:
            fig, ax = spu.plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(Z, 
                                    orientation="top",
                                    labels=sds.xr_dataset[dimension_description].values, 
                                    ax=ax, 
                                    leaf_font_size=14
                                    )
            
            spu.plt_savefig(save_name=fig_name, path=save_path)
        
        #STEP 2.  Find the sub_to_sup_region_id_dict for every level in the hierarchy
        regions_dict = {region_id: [region_id] for region_id in regions_list}
        regions_dict_complete = regions_dict.copy()                  #INFO: you can't assign regions_dict directly to regions_dict_complete in Python. You have to copy it. else manipulating one will change the other

        # Identify, which regions are merged together (new_merged_region_id_list)
        for i in range(len(Z)):

            # identify the keys of the sub regions that will be merged
            key_list = list(regions_dict_complete.keys())
            key_1 = key_list[int(Z[i][0])]
            key_2 = key_list[int(Z[i][1])]

            # get the region_id_list_s of the sub regions
            value_list = list(regions_dict_complete.values())
            sub_region_id_list_1 = value_list[int(Z[i][0])]
            sub_region_id_list_2 = value_list[int(Z[i][1])]

            # add the new region to the dict by merging the two region_id_lists
            sup_region_id = f'{key_1}_{key_2}'
            sup_region_id_list = sub_region_id_list_1.copy()
            sup_region_id_list.extend(sub_region_id_list_2)

            regions_dict_complete[sup_region_id] = sup_region_id_list
            regions_dict[sup_region_id] = sup_region_id_list
            del regions_dict[key_1]
            del regions_dict[key_2]

            if verbose:
                print(i)
                print('\t', 'keys:', key_1, key_2)
                print('\t', 'list_1', sub_region_id_list_1)
                print('\t', 'list_2', sub_region_id_list_2)
                print('\t', 'sup_region_id', sup_region_id)
                print('\t', 'sup_region_id_list', sup_region_id_list)

            aggregation_dict[n_regions - i - 1] = regions_dict.copy()
    
        #STEP 3.  Get Silhouette Coefficient scores
        silhouette_scores = gu.computeSilhouetteCoefficient(list(regions_list), precomputed_dist_matrix, aggregation_dict)
        print(silhouette_scores)


    #======================== SKLEARN - HIERARCHICAL ========================#

    if agg_mode == 'sklearn_hierarchical':  

        #STEP 1.  Obtain a matrix where 1 means two regions are connected and 0 means not 
        # (any one of the component and any one of it's 2d variable has to have a positive value)
        connectMatrix = gu.generateConnectivityMatrix(sds)

        silhouette_scores = []

        #STEP 2. Clustering for every number of regions from 1 to one less than n_regions 
        for i in range(1,n_regions):           #NOTE: each level in the hierarchy shows one merge. Looks like her it does not. 
                                                #Hence, for loop is used to perform clustering for every number of desired regions 
                                                #TODO: maybe investigate this?

            #STEP 2a. Hierarchical clustering with average linkage
            model = skc.AgglomerativeClustering(n_clusters=i,
                                                affinity='precomputed',
                                                linkage='average',
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
                                                      linkage='average',
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
        
        #STEP 5. Check for inconsistency 
        inconsistency = hierarchy.inconsistent(linkage_matrix)
        print('Inconsistencies:',list(inconsistency[:,3]))
        
        #STEP 6. Print Silhouette scores 
        print('Silhouette scores: ',silhouette_scores)


    #======================== SPECTRAL with precomputed affinity matrix ========================#
    
    if agg_mode == 'spectral_with_precomputedAffinity':  
        # List of weighting factors for 3 categories
        if weighting is None: weighting = [1, 1, 1]
                                      
        # delta value for RBF kernel -> to construct affinity matrix
        delta = 1

        #STEP 1a. (i) Obtain distance matrix for time series variable set (used pdist, which in turn uses default euclidean distance)
        distance_matrix_ts = hierarchy.distance.squareform(hierarchy.distance.pdist(feature_matrix_ts))  #INFO: pdist finds euclidean distance between regions,
                                                                                                         # here, hierarchy.distance.squareform converts this condensed matrix (rather a list) to a symmetric, hollow matrix
        #STEP 1a. (ii) Use RBF kernel to construct affinity matrix based on distance matrix of time series variable set
        affinity_ts = np.exp(- distance_matrix_ts ** 2 / (2. * delta ** 2))

        #STEP 1b. (i) Obtain distance matrix for 1d variable set (used pdist, which in turn uses default euclidean distance)
        distance_matrix_1d = hierarchy.distance.squareform(hierarchy.distance.pdist(feature_matrix_1d))
        #STEP 1b. (ii) Use RBF kernel to construct affinity matrix based on distance matrix of 1d variable set
        affinity_1d = np.exp(- distance_matrix_1d ** 2 / (2. * delta ** 2))

        #STEP 1c. (i) Obtain distance matrix for 2d variable set (used pdist, which in turn uses default euclidean distance)
        # Convert the adjacency matrix to a dissimilarity matrix similar to a distance matrix: high value=more dissimilarity, 0=identical elements  #TODO: why is this required?? why cant we set handle mode to dissimilarity directly??
        adjacency_2d_adverse = 1.0 / adjacency_matrix_2d   #NOTE: #adjacency_matrix_2d is affinity matrix, convert it into distance matrix by taking it's reciprocal
        max_value = adjacency_2d_adverse[np.isfinite(adjacency_2d_adverse)].max()  #NOTE: find the maximum value which is not infinity
        adjacency_2d_adverse[np.isinf(adjacency_2d_adverse)] = max_value + 10  #NOTE: infinity = max value + 10 (meant for non-connected regions)
        np.fill_diagonal(adjacency_2d_adverse,0)                   #NOTE: set diagonal values to 0 (meant for same region pair in the matrix)       

        #STEP 1c. (ii) Use RBF kernel to construct affinity matrix based on distance matrix of 2d variable set
        affinity_2d = np.exp(- adjacency_2d_adverse ** 2 / (2. * delta ** 2))

        #STEP 2. Compute a single affinity matrix
        affinity_matrix = (affinity_ts * weighting[0] + affinity_1d * weighting[1] + affinity_2d * weighting[2]) 

        # Evaluation indicators
        modularities = []
        #STEP 3. For 1 to one less than n regions: Perform the following sub steps
        for i in range(1,n_regions):
            #STEP 3a. clustering
            model = skc.SpectralClustering(n_clusters=i,affinity='precomputed').fit(affinity_matrix)
            regions_label_list = model.labels_

            #STEP 3b. compute modulatiy (calls computeModularity() )
            modularity = gu.computeModularity(affinity_matrix, regions_label_list)
            modularities.append(modularity)

            #STEP 3c. form resulting sub_to_sup_region_id_dict 
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sub_regions_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sub_regions_list)
                regions_dict[sup_region_id] = sub_regions_list.copy()

            aggregation_dict[i] = regions_dict.copy()
    
        print('Modularities',modularities)

        #STEP 4. Obtain Silhouette scores
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')
        distances = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)
        silhouette_scores = gu.computeSilhouetteCoefficient(list(regions_list), distances, aggregation_dict)
        print('Silhouette scores: ',silhouette_scores)
    
    #======================== SPECTRAL with RBF affinity ========================#
    if agg_mode =='spectral_with_RBFaffinity':

        #STEP 1. Scale the distance matrix between 0 and 1
        scaled_dist_matrix = gu.matrix_MinMaxScaler(precomputed_dist_matrix)

        #STEP 2. Use RBF kernel to construct affinity matrix based on scaled distance matrix
        delta = 1
        affinity_matrix = np.exp(- scaled_dist_matrix ** 2 / (2. * delta ** 2))

        # Evaluation indicators
        modularities = []
        # Silhouette Coefficient scores
        silhouette_scores = []

        #STEP 3. For 1 to one less than n regions: Perform the following sub steps 
        for i in range(1,n_regions):
            #STEP 3a. clustering
            model = skc.SpectralClustering(n_clusters=i,affinity='precomputed').fit(affinity_matrix)
            regions_label_list = model.labels_

            #STEP 3b. compute modulatiy (calls computeModularity() )
            modularity = gu.computeModularity(affinity_matrix, regions_label_list)
            modularities.append(modularity)

            #STEP 3c. Obtain Silhouette Coefficient score (skip for n_region=1 as this score can be computed only n_regions = 2 : n-1 regions)
            if i != 1:
                s = metrics.silhouette_score(scaled_dist_matrix, regions_label_list, metric='precomputed')
                silhouette_scores.append(s)

            #STEP 3d. form resulting sub_to_sup_region_id_dict 
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sub_regions_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sub_regions_list)
                regions_dict[sup_region_id] = sub_regions_list.copy()

            aggregation_dict[i] = regions_dict.copy()

        print('Modularites: ',modularities)

        print('Silhouette scores: ',silhouette_scores)

    return aggregation_dict



