"""Grouping algorithms to determine how to reduce a number of input regions to fewer regions while minimizing information loss.

"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

# Using SciPy.cluster for clustering
from scipy.cluster import hierarchy
from scipy.cluster import vq

# Using Scikit Learn for clustering
#from sklearn.cluster import KMeans
import sklearn.cluster as skc
from sklearn import metrics

import FINE.spagat.utils as spu

import FINE.spagat.grouping_utils as gu

logger_grouping = logging.getLogger("spagat_grouping")


def string_based_clustering(regions):
    """Creates a dictionary containing sup_regions and respective lists of sub_regions"""

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
    '''Cluster M regions based on centroid distance, hence closest regions are aggregated to obtain N regions.
    agg_modes -> 'sklearn_kmeans', 'sklearn_hierarchical', 'sklearn_spectral', 'scipy_kmeans', 'scipy_hierarchical'
    '''
    #TODO: maybe scipy can be dropped ?

    centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000  # km
    regions_list = sds.xr_dataset[dimension_description].values
    n_regions = len(regions_list)
    
    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

    ########################### SKLEARN - KMEANS #####################################

    if agg_mode == 'sklearn_kmeans':

        rss = [] # RSS (distortion) for different k values, in sklearn: inertia / within-cluster sum-of-squares

        for i in range(1,n_regions):

            # Compute K-Means clustering: configurations can be modified, e.g. init
            kmeans = skc.KMeans(n_clusters=i).fit(centroids)
            regions_label_list = kmeans.predict(centroids)
            rss.append(kmeans.inertia_)

            # Create a regions dictionary for the aggregated regions
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

            aggregation_dict[i] = regions_dict.copy()

        # Plotting the rss according to increase of k values, check if there exists an inflection point
        fig, ax = plt.subplots(figsize=(25, 12))
        ax.plot(range(1,n_regions),rss,'go-')
        ax.set_title('Within-cluster sum-of-squares')
        ax.set_xlabel('K (number_of_regions)')
        ax.set_ylabel('Distortion / Inertia')

        if save_path is not None:           #NOTE: fig saved before show() to avoid saving blanks 
            if fig_name is None: fig_name = 'sklearn_kmeans_distortion'

            spu.plt_savefig(path=save_path, save_name=fig_name)
        
        plt.show(block=False)

    ########################### SKLEARN - HIERARCHICAL #####################################

    if agg_mode == 'sklearn_hierarchical':

        for i in range(1,n_regions):

            # Computing hierarchical clustering
            model = skc.AgglomerativeClustering(n_clusters=i).fit(centroids)
            regions_label_list = model.labels_

            # Create a regions dictionary for the aggregated regions
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

            aggregation_dict[i] = regions_dict.copy()

        # Create linkage matrix for dendrogram
        clustering_tree = skc.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(centroids)
        ## Create the counts of samples under each node
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
            
        # Plot the hierarchical tree dendrogram
        distance_matrix = hierarchy.distance.pdist(centroids)
        

        # Evaluation 
        print('Statistics on this hiearchical clustering:')
        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(linkage_matrix, distance_matrix)[0])

        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(linkage_matrix)
        ax.plot(range(1,len(linkage_matrix)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency Coefficients: indicate where to cut the hierarchy', fontsize=14)
        ax.set_xlabel('Linkage height', fontsize=12)
        ax.set_ylabel('Inconsistencies', fontsize=12)

        plt.xticks(np.arange(1, len(linkage_matrix)+1, 1))
        plt.show(block=False)

        # If and how to save the hierarchical tree 
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


    ########################### SKLEARN - SPECTRAL #####################################    

    if agg_mode == 'sklearn_spectral':

        for i in range(1,n_regions):
            # Computing spectral clustering
            model = skc.SpectralClustering(n_clusters=i).fit(centroids)
            regions_label_list = model.labels_

            # Create a regions dictionary for the aggregated regions
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

            aggregation_dict[i] = regions_dict.copy()


    ########################### SCIPY - KMEANS #####################################  

    if agg_mode == 'scipy_kmeans':   
        # The input observations of kmeans must be normalized
        centroids_whitened = vq.whiten(centroids)

        rss = [] # RSS (distortion) for different k values - in vq.kmeans: average euclidean distance

        for i in range(1, n_regions):
            # Perform k-means on the original centroids to obtained k centroids of aggregated regions
            aggregation_centroids, distortion = vq.kmeans(centroids_whitened, i)
            
            rss.append(distortion)
            regions_label_list = vq.vq(centroids_whitened, aggregation_centroids)[0]
            
            # Create a regions dictionary for the aggregated regions
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

            aggregation_dict[i] = regions_dict.copy()

        # Plotting the rss according to increase of k values, check if there exists an inflection point
        fig, ax = spu.plt.subplots(figsize=(25, 12))
        ax.plot(range(1,n_regions),rss,'go-')
        ax.set_title('Impact of k on distortion')
        ax.set_xlabel('K (number_of_regions)')
        ax.set_ylabel('Distortion')

        if save_path is not None:
            if fig_name is None: fig_name = 'scipy_kmeans_distortion'
            spu.plt_savefig(path=save_path, save_name=fig_name) 

        plt.show(block=False)
        
        

    ########################### SCIPY - HIERARCHICAL ##################################### 

    if agg_mode == 'scipy_hierarchical':

        distance_matrix = hierarchy.distance.pdist(centroids)

        Z = hierarchy.linkage(distance_matrix, 'centroid') #Possibilities - 'centroid','average','weighted'
        
        #TODO: are the figures really required ? If yes, the function is misleading-> 
        # Inconsistency is plotted, dendrogram is saved. make this clearer

        #Evaluation 
        print('Statistics on this hiearchical clustering:')
        print('The cophentic correlation distance is ', hierarchy.cophenet(Z, distance_matrix)[0])
        
        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(Z)
        ax.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency Coefficients: indicate where to cut the hierarchy', fontsize=14) #TODO: Title should indicate where to cut, rather than just saying that it indicates so, 
        ax.set_xlabel('Linkage height', fontsize=12)
        ax.set_ylabel('Inconsistencies', fontsize=12)

        plt.xticks(np.arange(1, len(Z)+1, 1))
        plt.show(block=False)
        
        # If and how to save the hierarchical tree 
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

        
        regions_dict = {region_id: [region_id] for region_id in regions_list}
        regions_dict_complete = regions_dict.copy()

        # identify which regions are merged together (new_merged_region_id_list)
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
    
    # Original region list
    regions_list = sds.xr_dataset[dimension_description].values
    n_regions = len(regions_list)

    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}  #INFO: for 3 regions looks like -> {3: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']}}. Notice that it is dict within dict

    ''' 1. Using hierarchical clustering for all variables with custom defined distance
        - precomputed distance matrix using selfDistanceMatrix() function
        - linkage method: average
        - hierarchy clustering method of SciPy having spatial contiguity problem
        - hierarchy clustering method of Scikit learn solve spatial contiguity with additional connectivity matrix.
    '''

    ############################## SCIPY - HIERARCHICAL #########################################

    if agg_mode == 'scipy_hierarchical':

        #STEP 1.  Preprocess the whole dataset (grouping_utils - preprocessDataset())
        dict_ts, dict_1d, dict_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')  #TODO: this is a common step for all agg_modes, put it only once before if statements (keep in mind that the handle modes are different for different agg_modes)

        #STEP 2.  Calculate the overall distance between each region pair (uses custom distance)
        squared_dist_matrix = gu.selfDistanceMatrix(dict_ts, dict_1d, dict_2d, n_regions)

        #STEP 3. Clustering
        #STEP 3a.  Hierarchical clustering with average linkage
        distance_matrix = hierarchy.distance.squareform(squared_dist_matrix)
        Z = hierarchy.linkage(distance_matrix, method='average')

        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(Z, distance_matrix)[0])
        
        #### STEP 3b.  Figure for inconsistency check 
        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(Z)
        ax.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency of each Link with the Links Below', fontsize=14)
        ax.set_xlabel('Number of disjoint clusters under this link', fontsize=12)
        ax.set_ylabel('Inconsistency Coefficients', fontsize=12)

        plt.xticks(range(1,len(Z)+1), np.arange(len(Z)+1,1, -1))
        plt.show(block=False)  #INFO: block = False lets the computation continute during tests
                               # TODO: maybe save both figures and don't show them

        
        #STEP 3c.  If specified, figure for resulting dendrogram
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
        
        #STEP 4.  Find the sub_to_sup_region_id_dict for every level in the hierarchy
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
    
        #STEP 5.  Get Silhouette Coefficient scores
        silhouette_scores = gu.computeSilhouetteCoefficient(list(regions_list), squared_dist_matrix, aggregation_dict)
        print(silhouette_scores)


    ############################## SKLEARN - HIERARCHICAL #########################################

    if agg_mode == 'sklearn_hierarchical':  

        #STEP 1.  Preprocess the whole dataset (grouping_utils - preprocessDataset())
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')

        #STEP 2.  Calculate the overall distance between each region pair (uses custom distance)
        squared_distMatrix = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)

        #STEP 3.  Obtain a matrix where 1 means two regions are connected and 0 means not 
        # (any one of the component and any one of it's 2d variable has to have a positive value)
        connectMatrix = gu.generateConnectivityMatrix(sds)

        silhouette_scores = []

        #STEP 3. Clustering for every number of regions from 1 to one less than n_regions 
        for i in range(1,n_regions):           #NOTE: each level in the hierarchy shows one merge. Looks like her it does not. 
                                                #Hence, for loop is used to perform clustering for every number of desired regions 
                                                #TODO: maybe investigate this?

            # Computing hierarchical clustering
            model = skc.AgglomerativeClustering(n_clusters=i,affinity='precomputed',linkage='average',connectivity=connectMatrix).fit(squared_distMatrix)
            regions_label_list = model.labels_

            # Silhouette Coefficient score for this clustering results
            if i != 1:
                s = metrics.silhouette_score(squared_distMatrix, regions_label_list, metric='precomputed')
                silhouette_scores.append(s)

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sub_regions_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sub_regions_list)
                regions_dict[sup_region_id] = sub_regions_list.copy()

            aggregation_dict[i] = regions_dict.copy()

        #STEP 4. Plot the hierarchical tree dendrogram
        clustering_tree = skc.AgglomerativeClustering(distance_threshold=0, 
                                                      n_clusters=None, 
                                                      affinity='precomputed', 
                                                      linkage='average',
                                                      connectivity=connectMatrix).fit(squared_distMatrix)

        #STEP 4. Cophenetic correlation coefficient                                              
        # Create the counts of samples under each node
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
        # Plot the corresponding dendrogram
        #hierarchy.dendrogram(linkage_matrix)

        distance_matrix = hierarchy.distance.squareform(squared_distMatrix)
        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(linkage_matrix, distance_matrix)[0])
        
        #STEP 5. Check for inconsistency 
        #fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(linkage_matrix)
        print('Inconsistencies:',list(inconsistency[:,3]))
        # ax.plot(range(1,len(linkage_matrix)+1),list(inconsistency[:,3]),'go-')
        # ax.set_title('Inconsistency of each Link with the Links Below', fontsize=14)
        # ax.set_xlabel('Number of disjoint clusters under this link', fontsize=12)
        # ax.set_ylabel('Inconsistencies', fontsize=12)

        # plt.xticks(range(1,len(linkage_matrix)+1), np.arange(len(linkage_matrix)+1,1, -1))
        # plt.show()

        #STEP 6. Print Silhouette scores 
        print('Silhouette scores: ',silhouette_scores)


    ############################## SKLEARN - SPECTRAL #########################################
    
    ''' 2. Using spectral clustering with precomputed affinity matrix
        - precomputed affinity matrix by converting distance matrices to similarity matrix using RBF kernel
        - also having spatial contiguity problem due to the created complete graph
        - solve it by considering the additional connectivity matrix to cut some edges
    '''
    ## Affinity matrix: combine three matrices of ts_vars, 1d_vars and 2d_vars to one single precomputed affinity matrix


    if agg_mode == 'sklearn_spectral1': #TODO: change this to something more intuitive 

        '''Spectral clustering applied on the input dataset:
            - affinity matrices for 1d-Vars and 2d-Vars: 
                - given the feature matrix of samples, 
                - obtain its distance matrix based on the features
                - transform the distance matrix to a similarity matrix
                
            - affinity matrix for 2d-Vars: 
                - the original matrices can be regarded directly as the adjacency matrix of the graph
                - for each variable and each component: an adjacency matrix
                - multiple variables & multiple components: need to combine them as one affinity matrix (with weighting factors)
                - transform the adjacency matrix to affinity matrix:
                    - firstly get its reciprocal: now it is like a dissimilarity matrix
                    - then apply rbf kernel to obtain the similarity scores
        
            - If affinity is the adjacency matrix of a graph, this method can be used to find normalized graph cuts.

            - If you have an affinity matrix, such as a distance matrix, 
                - for which 0 means identical elements, 
                - and high value means very dissimilar elements, 
            it can be transformed in a similarity matrix that is well suited for the algorithm by applying the Gaussian (RBF, heat) kernel
        
            - Affinity matrix for spectral clustering input: a kind of similarity matrix
                - 1 means identical elements
                - high value means more similar elements (stronger connections)
        '''

        #STEP 1.  Preprocess the whole dataset (grouping_utils - preprocessDataset())
        feature_matrix_ts, feature_matrix_1d, adjacency_matrix_2d = gu.preprocessDataset(sds, handle_mode='toAffinity')

        # List of weighting factors for 3 categories
        if weighting is None: weighting = [1, 1, 1]
                                      
        # delta value for RBF kernel -> to construct affinity matrix
        delta = 1

        #STEP 2a. (i) Obtain distance matrix for time series variable set (used pdist, which in turn uses default euclidean distance)
        distance_matrix_ts = hierarchy.distance.squareform(hierarchy.distance.pdist(feature_matrix_ts))  #NOTE: pdist finds euclidean distance between regions,
                                                                                                         # here, hierarchy.distance.squareform converts this condensed matrix (rather a list) to a symmetric, hollow matrix
        #STEP 2a. (ii) Use RBF kernel to construct affinity matrix based on distance matrix of time series variable set
        affinity_ts = np.exp(- distance_matrix_ts ** 2 / (2. * delta ** 2))

        #STEP 2b. (i) Obtain distance matrix for 1d variable set (used pdist, which in turn uses default euclidean distance)
        distance_matrix_1d = hierarchy.distance.squareform(hierarchy.distance.pdist(feature_matrix_1d))
        #STEP 2b. (ii) Use RBF kernel to construct affinity matrix based on distance matrix of 1d variable set
        affinity_1d = np.exp(- distance_matrix_1d ** 2 / (2. * delta ** 2))

        #STEP 2c. (i) Obtain distance matrix for 2d variable set (used pdist, which in turn uses default euclidean distance)
        # Convert the adjacency matrix to a dissimilarity matrix similar to a distance matrix: high value=more dissimilarity, 0=identical elements  #TODO: why is this required?? why cant we set handle mode to dissimilarity directly??
        adjacency_2d_adverse = 1.0 / adjacency_matrix_2d   #NOTE: #adjacency_matrix_2d is affinity matrix, convert it into distance matrix by taking it's reciprocal
        max_value = adjacency_2d_adverse[np.isfinite(adjacency_2d_adverse)].max()  #NOTE: find the maximum value which is not infinity
        adjacency_2d_adverse[np.isinf(adjacency_2d_adverse)] = max_value + 10  #NOTE: infinity = max value + 10 (meant for non-connected regions)
        np.fill_diagonal(adjacency_2d_adverse,0)                   #NOTE: set diagonal values to 0 (meant for same region pair in the matrix)       

        #STEP 2c. (ii) Use RBF kernel to construct affinity matrix based on distance matrix of 2d variable set
        affinity_2d = np.exp(- adjacency_2d_adverse ** 2 / (2. * delta ** 2))

        #STEP 3. Compute a single affinity matrix
        affinity_matrix = (affinity_ts * weighting[0] + affinity_1d * weighting[1] + affinity_2d * weighting[2]) 

        # ##### Solve the spatial contiguity problem with the connectivity condition
        # # Connectivity matrix for neighboring structure
        # connectMatrix = gu.generateConnectivityMatrix(sds)
        # # Cut down the edges that have zero value in connectivity matrix
        # affinity_matrix[connectMatrix==0] = 0

        # Evaluation indicators
        modularities = []
        #STEP 4. For 1 to one less than n regions: Perform the following sub steps
        for i in range(1,n_regions):
            #STEP 4a. clustering
            model = skc.SpectralClustering(n_clusters=i,affinity='precomputed').fit(affinity_matrix)
            regions_label_list = model.labels_

            #STEP 4b. compute modulatiy (calls computeModularity() )
            modularity = gu.computeModularity(affinity_matrix, regions_label_list)
            modularities.append(modularity)

            #STEP 4c. form resulting sub_to_sup_region_id_dict 
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sub_regions_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sub_regions_list)
                regions_dict[sup_region_id] = sub_regions_list.copy()

            aggregation_dict[i] = regions_dict.copy()
    
        # Plotting the modularites according to increase of k values, check if there exists an inflection point
        # fig, ax = spu.plt.subplots(figsize=(25, 12))
        # ax.plot(range(1,n_regions),modularities,'go-')
        # ax.set_title('Impact of aggregated regions on modularity')
        # ax.set_xlabel('number of aggregated regions')
        # ax.set_ylabel('Modularity')
        # plt.show()

        print('Modularities',modularities)

        #### STEP 5. Obtain Silhouette scores
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')
        distances = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)
        silhouette_scores = gu.computeSilhouetteCoefficient(list(regions_list), distances, aggregation_dict)
        print('Silhouette scores: ',silhouette_scores)
    
    ############################## SKLEARN - SPECTRAL #########################################

    ## Affinity matrix: construct a distance matrix based on selfDistanceMatrix function, transform it to similarity matrix
    if agg_mode =='sklearn_spectral2':

        #STEP 1.  Preprocess the whole dataset (grouping_utils - preprocessDataset())
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')

        #STEP 2.  Calculate the overall distance between each region pair (uses custom distance)
        distMatrix = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)

        #STEP 3. Scale the distance matrix between 0 and 1
        distMatrix = gu.matrix_MinMaxScaler(distMatrix)

        #STEP 4. Use RBF kernel to construct affinity matrix based on distance matrix
        delta = 1
        affinity_matrix = np.exp(- distMatrix ** 2 / (2. * delta ** 2))

        # # Connectivity matrix for neighboring structure
        # connectMatrix = gu.generateConnectivityMatrix(sds)
        # # Cut down the edges that have zero value in connectivity matrix
        # affinity_matrix[connectMatrix==0] = 0

        # Evaluation indicators
        modularities = []
        # Silhouette Coefficient scores
        silhouette_scores = []

        #STEP 5. For 1 to one less than n regions: Perform the following sub steps 
        for i in range(1,n_regions):
            #STEP 5a. clustering
            model = skc.SpectralClustering(n_clusters=i,affinity='precomputed').fit(affinity_matrix)
            regions_label_list = model.labels_

            #STEP 5b. compute modulatiy (calls computeModularity() )
            modularity = gu.computeModularity(affinity_matrix, regions_label_list)
            modularities.append(modularity)

            #STEP 5c. Obtain Silhouette Coefficient score (skip for n_region=1 as this score can be computed only n_regions = 2 : n-1 regions)
            if i != 1:
                s = metrics.silhouette_score(distMatrix, regions_label_list, metric='precomputed')
                silhouette_scores.append(s)

            #STEP 5d. form resulting sub_to_sup_region_id_dict 
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sub_regions_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sub_regions_list)
                regions_dict[sup_region_id] = sub_regions_list.copy()

            aggregation_dict[i] = regions_dict.copy()

        # Plotting the modularites according to increase of k values, check if there exists an inflection point
        # fig, ax = spu.plt.subplots(figsize=(25, 12))
        # ax.plot(range(1,n_regions),modularities,'go-')
        # ax.set_title('Impact of aggregated regions on modularity')
        # ax.set_xlabel('number of aggregated regions')
        # ax.set_ylabel('Modularity')
        # plt.show()

        print('Modularites: ',modularities)

        print('Silhouette scores: ',silhouette_scores)

    return aggregation_dict



