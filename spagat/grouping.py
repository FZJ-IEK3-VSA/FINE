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

import spagat.utils as spu

import spagat.grouping_utils as gu

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
def distance_based_clustering(sds, agg_mode, verbose=False, ax_illustration=None, save_fig=None, dimension_description='space'):
    '''Cluster M regions based on centroid distance, hence closest regions are aggregated to obtain N regions.'''
    
    centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000  # km
    regions_list = sds.xr_dataset[dimension_description].values
    n_regions = len(regions_list)

    '''Clustering methods via SciPy.cluster module'''

    if agg_mode == 'hierarchical':

        distance_matrix = hierarchy.distance.pdist(centroids)

        # Various methods, e.g. 'average', 'weighted', 'centroid' -> representation of new clusters 
        Z = hierarchy.linkage(distance_matrix, 'centroid')

        # Evaluation of this clustering methods
        print('Statistics on this hiearchical clustering:')
        
        print('The cophentic correlation distance is ', hierarchy.cophenet(Z, distance_matrix)[0])
        
        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(Z)
        ax.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency Coefficients: indicate where to cut the hierarchy', fontsize=14)
        ax.set_xlabel('Linkage height', fontsize=12)
        ax.set_ylabel('Inconsistencies', fontsize=12)

        plt.xticks(np.arange(1, len(Z)+1, 1))
        plt.show()
        
        # If and how to save the hierarchical tree 
        if ax_illustration is not None:
            R = hierarchy.dendrogram(
                Z,
                orientation="top",
                labels=sds.xr_dataset[dimension_description].values,
                ax=ax_illustration,
                leaf_font_size=14,
            )

            if save_fig is not None:

                spu.plt_savefig(save_name=save_fig)
        elif save_fig is not None:

            fig, ax = spu.plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(
                Z,
                orientation="top",
                labels=sds.xr_dataset[dimension_description].values,
                ax=ax,
                leaf_font_size=14,
            )

            spu.plt_savefig(fig=fig, save_name=save_fig)

        #n_regions = len(Z)

        aggregation_dict = {}

        regions_dict = {
            region_id: [region_id]
            for region_id in list(sds.xr_dataset[dimension_description].values)
        }

        regions_dict_complete = {
            region_id: [region_id]
            for region_id in list(sds.xr_dataset[dimension_description].values)
        }

        aggregation_dict[n_regions] = regions_dict.copy()

        all_region_id_list = []

        # identify, which regions are merged together (new_merged_region_id_list)
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

            if verbose:
                print(i)
                print("\t", "keys:", key_1, key_2)
                print("\t", "list_1", sub_region_id_list_1)
                print("\t", "list_2", sub_region_id_list_2)
                print("\t", "sup_region_id", sup_region_id)
                print("\t", "sup_region_id_list", sup_region_id_list)

            aggregation_dict[n_regions - i - 1] = regions_dict.copy()

        return aggregation_dict

    if agg_mode == 'kmeans':   
        # The input observations of kmeans must be "whitened"
        centroids_whitened = vq.whiten(centroids)

        aggregation_dict = {}
        aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}
        
        rss = [] # RSS (distortion) for different k values - in vq.kmeans: average euclidean distance

        for k in range(1,n_regions):
            # Perform k-means on the original centroids to obtained k centroids of aggregated regions
            aggregation_centroids, distortion = vq.kmeans(centroids_whitened, k)
            
            rss.append(distortion)
            regions_label_list = vq.vq(centroids_whitened, aggregation_centroids)[0]
            
            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(k):
                # Group the regions of this regions label
                sup_region = regions_list[regions_label_list == label]
                sup_region_id = '_'.join(sup_region)
                regions_dict[sup_region_id] = sup_region.copy()

            if verbose:
                print(i)
                print('\t', 'lables:', regions_label_list)
                for sup_region_id, sup_region_list in regions_dict.items():
                    print('\t', sup_region_id, ': ', sup_region_list)

            aggregation_dict[k] = regions_dict.copy()

        # Plotting the rss according to increase of k values, check if there exists an inflection point
        fig, ax = spu.plt.subplots(figsize=(25, 12))
        ax.plot(range(1,n_regions),rss,'go-')
        ax.set_title('Impact of k on distortion')
        ax.set_xlabel('K (number_of_regions)')
        ax.set_ylabel('Distortion')
        plt.show()

        path = '/home/s-xing/code/spagat/output/ClusteringAnalysis/'
        figname = save_fig if save_fig is not None else 'Distance_based_scipy_kmeans_Distortion.png'

        spu.plt_savefig(fig=fig, path=path, save_name=figname)

        return aggregation_dict

    # The selection of initialization is also available in sklearn.KMeans!
    '''
    if mode == 'kmeans2':
        regions_list = sds.xr_dataset[dimension_description].values
        centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000
        n_regions = len(centroids)

        aggregation_dict = {}
        aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}
        
        for k in range(1,n_regions):
            # minit can be changed to other initialization method!
            regions_label_list = vq.kmeans2(centroids,k, minit='points')[1]

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(k):
                # Group the regions of this regions label
                sup_region = regions_list[regions_label_list == label]
                sup_region_id = '_'.join(sup_region)
                regions_dict[sup_region_id] = sup_region.copy()

            aggregation_dict[k] = regions_dict.copy()
        
        return aggregation_dict
    '''


    '''Clustering methods via Scikit Learn module'''

    if agg_mode == 'kmeans2':

        aggregation_dict = {}
        aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

        rss = [] # RSS (distortion) for different k values, in sklearn: inertia / within-cluster sum-of-squares

        for k in range(1,n_regions):

            # Compute K-Means clustering: configurations can be modified, e.g. init
            kmeans = skc.KMeans(n_clusters=k).fit(centroids)
            regions_label_list = kmeans.predict(centroids)
            rss.append(kmeans.inertia_)

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(k):
                # Group the regions of this regions label
                sup_region = regions_list[regions_label_list == label]
                sup_region_id = '_'.join(sup_region)
                regions_dict[sup_region_id] = sup_region.copy()

            if verbose:
                print(i)
                print('\t', 'lables:', regions_label_list)
                for sup_region_id, sup_region_list in regions_dict.items():
                    print('\t', sup_region_id, ': ', sup_region_list)

            aggregation_dict[k] = regions_dict.copy()

        # Plotting the rss according to increase of k values, check if there exists an inflection point
        fig, ax = spu.plt.subplots(figsize=(25, 12))
        ax.plot(range(1,n_regions),rss,'go-')
        ax.set_title('Within-cluster sum-of-squares')
        ax.set_xlabel('K (number_of_regions)')
        ax.set_ylabel('Distortion / Inertia')
        plt.show()

        path = '/home/s-xing/code/spagat/output/ClusteringAnalysis/'
        figname = save_fig if save_fig is not None else 'sklearn_kmeans_Distortion.png'

        spu.plt_savefig(fig=fig, path=path, save_name=figname)

        return aggregation_dict
        
    if agg_mode == 'hierarchical2':

        aggregation_dict = {}
        aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

        for i in range(1,n_regions):

            # Computing hierarchical clustering
            model = skc.AgglomerativeClustering(n_clusters=i).fit(centroids)
            regions_label_list = model.labels_

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sup_region_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sup_region_list)
                regions_dict[sup_region_id] = sup_region_list.copy()
            
            if verbose:
                print(i)
                print('\t', 'lables:', regions_label_list)
                for sup_region_id, sup_region_list in regions_dict.items():
                    print('\t', sup_region_id, ': ', sup_region_list)

            aggregation_dict[i] = regions_dict.copy()

        # Create linkage matrix for dendrogram
        def createLinkages():
            clustering_tree = skc.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(centroids)
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
            
            return linkage_matrix

        # Plot the hierarchical tree dendrogram
        linkage_matrix = createLinkages()
        distance_matrix = hierarchy.distance.pdist(centroids)
        #hierarchy.dendrogram(linkage_matrix)

        # Evaluation of this clustering methods
        print('Statistics on this hiearchical clustering:')
        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(linkage_matrix, distance_matrix)[0])

        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(linkage_matrix)
        ax.plot(range(1,len(linkage_matrix)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency Coefficients: indicate where to cut the hierarchy', fontsize=14)
        ax.set_xlabel('Linkage height', fontsize=12)
        ax.set_ylabel('Inconsistencies', fontsize=12)

        plt.xticks(np.arange(1, len(linkage_matrix)+1, 1))
        plt.show()

        # If and how to save the hierarchical tree 
        if ax_illustration is not None:
            R = hierarchy.dendrogram(linkage_matrix, orientation="top",
                                     labels=sds.xr_dataset[dimension_description].values, ax=ax_illustration, leaf_font_size=14)

            if save_fig is not None:

                spu.plt_savefig(save_name=save_fig)
        elif save_fig is not None:

            fig, ax = spu.plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(linkage_matrix, orientation="top",
                                     labels=sds.xr_dataset[dimension_description].values, ax=ax, leaf_font_size=14)

            spu.plt_savefig(fig=fig, save_name=save_fig)

        return aggregation_dict

    if agg_mode == 'spectral':
        aggregation_dict = {}
        aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

        for i in range(1,n_regions):
            # Computing spectral clustering
            model = skc.SpectralClustering(n_clusters=i).fit(centroids)
            regions_label_list = model.labels_

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sup_region_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sup_region_list)
                regions_dict[sup_region_id] = sup_region_list.copy()

            if verbose:
                print(i)
                print('\t', 'lables:', regions_label_list)
                for sup_region_id, sup_region_list in regions_dict.items():
                    print('\t', sup_region_id, ': ', sup_region_list)

            aggregation_dict[i] = regions_dict.copy()
        
        return aggregation_dict




@spu.timer
def all_variable_based_clustering(sds,agg_mode,verbose=False, ax_illustration=None, save_fig=None, dimension_description='space',weighting=None):
    
    '''Original region list'''
    regions_list = sds.xr_dataset['space'].values
    n_regions = len(regions_list)

    aggregation_dict = {}
    aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}

    ''' 1. Using hierarchical clustering for all variables with custom defined distance
        - precomputed distance matrix using selfDistanceMatrix() function
        - linkage method: average
        - hierarchy clustering method of SciPy having spatial contiguity problem
        - hierarchy clustering method of Scikit learn solve spatial contiguity with additional connectivity matrix.
    '''
    ## Clustering methods via SciPy.cluster module
    if agg_mode == 'hierarchical':

        # Obtain the data dictionaries for three var categories after preprocessing
        dict_ts, dict_1d, dict_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')

        # Apply clustering methods based on the Custom Distance Function
        squared_dist_matrix = gu.selfDistanceMatrix(dict_ts, dict_1d, dict_2d, n_regions)
        distance_matrix = hierarchy.distance.squareform(squared_dist_matrix)
        Z = hierarchy.linkage(distance_matrix, method='average')

        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(Z, distance_matrix)[0])
       
        fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(Z)
        ax.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')
        ax.set_title('Inconsistency of each Link with the Links Below', fontsize=14)
        ax.set_xlabel('Number of disjoint clusters under this link', fontsize=12)
        ax.set_ylabel('Inconsistency Coefficients', fontsize=12)

        plt.xticks(range(1,len(Z)+1), np.arange(len(Z)+1,1, -1))
        plt.show()

        #print(list(inconsistency[:,3]))
        
        # If and how to save the hierarchical tree 
        if ax_illustration is not None:
            R = hierarchy.dendrogram(Z, orientation="top",
                                     labels=sds.xr_dataset[dimension_description].values, ax=ax_illustration, leaf_font_size=14)

            if save_fig is not None:

                spu.plt_savefig(save_name=save_fig)
        elif save_fig is not None:

            fig, ax = spu.plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(Z, orientation="top",
                                     labels=sds.xr_dataset[dimension_description].values, ax=ax, leaf_font_size=14)

            spu.plt_savefig(fig=fig, save_name=save_fig)
        
        # regions_dict to record the newest region set after each merging step, regions_dict_complete for all regions appearing during clustering
        regions_dict = {region_id: [region_id] for region_id in regions_list}
        regions_dict_complete = regions_dict.copy()

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
    
        # Silhouette Coefficient scores
        silhouette_scores = gu.computeSilhouetteCoefficient(list(regions_list), squared_dist_matrix, aggregation_dict)
        print(silhouette_scores)

    ## Clustering methods via Scikit Learn module'''
    if agg_mode == 'hierarchical2':

        # Obtain the data dictionaries for three var categories after preprocessing
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')

        # Precompute the distance matrix according to the Custom Distance Function
        squared_distMatrix = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)

        # Connectivity matrix for neighboring structure
        connectMatrix = gu.generateConnectivityMatrix(sds)

        # Silhouette Coefficient scores
        silhouette_scores = []

        for i in range(1,n_regions):
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
                sup_region_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sup_region_list)
                regions_dict[sup_region_id] = sup_region_list.copy()

            aggregation_dict[i] = regions_dict.copy()

        # Plot the hierarchical tree dendrogram
        clustering_tree = skc.AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average',connectivity=connectMatrix).fit(squared_distMatrix)
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
        
        #fig, ax = plt.subplots(figsize=(18,7))
        inconsistency = hierarchy.inconsistent(linkage_matrix)
        print('Inconsistencies:',list(inconsistency[:,3]))
        # ax.plot(range(1,len(linkage_matrix)+1),list(inconsistency[:,3]),'go-')
        # ax.set_title('Inconsistency of each Link with the Links Below', fontsize=14)
        # ax.set_xlabel('Number of disjoint clusters under this link', fontsize=12)
        # ax.set_ylabel('Inconsistencies', fontsize=12)

        # plt.xticks(range(1,len(linkage_matrix)+1), np.arange(len(linkage_matrix)+1,1, -1))
        # plt.show()

        print('Silhouette scores: ',silhouette_scores)

        

    ''' 2. Using spectral clustering with precomputed affinity matrix
        - precomputed affinity matrix by conversation from distance matrices to similarity matrix using RBF kernel
        - also having spatial contiguity problem due to the created complete graph
        - solve it by considering the additional connectivity matrix to cut some edges
    '''
    ## Affinity matrix: combine three matrices of ts_vars, 1d_vars and 2d_vars to one single precomputed affinity matrix
    if agg_mode == 'spectral':
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

        # Obtain the matrices for three var categories after preprocessing
        feature_matrix_ts, feature_matrix_1d, adjacency_matrix_2d = gu.preprocessDataset(sds, handle_mode='toAffinity')

        # List of weighting factors for 3 categories
        if weighting:
            weighting = weighting
        else:
            weighting = [1,1,1]

        # Using RBF kernel to construct affinity matrix
        delta = 1

        ##### Obtain affinity matrix for TimeSeries part via RBF kernel applied on distance matrix
        distance_matrix_ts = hierarchy.distance.squareform(hierarchy.distance.pdist(feature_matrix_ts))

        affinity_ts = np.exp(- distance_matrix_ts ** 2 / (2. * delta ** 2))

        ##### Obtain affinity matrix for 1d-Vars part via RBF kernel applied on distance matrix
        distance_matrix_1d = hierarchy.distance.squareform(hierarchy.distance.pdist(feature_matrix_1d))

        affinity_1d = np.exp(- distance_matrix_1d ** 2 / (2. * delta ** 2))

        ##### Obtain affinity matrix for 2d-Vars

        # Convert the adjacency matrix to a dissimilarity matrix similar to a distance matrix: high value=more dissimilarity, 0=identical elements
        adjacency_2d_adverse = 1.0 / adjacency_matrix_2d
        max_value = adjacency_2d_adverse[np.isfinite(adjacency_2d_adverse)].max()
        adjacency_2d_adverse[np.isinf(adjacency_2d_adverse)] = max_value + 10
        np.fill_diagonal(adjacency_2d_adverse,0)

        # Construct the affinity matrix by applying RBF on the dissimilarity matrix
        affinity_2d = np.exp(- adjacency_2d_adverse ** 2 / (2. * delta ** 2))

        ##### The precomputed affinity matrix for spectral clustering
        affinity_matrix = (affinity_ts * weighting[0] + affinity_1d * weighting[1] + affinity_2d * weighting[2]) 

        # ##### Solve the spatial contiguity problem with the connectivity condition
        # # Connectivity matrix for neighboring structure
        # connectMatrix = gu.generateConnectivityMatrix(sds)
        # # Cut down the edges that have zero value in connectivity matrix
        # affinity_matrix[connectMatrix==0] = 0

        # Evaluation indicators
        modularities = []

        for i in range(1,n_regions):
            # Perform the spectral clustering with the precomputed affinity matrix (adjacency matrix)
            model = skc.SpectralClustering(n_clusters=i,affinity='precomputed').fit(affinity_matrix)
            regions_label_list = model.labels_

            # Compute the modularity for evaluation, using affinity matrix as adjacency matrix of a graph
            modularity = gu.computeModularity(affinity_matrix, regions_label_list)
            modularities.append(modularity)

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sup_region_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sup_region_list)
                regions_dict[sup_region_id] = sup_region_list.copy()

            aggregation_dict[i] = regions_dict.copy()
    
        # Plotting the modularites according to increase of k values, check if there exists an inflection point
        # fig, ax = spu.plt.subplots(figsize=(25, 12))
        # ax.plot(range(1,n_regions),modularities,'go-')
        # ax.set_title('Impact of aggregated regions on modularity')
        # ax.set_xlabel('number of aggregated regions')
        # ax.set_ylabel('Modularity')
        # plt.show()

        print('Modularities',modularities)

        # Silhouette Coefficient scores
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')
        distances = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)
        silhouette_scores = gu.computeSilhouetteCoefficient(list(regions_list), distances, aggregation_dict)
        print('Silhouette scores: ',silhouette_scores)

    ## Affinity matrix: construct a distance matrix based on selfDistanceMatrix function, transform it to similarity matrix
    if agg_mode =='spectral2':

        # Obtain the data dictionaries for three var categories after preprocessing
        ds_ts, ds_1d, ds_2d = gu.preprocessDataset(sds, handle_mode='toDissimilarity')

        # Precompute the distance matrix according to the Custom Distance Function
        distMatrix = gu.selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions)
        # Rescaling the matrix in order to generate valid affinity_matrix
        distMatrix = gu.matrix_MinMaxScaler(distMatrix)

        # Obtain affinity matrix for part_1 via RBF kernel applied on distance matrix
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

        for i in range(1,n_regions):
            # Perform the spectral clustering with the precomputed affinity matrix (adjacency matrix)
            model = skc.SpectralClustering(n_clusters=i,affinity='precomputed').fit(affinity_matrix)
            regions_label_list = model.labels_

            # Compute the modularity for evaluation, using affinity matrix as adjacency matrix of a graph
            modularity = gu.computeModularity(affinity_matrix, regions_label_list)
            modularities.append(modularity)

            # Silhouette Coefficient score for this clustering results
            if i != 1:
                s = metrics.silhouette_score(distMatrix, regions_label_list, metric='precomputed')
                silhouette_scores.append(s)

            # Create a regions dictionary for the aggregated regions
            regions_dict = {}
            for label in range(i):
                # Group the regions of this regions label
                sup_region_list = list(regions_list[regions_label_list == label])
                sup_region_id = '_'.join(sup_region_list)
                regions_dict[sup_region_id] = sup_region_list.copy()

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



@spu.timer
def variables_and_distance_ensemble_clustering(sds,agg_mode='hierarchical',verbose=False, ax_illustration=None, save_fig=None, dimension_description='space',weighting=None):
    '''Ensemble clustering with basic clusterings from:
        - geographical centroids
        - part_1 of dataset (time seiries data + 1d variables) by various clustering methods
        - part_2 of dataset (2d variables) as (un)directed graph
    '''
    return None

