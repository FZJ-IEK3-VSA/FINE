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

# import spagat.dataset as spd
import metis_utils.io_tools as ito
import metis_utils.plot_tools as pto
import metis_utils.time_tools as tto

# import pypsa
# import pypsa.networkclustering as nc
# from sklearn.cluster import AgglomerativeClustering

logger_grouping = logging.getLogger('spagat_grouping')


def string_based_clustering(regions):
    '''Creates a dictionary containing sup_regions and respective lists of sub_regions'''

    # TODO: this is implemented spefically for the e-id: '01_es' -> generalize this!
    nation_set = set([region_id.split('_')[1] for region_id in regions])

    sub_to_sup_region_id_dict = {}

    for nation in nation_set:
        sub_to_sup_region_id_dict[nation] = [region_id
                                             for region_id in regions
                                             if region_id.split('_')[1] == nation]

    return sub_to_sup_region_id_dict


@tto.timer
def distance_based_clustering(sds, mode, verbose=False, ax_illustration=None, save_fig=None, dimension_description='space'):
    '''Cluster M regions based on centroid distance, hence closest regions are aggregated to obtain N regions.'''
    
    centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000  # km
    regions_list = sds.xr_dataset[dimension_description].values
    n_regions = len(centroids)

    '''Clustering methods via SciPy.cluster module'''

    if mode == 'hierarchical':

        distance_matrix = hierarchy.distance.pdist(centroids)

        # TO-DO: can investigate various methods, e.g. 'average', 'weighted', 'centroid'
        Z = hierarchy.linkage(distance_matrix, 'centroid')

        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(Z, distance_matrix)[0])
        hierarchy.dendrogram(Z)

        plt.figure(2)
        inconsistency = hierarchy.inconsistent(Z)
        plt.plot(range(1,len(Z)+1),list(inconsistency[:,3]),'go-')

        if ax_illustration is not None:
            R = hierarchy.dendrogram(Z, orientation="top",
                                     labels=sds.xr_dataset[dimension_description].values, ax=ax_illustration, leaf_font_size=14)

            if save_fig is not None:

                pto.plt_savefig(save_name=save_fig)

        elif save_fig is not None:

            fig, ax = pto.plt.subplots(figsize=(25, 12))

            R = hierarchy.dendrogram(Z, orientation="top",
                                     labels=sds.xr_dataset[dimension_description].values, ax=ax, leaf_font_size=14)

            pto.plt_savefig(fig=fig, save_name=save_fig)

        #n_regions = len(Z)

        aggregation_dict = {}

        regions_dict = {region_id: [region_id] for region_id in list(sds.xr_dataset[dimension_description].values)}

        regions_dict_complete = {region_id: [region_id] for region_id in list(sds.xr_dataset[dimension_description].values)}

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

            aggregation_dict[n_regions - i] = regions_dict.copy()

        return aggregation_dict

    if mode == 'kmeans':   
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

            aggregation_dict[k] = regions_dict.copy()

        # Plotting the rss according to increase of k values, check if there exists an inflection point
        plt.plot(range(1,n_regions),rss,'go-')
        plt.title('Impact of k on distortion')
        plt.xlabel('K (number_of_regions)')
        plt.ylabel('Distortion')

        plt.savefig('/home/s-xing/code/spagat/output/ClusteringAnalysis/scipy_kmeans_Distortion.png')
        plt.show()


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

    if mode == 'kmeans2':

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

            aggregation_dict[k] = regions_dict.copy()

        # Plotting the rss according to increase of k values, check if there exists an inflection point
        plt.plot(range(1,n_regions),rss,'go-')
        plt.title('Within-cluster sum-of-squares')
        plt.xlabel('K (number_of_regions)')
        plt.ylabel('Inertia')

        plt.savefig('/home/s-xing/code/spagat/output/ClusteringAnalysis/sklearn_kmeans_Distortion.png')
        plt.show()

        return aggregation_dict
        
    if mode == 'hierarchical2':

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
                sup_region = regions_list[regions_label_list == label]
                sup_region_id = '_'.join(sup_region)
                regions_dict[sup_region_id] = sup_region.copy()

            aggregation_dict[i] = regions_dict.copy()

        # Plot the hierarchical tree dendrogram
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
        # Plot the corresponding dendrogram
        hierarchy.dendrogram(linkage_matrix)

        distance_matrix = hierarchy.distance.pdist(centroids)
        print('The cophenetic correlation coefficient of the hiearchical clustering is ', hierarchy.cophenet(linkage_matrix, distance_matrix)[0])

        return aggregation_dict
