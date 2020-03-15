import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster import vq

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

    if mode == 'hierarchical':

        centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000  # km

        Z = hierarchy.linkage(centroids, 'centroid')

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

        n_regions = len(Z)

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
        regions_list = sds.xr_dataset[dimension_description].values
        centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000
        centroids_whitened = vq.whiten(centroids)
        n_regions = len(centroids)

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
        plt.plot(range(1,n_regions),rss,'g^-')
        plt.title('Impact of k on distortion')
        plt.xlabel('K (number_of_regions)')
        plt.ylabel('Distortion')
        plt.show()

        return aggregation_dict

    if mode == 'kmeans2':
        regions_list = sds.xr_dataset[dimension_description].values
        centroids = np.asarray([[point.item().x, point.item().y] for point in sds.xr_dataset.gpd_centroids])/1000
        n_regions = len(centroids)

        aggregation_dict = {}
        aggregation_dict[n_regions] = {region_id: [region_id] for region_id in regions_list}
        
        for k in range(1,n_regions):
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
