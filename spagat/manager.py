import logging

import metis_utils.io_tools as ito
import spagat.dataset as spd
import spagat.grouping as spg
import spagat.representation as spr

logger_manager = logging.getLogger('spagat_manager')


class SpagatManager:
    """Spagat Manager manages the spatial aggregation analysis toolchain."""

    def __init__(self):
        self.sds = spd.SpagatDataSet()
        self.aggregation_dict = None
        self.sds_out = spd.SpagatDataSet()
        self.analysis_path = None
        self.aggregation_function_dict = None

    def read_data(self, sds_folder_path):

        self.sds.read_dataset(sds_folder_path=sds_folder_path)
        spr.add_region_centroids(self.sds)

    def grouping(self, mode='all', dimension_description='space'):
        
        aggregation_mode = 'spectral2'

        # Using distanced_based_clustering (geographical distance)
        if mode == 'distance based':
            if self.analysis_path is not None:
                save_path = self.analysis_path / 'cluster_dendrogram'
            else:
                save_path = None

            self.aggregation_dict = spg.distance_based_clustering(self.sds, agg_mode=aggregation_mode, save_fig=save_path, dimension_description=dimension_description)

        # Using clustering methods based on all variables
        if mode == 'all':
            if self.analysis_path is not None:
                save_path = self.analysis_path / 'cluster_dendrogram'
            else:
                save_path = None

            self.aggregation_dict = spg.all_variable_based_clustering(self.sds, agg_mode=aggregation_mode, save_fig=save_path,dimension_description=dimension_description)


    def representation(self, number_of_regions):
        self.sds_out = spr.aggregate_based_on_sub_to_sup_region_id_dict(self.sds,
                                                                        self.aggregation_dict[number_of_regions], 
                                                                        aggregation_function_dict=self.aggregation_function_dict)

    def save_data(self, sds_folder, save_initial_sds=True):

        spr.create_grid_shapefile(self.sds_out, filename=sds_folder / 'sds_AC_lines.shp')

        self.sds_out.save_sds(sds_folder)

        if save_initial_sds:
            spr.create_grid_shapefile(self.sds, filename=sds_folder / 'initial_sds_AC_lines.shp')
            self.sds.save_sds(sds_folder,
                              sds_region_filename=f'initial_sds_regions.shp',
                              sds_xr_dataset_filename=f'initial_sds_xr_dataset.nc4')
