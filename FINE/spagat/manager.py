# #TODO: delete this who file 

# """Module that manages the other submodules.

# """

# import logging
# import os

# import FINE.spagat.dataset as spd
# import FINE.spagat.grouping as spg
# import FINE.spagat.representation as spr

# logger_manager = logging.getLogger("spagat_manager")


# class SpagatManager:
#     """
#     High-level access point to manage spatial aggregations.
    

#     Examples
#     --------


#     """

#     def __init__(self):
#         self.sds = spd.SpagatDataset()
#         self.aggregation_dict = None
#         self.sds_out = spd.SpagatDataset()
#         self.analysis_path = None
#         self.aggregation_function_dict = None

#     def read_data(self, sds_folder_path):

#         self.sds.read_dataset(sds_folder_path=sds_folder_path)
#         spr.add_region_centroids(self.sds)

#     def grouping(self, mode='all', dimension_description='space'):  #TODO: make changes here based on notes in xarray_io.spatial_aggregation()
#         #FIXME: why isnt string_based_clustering not an option here??

#         aggregation_mode = 'spectral2'   #FIXME: should this be hardcoded ??

#         # Using distanced_based_clustering (geographical distance)
#         if mode == 'distance based':
#             if self.analysis_path is not None:
#                 save_path = os.path.join( self.analysis_path, "cluster_dendrogram")
#             else:
#                 save_path = None

#             self.aggregation_dict = spg.distance_based_clustering(self.sds, agg_mode=aggregation_mode, save_fig=save_path, dimension_description=dimension_description)
#            #TODO: check how save_path variable is passed here (maybe using **kwargs (refer to xarray_io.spatial_aggregation())) 
#         # Using clustering methods based on all variables
#         if mode == 'all':                         #TODO: maybe change this to all_variable_based_clustering
#             if self.analysis_path is not None:
#                 save_path = os.path.join( self.analysis_path, "cluster_dendrogram")
#             else:
#                 save_path = None

#             self.aggregation_dict = spg.all_variable_based_clustering(self.sds, agg_mode=aggregation_mode, save_fig=save_path,dimension_description=dimension_description)


#     def representation(self, number_of_regions):         
#         self.sds_out = spr.aggregate_based_on_sub_to_sup_region_id_dict(      
#             self.sds,
#             self.aggregation_dict[number_of_regions],         #NOTE: sub_to_sup_region_id_dict is provided by aggregation_dict from grouping function and since it groups 
#                                                               #for all number of regions (from 8 to 7,6,5,4,3,2,1 for example) number of regions for which representation 
#                                                               #should be run must be specified here. 
#             aggregation_function_dict=self.aggregation_function_dict,
#         )

#     def save_data(
#         self,
#         sds_folder,
#         save_initial_sds=True,
#         eligibility_variable="2d_locationalEligibility",
#         eligibility_component="Transmission, AC cables",
#     ):

#         spr.create_grid_shapefile(
#             self.sds_out,
#             filename= os.path.join(sds_folder , "sds_grid_shape.shp"),
#             eligibility_variable=eligibility_variable,
#             eligibility_component=eligibility_component,
#         )

#         self.sds_out.save_sds(sds_folder)