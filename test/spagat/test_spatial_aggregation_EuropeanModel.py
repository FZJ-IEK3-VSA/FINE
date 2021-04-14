# import os
# import sys
# import pytest

# import xarray as xr

# import FINE as fn

# #NOTE: test requires MATES. This will change in the future
# try:
#     from MATES import OptimizationManager
# except ImportError:
#     pass

# @pytest.mark.skipif('MATES' not in sys.modules, reason="requires MATES")
# def test_spatial_aggregation_EuropeanModel():
#     INPUT_PATH = os.path.join(os.path.dirname(__file__), \
#         '../../examples/Multi-regional_Energy_System_Workflow/', 
#             'InputData/EuropeanModelData')

#     aggregation_function_dict = {'operationRateMax': ('mean', None),
#                               'operationRateFix': ('sum', None),
#                               'locationalEligibility': ('bool', None),
#                               'capacityMax': ('sum', None),
#                               'investPerCapacity': ('sum', None),
#                               'investIfBuilt': ('sum', None),
#                               'opexPerOperation': ('sum', None),
#                               'opexPerCapacity': ('sum', None),
#                               'opexIfBuilt': ('sum', None),
#                               'interestRate': ('mean', None),
#                               'economicLifetime': ('mean', None),
#                               'capacityFix': ('sum', None),
#                               'losses': ('mean', None),
#                               'distances': ('mean', None),
#                               'commodityCost': ('mean', None),
#                               'commodityRevenue': ('mean', None),
#                               'opexPerChargeOperation': ('mean', None),
#                               'opexPerDischargeOperation': ('mean', None),
#                               'QPcostScale': ('sum', None), 
#                               'technicalLifetime': ('sum', None),
#                               'reactances': ('sum', None), 
#                               }

#     n_regions = 6

#     input_data = os.path.join(INPUT_PATH,'new_EuropeanScenario_Ch5_Section1_DGC.nc')
#     input_shape_path = os.path.join(INPUT_PATH,'supregions.shp')

#     opM = OptimizationManager(input_data, commodityUnitsDict={'electricity': 'GW_el', 
#                                                             'hydrogen': 'GW_h2', 
#                                                             'water': 'GW_wt', 
#                                                             'waterRes':'GW_wt', 
#                                                             'biomass':'GW_bio'})
#     esM_initial = opM.setup()

#     aggregated_esM = esM_initial.aggregateSpatially(shapefilePath = input_shape_path, 
#                                                     grouping_mode='all_variable_based', 
#                                                     nRegionsForRepresentation = n_regions,
#                                                                                                #aggregatedResultsPath = OUTPUT_PATH,
#                                                     agg_mode='sklearn_hierarchical',
#                                                                                                #spatial_contiguity = True, 
#                                                     aggregation_function_dict=aggregation_function_dict)
#                                                                                                 # sds_region_filename=groupingResult_shp_file,
#                                                                                                 # sds_xr_dataset_filename=groupingResult_xr_file) 
    