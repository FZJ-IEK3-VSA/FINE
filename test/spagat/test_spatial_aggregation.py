
import os
import pytest

import xarray as xr

import FINE as fn

def test_spatial_aggregation_string_based(multi_node_test_esM_init):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData_Germany/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = 'string_based',                              
                                                                aggregatedResultsPath=None)   

    #ASSERTION 
    assert len(aggregated_esM.locations) == 8
    # Additional check - if the optimization runs through
    aggregated_esM.cluster(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')


@pytest.mark.parametrize("grouping_mode, agg_mode", [('distance_based', 'sklearn_kmeans'), 
                                                       ('distance_based', 'sklearn_spectral'),
                                                       ('distance_based', 'scipy_hierarchical'),
                                                       ('all_variable_based', 'scipy_hierarchical'),
                                                       ('all_variable_based', 'spectral_with_precomputedAffinity'),
                                                       ('all_variable_based', 'spectral_with_RBFaffinity')
                                                    ])
@pytest.mark.parametrize("n_regions", [1, 3, 8]) 
def test_spatial_aggregation(multi_node_test_esM_init,
                            grouping_mode, 
                            agg_mode,
                            n_regions):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData_Germany/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = grouping_mode, 
                                                                nRegionsForRepresentation = n_regions, 
                                                                aggregatedResultsPath=None,
                                                                agg_mode=agg_mode)   

    #ASSERTION 
    assert len(aggregated_esM.locations) == n_regions
    # Additional check - if the optimization runs through
    aggregated_esM.cluster(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')
    #TODO: add test to check if optimization summary is available

@pytest.mark.parametrize("aggregation_function_dict", [None, {'operationRateMax': ('weighted mean', 'capacityMax'), 
                                                            'operationRateFix': ('sum', None), 
                                                            'capacityMax': ('sum', None), 
                                                            'capacityFix': ('sum', None), 
                                                            'locationalEligibility': ('bool', None)} ]) 
def test_spatial_aggregation_with_aggregation_function_dict(multi_node_test_esM_init, 
                                                        aggregation_function_dict):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData_Germany/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = 'all_variable_based', 
                                                                nRegionsForRepresentation = 2, 
                                                                aggregatedResultsPath=None,
                                                                agg_mode='spectral_with_RBFaffinity',
                                                                aggregation_function_dict=aggregation_function_dict)   

    #ASSERTION 
    assert len(aggregated_esM.locations) == 2
    # Additional check - if the optimization runs through
    aggregated_esM.cluster(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')
    #TODO: add test to check if optimization summary is available 