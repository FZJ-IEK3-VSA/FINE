
import os
import pytest

import xarray as xr

import FINE as fn

def test_spatial_aggregation_string_based(multi_node_test_esM_init, 
                                          solver):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = 'string_based',                              
                                                                aggregatedResultsPath=None)   

    #ASSERTION 
    assert len(aggregated_esM.locations) == 8
    # Additional check - if the optimization runs through
    aggregated_esM.cluster(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver=solver)

    
@pytest.mark.parametrize("grouping_mode, agg_mode", [('distance_based', 'sklearn_kmeans'), 
                                                       ('distance_based', 'sklearn_spectral'),
                                                       ('distance_based', 'scipy_hierarchical'),
                                                       ('all_variable_based', 'scipy_hierarchical'),
                                                       ('all_variable_based', 'spectral_with_precomputedAffinity'),
                                                       ('all_variable_based', 'spectral_with_RBFaffinity')
                                                    ])
@pytest.mark.parametrize("n_regions", [1, 3, 8]) 
def test_spatial_aggregation(multi_node_test_esM_init, 
                            solver, 
                            grouping_mode, 
                            agg_mode,
                            n_regions):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData/ShapeFiles/clusteredRegions.shp')

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
    aggregated_esM.optimize(timeSeriesAggregation=True, solver=solver)
    #TODO: add test to check if optimization summary is available