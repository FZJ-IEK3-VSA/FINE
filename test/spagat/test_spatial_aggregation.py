
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
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')


@pytest.mark.parametrize("agg_mode", ['sklearn_kmeans', 'sklearn_spectral', 'scipy_hierarchical'])
@pytest.mark.parametrize("n_regions", [1, 3, 8]) 
def test_spatial_aggregation_distance_based(multi_node_test_esM_init, agg_mode, n_regions):   
    
    n_regions = 3
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData_Germany/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = "distance_based", 
                                                                nRegionsForRepresentation = n_regions, 
                                                                aggregatedResultsPath=None,
                                                                agg_mode=agg_mode)   

    #ASSERTION 
    assert len(aggregated_esM.locations) == n_regions
    # Additional check - if the optimization runs through
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')
    #TODO: add test to check if optimization summary is available


@pytest.mark.parametrize("aggregation_function_dict", [None, {'operationRateMax': ('weighted mean', 'capacityMax'), 
                                                            'operationRateFix': ('sum', None), 
                                                            'capacityMax': ('sum', None), 
                                                            'capacityFix': ('sum', None), 
                                                            'locationalEligibility': ('bool', None)} ]) 
@pytest.mark.parametrize("n_regions", [1, 3, 8])                                                            
def test_spatial_aggregation_parameter_based(multi_node_test_esM_init, 
                                            aggregation_function_dict,
                                            n_regions):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData_Germany/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = 'parameter_based', 
                                                                nRegionsForRepresentation = n_regions, 
                                                                aggregatedResultsPath=None,
                                                                aggregation_function_dict=aggregation_function_dict,
                                                                var_weights={'1d_vars' : 10})   

    #ASSERTION 
    assert len(aggregated_esM.locations) == n_regions
    # Additional check - if the optimization runs through
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')
    #TODO: add test to check if optimization summary is available 