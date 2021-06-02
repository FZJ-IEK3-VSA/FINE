
import os
import pytest

import xarray as xr
import numpy as np 

import FINE as fn

@pytest.mark.parametrize("grouping_mode", ['string_based', 'distance_based', 'parameter_based'])
def test_esm_to_xr_and_back_during_spatial_aggregation(grouping_mode, multi_node_test_esM_init):
    """Resulting number of regions would be the same as the original number. No aggregation 
    actually takes place. Tests if the esm instance, created after spatial aggregation
    is run, has all the info originally present. 
    """
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData/ShapeFiles/clusteredRegions.shp')

    #FUNCTION CALL 
    aggregated_esM = multi_node_test_esM_init.aggregateSpatially(shapefilePath = SHAPEFILE_PATH, 
                                                                grouping_mode = grouping_mode, 
                                                                nRegionsForRepresentation = 8)   
    
    #ASSERTION 
    if grouping_mode == 'string_based':
        expected_locations = [loc[-1:] for loc in multi_node_test_esM_init.locations]
        assert sorted(aggregated_esM.locations) == sorted(expected_locations)
    else:
        assert sorted(aggregated_esM.locations) == sorted(multi_node_test_esM_init.locations)

    expected_ts = multi_node_test_esM_init.getComponentAttribute('Hydrogen demand', 'operationRateFix').values
    output_ts = aggregated_esM.getComponentAttribute('Hydrogen demand', 'operationRateFix').values
    assert np.array_equal(expected_ts, output_ts)

    expected_2d = multi_node_test_esM_init.getComponentAttribute('DC cables', 'locationalEligibility').values
    output_2d = aggregated_esM.getComponentAttribute('DC cables', 'locationalEligibility').values 
    assert np.array_equal(output_2d, expected_2d)

    expected_1d = multi_node_test_esM_init.getComponentAttribute('Pumped hydro storage', 'capacityFix').values 
    output_1d = aggregated_esM.getComponentAttribute('Pumped hydro storage', 'capacityFix').values 
    assert np.array_equal(output_1d, expected_1d)

    expected_0d = multi_node_test_esM_init.getComponentAttribute('Electroylzers', 'investPerCapacity').values 
    output_0d = aggregated_esM.getComponentAttribute('Electroylzers', 'investPerCapacity').values 
    assert np.array_equal(output_0d, expected_0d)

    expected_0d_bool = multi_node_test_esM_init.getComponentAttribute('CO2 from enviroment', 'hasCapacityVariable')
    output_0d_bool = aggregated_esM.getComponentAttribute('CO2 from enviroment', 'hasCapacityVariable')
    assert output_0d_bool  == expected_0d_bool 

def test_spatial_aggregation_string_based(multi_node_test_esM_init):   #TODO: run test for dummy data where some regions actually merge!
    
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
    aggregated_esM.aggregateTemporally(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver='glpk')


@pytest.mark.parametrize("agg_mode", ['sklearn_kmeans', 'sklearn_spectral', 'scipy_hierarchical'])
@pytest.mark.parametrize("n_regions", [2, 3]) #TODO: test for 1 region 
def test_spatial_aggregation_distance_based(multi_node_test_esM_init, agg_mode, n_regions):   

    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData/ShapeFiles/clusteredRegions.shp')

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
@pytest.mark.parametrize("n_regions", [3, 6])         #TODO: test for 1 region                                                     
def test_spatial_aggregation_parameter_based(multi_node_test_esM_init, 
                                            aggregation_function_dict,
                                            n_regions):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData/ShapeFiles/clusteredRegions.shp')

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