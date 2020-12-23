import os
import pytest

import numpy as np
import xarray as xr
import geopandas as gpd

import FINE as fn

from FINE.IOManagement import dictIO
import FINE.IOManagement.xarray_io as xrio 

import FINE.spagat.dataset as spd
import FINE.spagat.grouping as spg
import FINE.spagat.representation as spr 

def test_generate_iteration_dicts(minimal_test_esM):
    esm_dict, component_dict = dictIO.exportToDict(minimal_test_esM)
    output_df_dict, output_series_dict = xrio.generate_iteration_dicts(esm_dict, component_dict)

    assert output_series_dict.get('locationalEligibility') == [('Source', 'Electricity market'), 
                                                                ('Sink', 'Industry site'), 
                                                                ('Conversion', 'Electrolyzers'),
                                                                ('Storage', 'Pressure tank'),
                                                                ('Transmission', 'Pipelines')]

    assert output_series_dict.get('commodityRevenue') == [('Source', 'Electricity market'), 
                                                            ('Sink', 'Industry site')]

    assert output_series_dict.get('opexPerChargeOperation') == [('Storage', 'Pressure tank')]

    assert output_df_dict == {'operationRateMax': [('Source', 'Electricity market')],
                            'commodityCostTimeSeries': [('Source', 'Electricity market')],
                            'commodityRevenueTimeSeries': [('Source', 'Electricity market')],
                            'operationRateFix': [('Sink', 'Industry site')]} 

def test_dimensional_data_to_xarray_dataset(minimal_test_esM):
    esm_dict, component_dict = dictIO.exportToDict(minimal_test_esM)

    expected_locations = list(minimal_test_esM.locations).sort()
    expected_Electricitymarket_operationRateMax = \
        minimal_test_esM.getComponentAttribute('Electricity market', 'operationRateMax').values
    expected_Electrolyzers_investPerCapacity = \
        minimal_test_esM.getComponentAttribute('Electrolyzers', 'investPerCapacity').values
    
    #FUNCTION CALL
    output_xarray = xrio.dimensional_data_to_xarray_dataset(esm_dict, component_dict)
    ## a time series variable
    output_locations = list(output_xarray.space.values).sort()
    output_Electricitymarket_operationRateMax = \
                    output_xarray.operationRateMax.loc['Source, Electricity market', 0, :, :].values
    ## a 1d variable
    output_Electrolyzers_investPerCapacity = \
        output_xarray['1d_investPerCapacity'].loc['Conversion, Electrolyzers', :].values

    ## a 2d variable
    output_Pipelines_investPerCapacity = \
        output_xarray['2d_investPerCapacity'].loc['Transmission, Pipelines', :, :].values

    #ASSERTION
    assert output_locations == expected_locations
    assert np.array_equal(output_Electricitymarket_operationRateMax, expected_Electricitymarket_operationRateMax)
    assert np.array_equal(output_Electrolyzers_investPerCapacity, expected_Electrolyzers_investPerCapacity)
    assert (output_Pipelines_investPerCapacity[0][0] == 0.0) and \
                      (output_Pipelines_investPerCapacity[0][1] == 0.0885)

    
def test_update_dicts_based_on_xarray_dataset(multi_node_test_esM_init):
    # TEST DATA
    #NOTE: aggregated_xr_dataset is obtained by running spatial_aggregation(),
    # distance_based clustering with all default settings.
    # Representation is performed for nRegionsForRepresentation = 2 
    TEST_FILE_PATH = os.path.join(os.path.dirname(__file__), '../../test/spagat/data/input/aggregated_sds_xr_dataset.nc4')
    aggregated_xr_dataset = xr.open_dataset(TEST_FILE_PATH)

    esm_dict, comp_dict = dictIO.exportToDict(multi_node_test_esM_init)
    
    #EXPECTED 
    expected_locations = list(aggregated_xr_dataset.space.values).sort()
    expected_opexPerCapacity_Windoffshore = \
        sorted(aggregated_xr_dataset['1d_opexPerCapacity'].loc['Source, Wind (offshore)', :].values)

    #FUNCTION CALL 
    output_esm_dict, output_comp_dict = \
        xrio.update_dicts_based_on_xarray_dataset(esm_dict, 
                                                comp_dict, 
                                                xarray_dataset=aggregated_xr_dataset)
    
    output_locations = list(output_esm_dict.get('locations')).sort()
    output_opexPerCapacity_Windoffshore = \
        sorted(output_comp_dict.get('Source').get('Wind (offshore)').get('opexPerCapacity').values, key=float)
    
    #ASSERTION 
    assert output_locations == expected_locations
    assert output_opexPerCapacity_Windoffshore == expected_opexPerCapacity_Windoffshore
     
def test_spatial_aggregation_string_based(multi_node_test_esM_init, 
                                          solver):   
    
    SHAPEFILE_PATH = os.path.join(os.path.dirname(__file__), \
        '../../examples/Multi-regional_Energy_System_Workflow/', 
            'InputData/SpatialData/ShapeFiles/clusteredRegions.shp')
    regions_gdf = gpd.read_file(SHAPEFILE_PATH)

    #FUNCTION CALL 
    aggregated_esM = xrio.spatial_aggregation(esM = multi_node_test_esM_init, 
                                            gdfRegions = regions_gdf, 
                                            grouping_mode = 'string_based', 
                                            aggregation_function_dict = {'locationalEligibility': ('bool', None)}, #TODO: this is a temp bug fix,                                      
                                            aggregatedResultsPath=None,                       # remove it after the bug is fixed in representation 
                                            agg_mode='sklearn_spectral2')   

    #ASSERTION 
    assert len(aggregated_esM.locations) == 8
    # Additional check - if the optimization runs through
    aggregated_esM.cluster(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver=solver)

    
@pytest.mark.parametrize("grouping_mode, agg_mode", [('distance_based', 'sklearn_kmeans'), 
                                                       ('distance_based', 'sklearn_spectral'),
                                                       ('distance_based', 'scipy_hierarchical'),
                                                       ('all_variable_based', 'scipy_hierarchical'),
                                                       ('all_variable_based', 'sklearn_spectral1'),
                                                       ('all_variable_based', 'sklearn_spectral2')
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
    regions_gdf = gpd.read_file(SHAPEFILE_PATH)

    #FUNCTION CALL 
    aggregated_esM = xrio.spatial_aggregation(esM = multi_node_test_esM_init, 
                                            gdfRegions = regions_gdf, 
                                            grouping_mode = grouping_mode, 
                                            aggregation_function_dict = {'locationalEligibility': ('bool', None)}, #TODO: this is a temp bug fix,                                      
                                            nRegionsForRepresentation = n_regions,                                         # remove it after the bug is fixed in representation 
                                            aggregatedResultsPath=None,
                                            agg_mode=agg_mode)   

    #ASSERTION 
    assert len(aggregated_esM.locations) == n_regions
    # Additional check - if the optimization runs through
    aggregated_esM.cluster(numberOfTypicalPeriods=2)
    aggregated_esM.optimize(timeSeriesAggregation=True, solver=solver)



    

