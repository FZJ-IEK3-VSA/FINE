import os

import pytest
import numpy as np
import xarray as xr

import FINE.spagat.representation as spr

def test_aggregate_geometries(xr_and_dict_for_basic_representation):

    sub_to_sup_region_id_dict, xr_for_basic_representation = xr_and_dict_for_basic_representation
    
    test_xarray = xr_for_basic_representation['gpd_geometries']

    #FUNCTION CALL 
    output_xarray = spr.aggregate_geometries(test_xarray, sub_to_sup_region_id_dict)   

    #ASSERTION 
    assert list(output_xarray.space.values) == list(sub_to_sup_region_id_dict.keys())

    for joined_polygon in output_xarray.values:  #NOTE: Only length and geom_type of the resutling polygons 
                                                # are tested. Even in shapely source code only geom_type is checked!
                                                # Need to unpermute the resulting coordinates in order to assert 
                                                # against expected coordinates. 
                                                 
        assert len(joined_polygon.exterior.coords) == 7 
        assert (joined_polygon.geom_type in ('MultiPolygon'))

 
@pytest.mark.parametrize("mode, expected", [("mean", 3), ("sum", 6)])
def test_aggregate_time_series_mean_and_sum(xr_and_dict_for_basic_representation, 
                               mode, 
                               expected):

    sub_to_sup_region_id_dict, xr_for_basic_representation = xr_and_dict_for_basic_representation

    test_xarray = xr_for_basic_representation['ts_operationRateMax']
    
    
    #FUNCTION CALL
    time_series_aggregated = spr.aggregate_time_series(test_xarray, 
                                                        sub_to_sup_region_id_dict, 
                                                        mode=mode)
    
    #ASSERTION 
    ## for valid component, source_comp
    assert time_series_aggregated.loc['source_comp', 'T0', '01_reg_02_reg'].values == expected
    
    ## for invalid component, sink_comp
    assert np.isnan(time_series_aggregated.loc['sink_comp', 'T0', '01_reg_02_reg'].values)
    
@pytest.mark.parametrize("data, weight, expected_grp1, expected_grp2", 
                           [ 
                            # all non zero values for valid components 
                            ( np.array([ [ [3, 3, 3, 3] for i in range(2)],

                                  [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ]),

                            np.array([ [3,  3,  3, 3], [np.nan] *4  ]),
                            
                            3, 3), 

                            # all zero values for valid components in one region group
                            ( np.array([ [ [3, 3, 0, 0] for i in range(2)],

                                  [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ]),

                            np.array([ [3,  3,  0, 0], [np.nan] *4  ]),
                            
                            3, 0), 

                            # all zero values for valid components in one region 
                            ( np.array([ [ [3, 3, 3, 0] for i in range(2)],

                                  [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ]),

                            np.array([ [3,  3,  3, 0], [np.nan] *4  ]),
                            
                            3, 3)
                           ])
def test_aggregate_time_series_weighted_mean(data, weight, expected_grp1, expected_grp2):

    component_list = ['source_comp','sink_comp']  
    space_list = ['01_reg','02_reg','03_reg','04_reg']
    time_list = ['T0','T1']

    data_xr = xr.DataArray(data, 
                            coords=[component_list, time_list, space_list], 
                            dims=['component', 'time','space'])
    
    weight_xr = xr.DataArray(weight, 
                            coords=[component_list, space_list], 
                            dims=['component', 'space'])

    sub_to_sup_region_id_dict = {'01_reg_02_reg': ['01_reg','02_reg'], 
                                 '03_reg_04_reg': ['03_reg','04_reg']}

    #FUNCTION CALL
    time_series_aggregated = spr.aggregate_time_series(data_xr, 
                                                        sub_to_sup_region_id_dict, 
                                                        mode="weighted mean", 
                                                        xr_weight_array=weight_xr)
    
    #ASSERTION 
    ## for valid component, source_comp
    assert time_series_aggregated.loc['source_comp', 'T0', '01_reg_02_reg'].values == expected_grp1
    assert time_series_aggregated.loc['source_comp', 'T0', '03_reg_04_reg'].values == expected_grp2

    ## for invalid component, sink_comp
    assert np.isnan(time_series_aggregated.loc['sink_comp', 'T0', '01_reg_02_reg'].values)

@pytest.mark.parametrize("mode, expected_grp1, expected_grp2", [("mean", 15, 0), ("sum", 30, 0), ("bool", 1, 0)])    
def test_aggregate_values(xr_and_dict_for_basic_representation, 
                        mode, 
                        expected_grp1,
                        expected_grp2):

    sub_to_sup_region_id_dict, xr_for_basic_representation = xr_and_dict_for_basic_representation

    test_xarray = xr_for_basic_representation['1d_capacityMax']
    
    #FUNCTION CALL 
    values_aggregated = spr.aggregate_values(test_xarray, 
                                            sub_to_sup_region_id_dict, 
                                            mode=mode)
    
    #ASSERTION
    ## for valid component, source_comp
    assert values_aggregated.loc['source_comp', '01_reg_02_reg'].values == expected_grp1 
    assert values_aggregated.loc['source_comp', '03_reg_04_reg'].values == expected_grp2 

    ## for invalid component, sink_comp
    assert np.isnan(values_aggregated.loc['sink_comp', '01_reg_02_reg'].values)

    
@pytest.mark.parametrize("mode, expected_for_valid_component", 
                                                [ ("mean", np.array([ [0, 5], [5, 0] ])), 
                                                  ("sum", np.array([ [0, 20], [20, 0] ])), 
                                                  ("bool", np.array([ [0, 1], [1, 0] ]) )
                                            ])
def test_aggregate_connections(xr_and_dict_for_basic_representation,
                               mode,
                               expected_for_valid_component):

    sub_to_sup_region_id_dict, xr_for_basic_representation = xr_and_dict_for_basic_representation
    
    test_xarray = xr_for_basic_representation['2d_capacityMax']

    #FUNCTION CALL
    connections_aggregated = spr.aggregate_connections(test_xarray,
                                                    sub_to_sup_region_id_dict,
                                                    mode=mode)

    #ASSERTION 
    output_for_valid_component = connections_aggregated.loc['transmission_comp'].values
    output_for_invalid_component = connections_aggregated.loc['sink_comp'].values

    
    assert np.array_equal(output_for_valid_component, expected_for_valid_component)
    assert output_for_invalid_component[0][0] == 0 and np.isnan(output_for_invalid_component[0][1])  
    


test_data = [
            # no aggregation_function_dict provided 
            (None, 6, 10, 30, 10, np.array([ [0, 20], [20, 0] ]), np.array([ [0, 1], [0, 0] ]) ), 

            ({'operationRateMax': ('weighted mean', xr.DataArray(np.array([ [15,  15,  15, 15],
                                                                            [np.nan] *4, 
                                                                            [np.nan] *4, 
                                                                        ]), 
                                                            coords=[['source_comp','sink_comp', 'transmission_comp'], 
                                                                    ['01_reg','02_reg','03_reg','04_reg']], 
                                                            dims=['component', 'space']) ),
            'operationRateFix': ('mean', None), 
            'capacityMax': ('sum', None), 
            'capacityFix': ('sum', None), 
            'locationalEligibility': ('sum', None)} , 

            3, 5, 30, 10, np.array([ [0, 20], [20, 0] ]), np.array([ [0, 1], [0, 0] ]) ),

            
            ({'operationRateMax': ('weighted mean', 'capacityMax'), 
                'operationRateFix': ('sum', None), 
                'capacityMax': ('sum', None), 
                'capacityFix': ('sum', None), 
                'locationalEligibility': ('bool', None)} , 
            3, 10, 30, 10, np.array([ [0, 20], [20, 0] ]), np.array([ [0, 1], [0, 0] ]) )] 
        

@pytest.mark.parametrize("aggregation_function_dict, expected_ts_operationRateMax, \
                          expected_ts_operationRateFix, expected_1d_capacityMax, \
                          expected_1d_capacityFix, expected_2d_capacityMax, \
                          expected_2d_locationalEligibility", test_data)

def test_aggregate_based_on_sub_to_sup_region_id_dict(xr_and_dict_for_basic_representation,
                                                      aggregation_function_dict, 
                                                      expected_ts_operationRateMax, 
                                                      expected_ts_operationRateFix, 
                                                      expected_1d_capacityMax, 
                                                      expected_1d_capacityFix, 
                                                      expected_2d_capacityMax,
                                                      expected_2d_locationalEligibility):  
    
    sub_to_sup_region_id_dict, test_xr = xr_and_dict_for_basic_representation

    output_xarray = spr.aggregate_based_on_sub_to_sup_region_id_dict(test_xr,
                                                                  sub_to_sup_region_id_dict,
                                                                  aggregation_function_dict=aggregation_function_dict,
                                                                  )

    #ASSERTION    
    ## Time series variables 
    assert  output_xarray['ts_operationRateMax'].loc['source_comp', 'T0', '01_reg_02_reg'].values == expected_ts_operationRateMax
    assert output_xarray['ts_operationRateFix'].loc['sink_comp', 'T0','01_reg_02_reg'].values == expected_ts_operationRateFix
    
    ## 1d variable 
    assert output_xarray['1d_capacityMax'].loc['source_comp', '01_reg_02_reg'].values == expected_1d_capacityMax
    assert output_xarray['1d_capacityFix'].loc['sink_comp', '01_reg_02_reg'].values == expected_1d_capacityFix

    ## 2d variable 
    output_2d_capacityMax = output_xarray['2d_capacityMax'].loc['transmission_comp'].values
    assert  np.array_equal(output_2d_capacityMax, expected_2d_capacityMax)

    output_2d_locationalEligibility = output_xarray['2d_locationalEligibility'].loc['transmission_comp'].values
    assert  np.array_equal(output_2d_locationalEligibility, expected_2d_locationalEligibility)








     
    

    
