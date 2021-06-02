import os
import pathlib

import pytest
import numpy as np
import xarray as xr
import math 
from shapely.geometry import Point, Polygon

import FINE.spagat.representation as spr
import FINE.spagat.grouping as spg
import FINE.spagat.dataset as spd

def test_add_region_centroids_and_distances():
    test_geometries = [Polygon([(0,0), (2,0), (2,2), (0,2)]),
                       Polygon([(2,0), (4,0), (4,2), (2,2)])] 

    expected_centroids = [Point(1, 1), Point(3,1)]
    expected_centroid_distances = .001 * np.array([ [0, 2], [2, 0] ]) #Distances in km

    #FUNCTION CALL                                         
    sds = spd.SpagatDataset()
    sds.add_objects(description ='gpd_geometries',  
                    dimension_list =['space'], 
                    object_list = test_geometries)
    
    spr.add_region_centroids(sds) 
    spr.add_centroid_distances(sds)

    output_centroids = sds.xr_dataset['gpd_centroids'].values
    output_centroid_distances = sds.xr_dataset['centroid_distances'].values

    #ASSERTION 
    for output, expected in zip(output_centroids, expected_centroids):
        assert output.coords[:] == expected.coords[:]
    
    assert np.array_equal(output_centroid_distances, expected_centroid_distances)


def test_aggregate_geometries(sds_and_dict_for_basic_representation):

    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation
    
    test_xarray = sds_for_basic_representation.xr_dataset['gpd_geometries']

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
def test_aggregate_time_series_mean_and_sum(sds_and_dict_for_basic_representation, 
                               mode, 
                               expected):

    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation

    test_xarray = sds_for_basic_representation.xr_dataset['ts_operationRateMax']
    
    
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
def test_aggregate_values(sds_and_dict_for_basic_representation, 
                        mode, 
                        expected_grp1,
                        expected_grp2):

    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation

    test_xarray = sds_for_basic_representation.xr_dataset['1d_capacityMax']
    
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
def test_aggregate_connections(sds_and_dict_for_basic_representation,
                               mode,
                               expected_for_valid_component):

    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation
    
    test_xarray = sds_for_basic_representation.xr_dataset['2d_capacityMax']

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

def test_aggregate_based_on_sub_to_sup_region_id_dict(sds_and_dict_for_basic_representation,
                                                      aggregation_function_dict, 
                                                      expected_ts_operationRateMax, 
                                                      expected_ts_operationRateFix, 
                                                      expected_1d_capacityMax, 
                                                      expected_1d_capacityFix, 
                                                      expected_2d_capacityMax,
                                                      expected_2d_locationalEligibility):  
    
    sub_to_sup_region_id_dict, sds = sds_and_dict_for_basic_representation

    output_sds = spr.aggregate_based_on_sub_to_sup_region_id_dict(sds,
                                                                  sub_to_sup_region_id_dict,
                                                                  aggregation_function_dict=aggregation_function_dict,
                                                                  )

    #ASSERTION
    output_xarray = output_sds.xr_dataset
    
    ## 3. Time series variables 
    assert  output_xarray['ts_operationRateMax'].loc['source_comp', 'T0', '01_reg_02_reg'].values == expected_ts_operationRateMax
    assert output_xarray['ts_operationRateFix'].loc['sink_comp', 'T0','01_reg_02_reg'].values == expected_ts_operationRateFix
    
    ## 4. 1d variable 
    assert output_xarray['1d_capacityMax'].loc['source_comp', '01_reg_02_reg'].values == expected_1d_capacityMax
    assert output_xarray['1d_capacityFix'].loc['sink_comp', '01_reg_02_reg'].values == expected_1d_capacityFix

    ## 5. 2d variable 
    output_2d_capacityMax = output_xarray['2d_capacityMax'].loc['transmission_comp'].values
    assert  np.array_equal(output_2d_capacityMax, expected_2d_capacityMax)

    output_2d_locationalEligibility = output_xarray['2d_locationalEligibility'].loc['transmission_comp'].values
    assert  np.array_equal(output_2d_locationalEligibility, expected_2d_locationalEligibility)


# spagat.output
def test_create_grid_shapefile(sds_and_dict_for_basic_representation):
    sds_for_basic_representation = sds_and_dict_for_basic_representation.sds

    
    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output')
    files_name="test_ac_lines"

    spr.create_grid_shapefile(sds_for_basic_representation,
                            variable_description="var_2d",
                            component_description='c1',
                            file_path=path_to_test_dir,
                            files_name=files_name,
                            spatial_dim="space")
    
    #EXPECTED 
    ## File extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    #ASSERTION
    for file_extension in file_extensions_list:
        expected_file = os.path.join(path_to_test_dir, f'{files_name}{file_extension}')
        assert os.path.isfile(expected_file)

        os.remove(expected_file)


# spagat.output
def test_create_grid_shapefile(sds_and_dict_for_basic_representation):
    sds_for_basic_representation = sds_and_dict_for_basic_representation.sds

    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output')
    files_name="test_ac_lines"

    spr.create_grid_shapefile(sds_for_basic_representation,
                            variable_description="2d_capacityMax",
                            component_description="transmission_comp",
                            file_path=path_to_test_dir,
                            files_name=files_name)
    
    #EXPECTED 
    ## File extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    #ASSERTION
    for file_extension in file_extensions_list:
        expected_file = os.path.join(path_to_test_dir, f'{files_name}{file_extension}')
        assert os.path.isfile(expected_file)

        os.remove(expected_file)


test_data = [(None, 10, 10, np.array([ [0, 20], [20, 0] ]) ), 
             ({'var_ts': ('mean', None), 'var_1d': ('sum', None), 'var_2d': ('bool', None)}, 5, 10, np.array([ [0, 1], [1, 0] ]) ) 
            ]  



     
    

    
