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
    sds.add_objects(description ='gpd_geometries',   #NOTE: not sure if it is ok to call another function here
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


@pytest.mark.parametrize("mode, expected", [("mean", 5), ("weighted mean", 5), ("sum", 10)])
def test_aggregate_time_series(sds_and_dict_for_basic_representation, 
                               mode, 
                               expected):

    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation

    test_xarray = sds_for_basic_representation.xr_dataset['var_ts']
    
    #FUNCTION CALL
    time_series_aggregated = spr.aggregate_time_series(test_xarray, 
                                                        sub_to_sup_region_id_dict, 
                                                        mode=mode, 
                                                        xr_weight_array=test_xarray)
    
    #ASSERTION 
    ## for valid component, c1
    assert time_series_aggregated.loc['c1', 0, 'T0', '01_reg_02_reg'].values == expected
    ## for invalid component, c1
    if mode == 'sum': #NOTE: sum of nan gives 0 as output 
        assert time_series_aggregated.loc['c2', 0, 'T0', '01_reg_02_reg'].values == 0
    else:  
        assert math.isnan( time_series_aggregated.loc['c2', 0, 'T0', '01_reg_02_reg'].values )


@pytest.mark.parametrize("mode", ["mean", "sum", "bool"])       
def test_aggregate_values(sds_and_dict_for_basic_representation, mode):
    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation

    test_xarray = sds_for_basic_representation.xr_dataset['var_1d']
    
    #FUNCTION CALL 
    values_aggregated = spr.aggregate_values(test_xarray, 
                                            sub_to_sup_region_id_dict, 
                                            mode=mode)
    
    #ASSERTION
    if mode == "mean":
        assert values_aggregated.loc['c1', '01_reg_02_reg'].values == 5
        assert math.isnan(values_aggregated.loc['c2', '01_reg_02_reg'].values)

    elif mode == "sum":
        assert values_aggregated.loc['c1', '01_reg_02_reg'].values == 10
        assert values_aggregated.loc['c2', '01_reg_02_reg'].values == 0
    
    elif mode == "bool":  #NOTE: bool uses any. It outputs 1 if at least one value is TRUE. NA is also considered as TRUE #TODO: does bool mode make sense in this function
        assert  values_aggregated.loc['c1', '01_reg_02_reg'].values == 1
        assert  values_aggregated.loc['c2', '01_reg_02_reg'].values == 1



@pytest.mark.parametrize("set_diagonal_to_zero", [True, False])
@pytest.mark.parametrize("mode", ["mean", "sum", "bool"])
def test_aggregate_connections(sds_and_dict_for_basic_representation,
                               set_diagonal_to_zero,
                               mode):

    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation
    
    test_xarray = sds_for_basic_representation.xr_dataset['var_2d']

    #FUNCTION CALL
    connections_aggregated = spr.aggregate_connections(test_xarray,
                                                    sub_to_sup_region_id_dict,
                                                    mode=mode,
                                                    set_diagonal_to_zero=set_diagonal_to_zero,  #TODO: test for Flase as well
                                                    spatial_dim="space")

    #ASSERTION 
    output_for_valid_component = connections_aggregated.loc['c1'].values
    output_for_invalid_component = connections_aggregated.loc['c2'].values

    if set_diagonal_to_zero is True: 
        if mode == "mean":
            expected_for_valid_component = np.array([ [0, 5], [5, 0] ])

            assert np.array_equal(output_for_valid_component, expected_for_valid_component)
            assert output_for_invalid_component[0][0] == 0 and np.isnan(output_for_invalid_component[0][1]) #NOTE: cannot directly compare np.nan and nan
        
        else:
            if mode == "sum":
                expected_for_valid_component = np.array([ [0, 20], [20, 0] ])
                expected_for_invalid_component = np.array([ [0, 0], [0, 0] ]) 
            
            elif mode == "bool":
                expected_for_valid_component = np.array([ [0, 1], [1, 0] ])
                expected_for_invalid_component = np.array([ [0, 0], [0, 0] ]) 

            assert np.array_equal(output_for_valid_component, expected_for_valid_component)
            assert np.array_equal(output_for_invalid_component, expected_for_invalid_component)  
    
    else: 
        if mode == "mean":
            expected_for_valid_component = np.array([ [2.5, 5], [5, 2.5] ])

            assert np.array_equal(output_for_valid_component, expected_for_valid_component)
            assert np.isnan(output_for_invalid_component[0][0]) and np.isnan(output_for_invalid_component[0][1]) 
        
        else:
            if mode == "sum":
                expected_for_valid_component = np.array([ [10, 20], [20, 10] ])
                expected_for_invalid_component = np.array([ [0, 0], [0, 0] ])
            
            elif mode == "bool":
                expected_for_valid_component = np.array([ [1, 1], [1, 1] ])
                expected_for_invalid_component = np.array([ [0, 0], [0, 0] ]) 

            assert np.array_equal(output_for_valid_component, expected_for_valid_component)
            assert np.array_equal(output_for_invalid_component, expected_for_invalid_component)


test_data = [(None, 10, 10, np.array([ [0, 20], [20, 0] ]) ), 
             ({'var_ts': ('mean', None), 'var_1d': ('sum', None), 'var_2d': ('bool', None)}, 5, 10, np.array([ [0, 1], [1, 0] ]) ) 
            ]  #TODO: test with weighted mean also

@pytest.mark.parametrize("aggregation_function_dict, expected_ts, expected_1d, expected_2d", test_data)
def test_aggregate_based_on_sub_to_sup_region_id_dict(sds_and_dict_for_basic_representation,
                                                      aggregation_function_dict, 
                                                      expected_ts, 
                                                      expected_1d, 
                                                      expected_2d):  
    sub_to_sup_region_id_dict, sds_for_basic_representation = sds_and_dict_for_basic_representation

    output_sds = spr.aggregate_based_on_sub_to_sup_region_id_dict(sds_for_basic_representation,
                                                                  sub_to_sup_region_id_dict,
                                                                  aggregation_function_dict=aggregation_function_dict)

    #ASSERTION
    output_xarray = output_sds.xr_dataset
    
    ## 1. Geometries 
    assert list(output_xarray.space.values) == list(sub_to_sup_region_id_dict.keys())

    for joined_polygon in output_xarray['gpd_geometries'].values:                                                   
        assert len(joined_polygon.exterior.coords) == 7 
        assert (joined_polygon.geom_type in ('MultiPolygon'))
    
    ## 2. Centroids  
    expected_centroids = [Point(2, 1), Point(2,2)]
    output_centroids = output_xarray['gpd_centroids'].values

    for output, expected in zip(output_centroids, expected_centroids):
        assert output.coords[:] == expected.coords[:]

    ## 3. Time series variable 
    assert output_xarray['var_ts'].loc['c1', 0, 'T0','01_reg_02_reg'].values == expected_ts
    
    ## 4. 1d variable 
    assert output_xarray['var_1d'].loc['c1', '01_reg_02_reg'].values == expected_1d

    ## 5. 2d variable 
    output= output_xarray['var_2d'].loc['c1'].values
    assert  np.array_equal(output, expected_2d)



# spagat.output
def test_create_grid_shapefile(sds_and_dict_for_basic_representation):
    sds_for_basic_representation = sds_and_dict_for_basic_representation.sds

    
    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output')
    files_name="test_ac_lines"

    spr.create_grid_shapefile(sds_for_basic_representation,
                          file_path=path_to_test_dir,
                          files_name=files_name,
                          spatial_dim="space",
                          eligibility_variable="var_2d",
                          eligibility_component='c1')
    
    #EXPECTED 
    ## File extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    #ASSERTION
    for file_extension in file_extensions_list:
        expected_file = os.path.join(path_to_test_dir, f'{files_name}{file_extension}')
        assert os.path.isfile(expected_file)

        os.remove(expected_file)




     
    

    
