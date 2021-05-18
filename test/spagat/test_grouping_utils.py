import pytest 

import numpy as np 
import xarray as xr 
from sklearn.datasets import make_blobs

import FINE.spagat.grouping_utils as gu 
import FINE.spagat.dataset as spd


def test_get_scaled_matrix():

    test_matrix = np.array([[10, 9, 8], [7, 4, 6], [2, 1, 0]])

    expected_matrix = 0.1 * test_matrix

    output_matrix = gu.get_scaled_matrix(test_matrix, scaled_min=0, scaled_max=1)

    #floating points need to be converted to decimals, otherwise they do not match 
    output_matrix = np.round(output_matrix, 1)
    expected_matrix = np.round(expected_matrix, 1)

    assert np.array_equal(output_matrix, expected_matrix) 


def test_preprocess_time_series():
    #TEST DATA 
    test_dict = {}

    var_list = ['var1', 'var2']
    component_list = ['c1','c2','c3','c4']
    space_list = ['01_reg','02_reg','03_reg']
    TimeStep_list = ['T0','T1']

    var1_data = np.array([ [[ 0, 1],
                            [ 1, 1],
                            [ 1, 10]],
                            
                            [[ 0, 2],
                             [ 2, 2],
                             [ 2, 10]],

                            [[np.nan] * 2 for i in range(3)],

                            [[np.nan] * 2 for i in range(3)] ])

    test_dict['var1'] = xr.DataArray(var1_data, 
                                  coords=[component_list, space_list, TimeStep_list], 
                                  dims=['component', 'space', 'TimeStep'],
                                  name = 'var1')


    var2_data = np.array([ [[np.nan] * 2 for i in range(3)],

                            [[np.nan] * 2 for i in range(3)],

                            [[ 0, 8],
                             [ 8, 8],
                             [ 8, 10]],

                            [[ 0, 9],
                             [ 9, 9], 
                             [ 9, 10]] ])
                        
    test_dict['var2'] = xr.DataArray(var2_data, 
                                    coords=[component_list, space_list, TimeStep_list], 
                                    dims=['component', 'space', 'TimeStep'],
                                    name = 'var2')

    
    #EXPECTED DATA
    expected_dict = {}

    c1_matrix = 0.1 * np.array([ [0, 1],
                                [ 1,  1],
                                [ 1,  10] ])
    
    c2_matrix = 0.1 * np.array([ [ 0, 2],
                                [ 2,  2],
                                [ 2,  10]])

    expected_dict['var1'] = np.concatenate((c1_matrix, c2_matrix), axis=1)


    c3_matrix = 0.1 * np.array([ [ 0,  8],           #NOTE: some of the numbers like 6 and 3 throw assertion error due to difference in resulting floating values
                                [ 8,  8],
                                [ 8,  10]])
    
    c4_matrix = 0.1 * np.array([ [ 0,  9],
                                [ 9,  9],
                                [ 9,  10]])

    expected_dict['var2'] = np.concatenate((c3_matrix, c4_matrix), axis=1)

    # FUNCTION CALL 
    output_dict = gu.preprocess_time_series(test_dict, len(space_list), len(component_list))

    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in var_list:
        assert np.array_equal(output_dict[var], expected_dict[var])

    

def test_preprocess_1d_variables():

    #TEST DATA 
    test_dict = {} 

    var_list = ['var1', 'var2']
    component_list = ['c1','c2','c3','c4']
    space_list = ['01_reg','02_reg','03_reg']
    
    var1_data = np.array([ [0,  1,  10],
                           [0,  2,  10],
                           [np.nan] *3,
                           [np.nan] *3 ])

    test_dict['var1'] = xr.DataArray(var1_data, 
                                    coords=[component_list, space_list], 
                                    dims=['component', 'space'],
                                    name = 'var1')

    var2_data = np.array([ [np.nan] *3,
                       [np.nan] *3, 
                       [0,  8,  10],
                       [0,  9,  10] ])

    test_dict['var2'] = xr.DataArray(var2_data, 
                                    coords=[component_list, space_list], 
                                    dims=['component', 'space'],
                                    name = 'var2')

    #EXPECTED DATA
    expected_dict = {}

    expected_dict['var1'] = 0.1 * np.array([ [0,  1,  10],
                                             [0,  2,  10] ]).T


    expected_dict['var2'] = 0.1 * np.array([ [0,  8,  10],
                                             [0,  9,  10] ]).T

    #FUNCTION CALL 
    output_dict = gu.preprocess_1d_variables(test_dict, len(component_list))

    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in var_list:
        assert np.array_equal(output_dict[var], expected_dict[var])
     

def test_preprocess_2d_variables():

    #TEST DATA 
    test_dict = {} 

    component_list = ['c1','c2','c3','c4']
    space_list = ['01_reg','02_reg','03_reg']  #TODO: test a scenario where order of space and space_2 are different

    var1_data = np.array([ [[ 0,  1,  10],
                            [ 1,  0,  1],
                            [ 10, 1,  0]],
                      
                            [[np.nan] * 3 for i in range(3)],
                            
                            [[ 0,  2,  10],
                            [ 2,  0,  2],
                            [ 10, 2,  0]],

                            [[np.nan] * 3 for i in range(3)] ])

    test_dict['var1'] = xr.DataArray(var1_data, 
                                    coords=[component_list, space_list, space_list], 
                                    dims=['component', 'space', 'space_2'],
                                    name = 'var1')

    var2_data = np.array([ [[np.nan] * 3 for i in range(3)],

                            [[ 0,  8,  10],
                            [ 8,  0,  8],
                            [ 10, 8,  0]],
                      
                            [[np.nan] * 3 for i in range(3)],
                          
                            [[ 0,  9,  10],
                            [ 9,  0,  9],
                            [ 10, 9,  0]]  ])

    test_dict['var2'] = xr.DataArray(var2_data, 
                                    coords=[component_list, space_list, space_list], 
                                    dims=['component', 'space', 'space_2'],
                                    name = 'var2')

    #EXPECTED DATA
    expected_dict = {}

    var1_c1_array = np.array([0.9, 0., 0.9])

    var1_c3_array = np.array([0.8, 0., 0.8])

    expected_dict['var1'] = {0: var1_c1_array, 2: var1_c3_array}

    var2_c2_array = np.array([0.2, 0., 0.2])
    
    var2_c4_array = np.array([0.1, 0., 0.1])

    expected_dict['var2'] = {1: var2_c2_array, 3: var2_c4_array}

    #FUNCTION CALL 
    output_dict = gu.preprocess_2d_variables(test_dict, len(component_list))

    #ASSERTION  
    for (output_var, output_comp_dict), (expected_var, expected_comp_dict) in zip(output_dict.items(), expected_dict.items()):
        assert output_var == expected_var
        
        for (output_comp_index, output_comp_array), (expected_comp_index, expected_comp_array) in zip(output_comp_dict.items(), expected_comp_dict.items()):
            assert output_comp_index == expected_comp_index
            
            #floating points need to be converted to decimals, otherwise they do not match #TODO: do the same for in all tests where this problem can occur
            output_comp_array = np.round(output_comp_array, 1)
            expected_comp_array = np.round(expected_comp_array, 1)
            
            assert np.array_equal(output_comp_array, expected_comp_array)

    
def test_preprocess_dataset():
    #TEST DATA 
    #var_list = ['var_ts_1', 'var_ts_2', 'var_1d_1', 'var_1d_2', 'var_2d_1', 'var_2d_2']
    component_list = ['c1','c2','c3','c4']

    space_list = ['01_reg','02_reg','03_reg']
    TimeStep_list = ['T0','T1']

    Period_list = [0]

    ## time series variable data
    var_ts_1_data = np.array([ [ [[np.nan, np.nan, np.nan] for i in range(2)] ],
                            
                                [ [[0, 1,  1],
                                    [1, 1,  10]] ],
                                
                                [ [[np.nan,np.nan, np.nan] for i in range(2)]  ],
                                
                                [ [[0,   2, 2],
                                    [2, 2, 10]] ] ])

    var_ts_1_DataArray = xr.DataArray(var_ts_1_data, 
                                    coords=[component_list, Period_list, TimeStep_list, space_list], 
                                    dims=['component', 'Period', 'TimeStep','space'])

    var_ts_2_data = np.array([ [ [[np.nan, np.nan, np.nan] for i in range(2)] ],
                            
                                [ [[np.nan,np.nan, np.nan] for i in range(2)]  ],
                                
                                [ [[0, 8,  8],
                                    [8, 8,  10]] ],
                                
                                [ [[0,   9, 9],
                                    [9, 9, 10]] ] ])

    var_ts_2_DataArray = xr.DataArray(var_ts_2_data, 
                                    coords=[component_list, Period_list, TimeStep_list, space_list], 
                                    dims=['component', 'Period', 'TimeStep','space'])
        
    ## 1d variable data
    var_1d_1_data = np.array([ [0,  1,  10],
                            [0,  2,  10],
                            [np.nan] *3,
                            [np.nan] *3 ])

    var_1d_1_DataArray = xr.DataArray(var_1d_1_data, 
                                        coords=[component_list, space_list], 
                                        dims=['component', 'space'])


    var_1d_2_data = np.array([ [0,  8,  10],
                            [np.nan] *3,
                            [np.nan] *3, 
                            [0,  9,  10] ])

    var_1d_2_DataArray = xr.DataArray(var_1d_2_data, 
                                        coords=[component_list, space_list], 
                                        dims=['component', 'space'])

    ## 2d variable data
    var_2d_1_data = np.array([ [[ 0,  1,  10],
                                [ 1,  0,  1],
                                [ 10, 1,  0]],
                        
                                [[np.nan] * 3 for i in range(3)],
                                
                                [[ 0,  2,  10],
                                [ 2,  0,  2],
                                [ 10, 2,  0]],

                                [[np.nan] * 3 for i in range(3)] ])

    var_2d_1_DataArray = xr.DataArray(var_2d_1_data, 
                                        coords=[component_list, space_list, space_list], 
                                        dims=['component', 'space', 'space_2'])

    var_2d_2_data = np.array([ [[np.nan] * 3 for i in range(3)],
                            
                            [[np.nan] * 3 for i in range(3)],
                            
                            [[ 0,  8,  10],
                                [ 8,  0,  8],
                                [ 10, 8,  0] ],
                                
                            [[ 0,  9,  10],
                                [ 9,  0,  9],
                                [ 10, 9,  0]]  ])

    var_2d_2_DataArray = xr.DataArray(var_2d_2_data, 
                                        coords=[component_list, space_list, space_list], 
                                        dims=['component', 'space', 'space_2'])
    
    ds = xr.Dataset({'var_ts_1': var_ts_1_DataArray, 
                    'var_ts_2': var_ts_2_DataArray,
                    'var_1d_1': var_1d_1_DataArray, 
                    'var_1d_2': var_1d_2_DataArray, 
                    'var_2d_1': var_2d_1_DataArray,
                    'var_2d_2': var_2d_2_DataArray})
    sds = spd.SpagatDataset()
    sds.xr_dataset = ds

    #EXPECTED DATA
    ## time series dict
    expected_ts_dict = {}

    var_ts_1_c2_matrix = 0.1 * np.array([ [0, 1],
                                    [ 1,  1],
                                    [ 1,  10] ])
        
    var_ts_1_c4_matrix = 0.1 * np.array([ [ 0, 2],
                                    [ 2,  2],
                                    [ 2,  10]])

    expected_ts_dict['var_ts_1'] = np.concatenate((var_ts_1_c2_matrix, var_ts_1_c4_matrix), axis=1)

    var_ts_2_c3_matrix = 0.1 * np.array([ [0, 8],
                                    [ 8,  8],
                                    [ 8,  10] ])
        
    var_ts_2_c4_matrix = 0.1 * np.array([ [ 0, 9],
                                    [ 9,  9],
                                    [ 9,  10]])

    expected_ts_dict['var_ts_2'] = np.concatenate((var_ts_2_c3_matrix, var_ts_2_c4_matrix), axis=1)

    ## 1d dict
    expected_1d_dict = {}

    expected_1d_dict['var_1d_1'] = 0.1 * np.array([ [0,  1,  10],
                                        [0,  2,  10] ]).T

    expected_1d_dict['var_1d_2'] = 0.1 * np.array([ [0,  8,  10],
                                        [0,  9,  10] ]).T

    ## 2d dict 
    expected_2d_dict = {}

    var_2d_1_c1_array = np.array([0.9, 0., 0.9])
    var_2d_1_c3_array = np.array([0.8, 0., 0.8])
    expected_2d_dict['var_2d_1'] = {0: var_2d_1_c1_array, 2: var_2d_1_c3_array}
    
    var_2d_2_c3_array = np.array([0.2, 0., 0.2])
    var_2d_2_c4_array = np.array([0.1, 0., 0.1])
    expected_2d_dict['var_2d_2'] = {2: var_2d_2_c3_array, 3: var_2d_2_c4_array}
        
    #FUNCTION CALL 
    output_ts_dict, output_1d_dict, output_2d_dict = gu.preprocess_dataset(sds) 
    
    # ASSERTION 
    ## ts 
    for (output_var, output_matrix), (expected_var, expected_matrix) in zip(output_ts_dict.items(), expected_ts_dict.items()):
        assert output_var == expected_var
        assert np.array_equal(output_matrix, expected_matrix)  

    ## 1d 
    for (output_var, output_matrix), (expected_var, expected_matrix) in zip(output_1d_dict.items(), expected_1d_dict.items()):
        assert output_var == expected_var
        print(output_matrix)
        print(expected_matrix)
        assert np.array_equal(output_matrix, expected_matrix)  

    
    ## 2d 
    for (output_var, output_comp_dict), (expected_var, expected_comp_dict) in zip(output_2d_dict.items(), expected_2d_dict.items()):
        assert output_var == expected_var
        
        for (output_comp_index, output_comp_array), (expected_comp_index, expected_comp_array) in zip(output_comp_dict.items(), expected_comp_dict.items()):
            assert output_comp_index == expected_comp_index
            
            #floating points need to be converted to decimals, otherwise they do not match 
            output_comp_array = np.round(output_comp_array, 1)
            expected_comp_array = np.round(expected_comp_array, 1)
            
            assert np.array_equal(output_comp_array, expected_comp_array)



@pytest.mark.parametrize("var_category_weights, var_weights, expected_dist_matrix", 
                        [ (None, None, np.array([ [0, 16, 64],
                                                [16, 0, 48],
                                                [64, 48, 0] ]) 
                        ),
                          ({'ts_vars' : 1, '1d_vars' : 2, '2d_vars' : 3}, None,  np.array([ [0, 28, 112],
                                                                                            [28, 0, 124],
                                                                                            [112, 124, 0] ]) 
                        ), 
                          (None, {'operationRateMax' : 2, 'losses' : 3}, np.array([ [0, 24, 96],
                                                                                    [24, 0, 88],
                                                                                    [96, 88, 0] ]) 
                        ),
                          ({'2d_vars' : 3}, {'capacityMax' : 2, 'capacityFix': 2}, np.array([ [0, 28, 112],
                                                                                                                [28, 0, 124],
                                                                                                                [112, 124, 0] ])
                        )
                        ])
def test_get_custom_distance_matrix(var_category_weights, 
                                    var_weights, 
                                    expected_dist_matrix,
                                    data_for_distance_measure):
    
    test_ts_dict, test_1d_dict, test_2d_dict = data_for_distance_measure 

    #FUNCTION CALL 
    n_regions = 3
    output_dist_matrix = gu.get_custom_distance_matrix(test_ts_dict, 
                                                    test_1d_dict, 
                                                    test_2d_dict, 
                                                    n_regions,
                                                    var_category_weights,
                                                    var_weights)     
    
    #ASSERTION 
    #floating points and ints need to be converted to decimals, otherwise they do not match 
    expected_dist_matrix = np.round(expected_dist_matrix, 1)
    output_dist_matrix = np.round(output_dist_matrix, 1)

    assert np.array_equal(expected_dist_matrix, output_dist_matrix)


def test_get_connectivity_matrix(sds_for_connectivity):
    #EXPECTED 
    expected_matrix = np.array([[ 1, 1, 0, 1, 1, 0, 0, 0],
                                [ 1, 1, 1, 1, 1, 1, 0, 0],
                                [ 0, 1, 1, 0, 1, 1, 0, 0],
                                [ 1, 1, 0, 1, 1, 0, 1, 0],
                                [ 1, 1, 1, 1, 1, 1, 0, 0],
                                [ 0, 1, 1, 0, 1, 1, 0, 1],
                                [ 0, 0, 0, 1, 0, 0, 1, 1],
                                [ 0, 0, 0, 0, 0, 1, 1, 1]])

    #FUNCTION CALL 
    output_matrix = gu.get_connectivity_matrix(sds_for_connectivity)

    #ASSERTION 
    assert np.array_equal(output_matrix, expected_matrix)


    
    
    



