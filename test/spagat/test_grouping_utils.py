import pytest 

import numpy as np 
import xarray as xr 
from sklearn.datasets import make_blobs

import FINE.spagat.grouping_utils as gu 
import FINE.spagat.dataset as spd


def test_matrix_MinMaxScaler():

    test_matrix = np.array([[10, 9, 8], [7, 4, 6], [2, 1, 0]])

    expected_matrix = 0.1 * test_matrix

    output_matrix = gu.matrix_MinMaxScaler(test_matrix, x_min=0, x_max=1)

    #floating points need to be converted to decimals, otherwise they do not match 
    output_matrix = np.round(output_matrix, 1)
    expected_matrix = np.round(expected_matrix, 1)

    assert np.array_equal(output_matrix, expected_matrix) 


def test_preprocessTimeSeries():
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
    output_dict = gu.preprocessTimeSeries(test_dict, len(space_list), len(component_list))

    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in var_list:
        assert np.array_equal(output_dict[var], expected_dict[var])

    

def test_preprocess1dVariables():

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
    output_dict = gu.preprocess1dVariables(test_dict, len(component_list))

    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in var_list:
        assert np.array_equal(output_dict[var], expected_dict[var])
     

@pytest.mark.parametrize("handle_mode", ['toAffinity', 'toDissimilarity'])
def test_preprocess2dVariables(handle_mode):

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


    # OPTION 1. handle_mode = 'toAffinity'
    if handle_mode == 'toAffinity':
        #EXPECTED DATA
        expected_dict = {}

        var1_c1_matrix = 0.1 * np.array([ [ 0,  1,  10],
                                        [ 1,  0,  1],
                                        [ 10, 1,  0] ])

        var1_c3_matrix = 0.1 * np.array([ [ 0,  2,  10],
                                        [ 2,  0,  2],
                                        [ 10, 2,  0] ])

        expected_dict['var1'] = {0: var1_c1_matrix, 2: var1_c3_matrix}

        var2_c2_matrix = 0.1 * np.array([ [ 0,  8,  10],
                                        [ 8,  0,  8],
                                        [ 10, 8,  0] ])
        
        var2_c4_matrix = 0.1 * np.array([ [ 0,  9,  10],
                                        [ 9,  0,  9],
                                        [ 10, 9,  0] ])

        expected_dict['var2'] = {1: var2_c2_matrix, 3: var2_c4_matrix}

        #FUNCTION CALL 
        output_dict = gu.preprocess2dVariables(test_dict, len(component_list), handle_mode=handle_mode)

        #ASSERTION 
        for (output_var, output_comp_dict), (expected_var, expected_comp_dict) in zip(output_dict.items(), expected_dict.items()):
            assert output_var == expected_var
            
            for (output_comp_index, output_comp_matrix), (expected_comp_index, expected_comp_matrix) in zip(output_comp_dict.items(), expected_comp_dict.items()):
                assert output_comp_index == expected_comp_index
                
                assert np.array_equal(output_comp_matrix, expected_comp_matrix)
    
    # OPTION 2. handle_mode = 'toDissimilarity'
    elif handle_mode == 'toDissimilarity':
        #EXPECTED DATA
        expected_dict = {}

        var1_c1_array = np.array([0.9, 0., 0.9])

        var1_c3_array = np.array([0.8, 0., 0.8])

        expected_dict['var1'] = {0: var1_c1_array, 2: var1_c3_array}

        var2_c2_array = np.array([0.2, 0., 0.2])
        
        var2_c4_array = np.array([0.1, 0., 0.1])

        expected_dict['var2'] = {1: var2_c2_array, 3: var2_c4_array}

        #FUNCTION CALL 
        output_dict = gu.preprocess2dVariables(test_dict, len(component_list), handle_mode=handle_mode)

        #ASSERTION  
        for (output_var, output_comp_dict), (expected_var, expected_comp_dict) in zip(output_dict.items(), expected_dict.items()):
            assert output_var == expected_var
            
            for (output_comp_index, output_comp_array), (expected_comp_index, expected_comp_array) in zip(output_comp_dict.items(), expected_comp_dict.items()):
                assert output_comp_index == expected_comp_index
                
                #floating points need to be converted to decimals, otherwise they do not match #TODO: do the same for in all tests where this problem can occur
                output_comp_array = np.round(output_comp_array, 1)
                expected_comp_array = np.round(expected_comp_array, 1)
                
                assert np.array_equal(output_comp_array, expected_comp_array)

    

@pytest.mark.parametrize("handle_mode", ['toAffinity', 'toDissimilarity'])
def test_preprocessDataset(handle_mode):
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

    # OPTION 1. handle_mode = 'toDissimilarity'
    if handle_mode == 'toDissimilarity':
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
        output_ts_dict, output_1d_dict, output_2d_dict = gu.preprocessDataset(sds, 
                                                                        handle_mode, 
                                                                        vars='all', 
                                                                        dims='all', 
                                                                        var_weightings=None) #TODO: test with different var_weightings
    
        
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

    # OPTION 2. handle_mode = 'toAffinity'
    elif handle_mode == 'toAffinity':
        #EXPECTED DATA
        ## time series matrix
        var_ts_1_c2_matrix = 0.1 * np.array([ [0, 1],
                                        [ 1,  1],
                                        [ 1,  10] ])

        var_ts_1_c4_matrix = 0.1 * np.array([ [ 0, 2],
                                        [ 2,  2],
                                        [ 2,  10]])

        var_ts_2_c3_matrix = 0.1 * np.array([ [0, 8],
                                        [ 8,  8],
                                        [ 8,  10] ])

        var_ts_2_c4_matrix = 0.1 * np.array([ [ 0, 9],
                                        [ 9,  9],
                                        [ 9,  10]])

        expected_ts_matrix = np.concatenate((var_ts_1_c2_matrix, 
                                            var_ts_1_c4_matrix, 
                                            var_ts_2_c3_matrix,
                                            var_ts_2_c4_matrix), axis=1)


        ## 1d dict
        var_1d_1_matrix = 0.1 * np.array([ [0,  1,  10],
                                            [0,  2,  10] ]).T

        var_1d_2_matrix = 0.1 * np.array([ [0,  8,  10],
                                            [0,  9,  10] ]).T

        expected_1d_matrix = np.concatenate((var_1d_1_matrix, 
                                            var_1d_2_matrix), axis=1)

        ## 2d dict 
        var_2d_1_c1_matrix = 0.1 * np.array([ [ 0,  1,  10],
                                            [ 1,  0,  1],
                                            [ 10, 1,  0] ])

        var_2d_1_c3_matrix = 0.1 * np.array([ [ 0,  2,  10],
                                            [ 2,  0,  2],
                                            [ 10, 2,  0] ])
        expected_2d_1_matrix = np.add(var_2d_1_c1_matrix, var_2d_1_c3_matrix)

        var_2d_2_c3_matrix = 0.1 * np.array([ [ 0,  8,  10],
                                            [ 8,  0,  8],
                                            [ 10, 8,  0] ])
        var_2d_2_c4_matrix = 0.1 * np.array([ [ 0,  9,  10],
                                            [ 9,  0,  9],
                                            [ 10, 9,  0] ])
        expected_2d_2_matrix = np.add(var_2d_2_c3_matrix, var_2d_2_c4_matrix)

        expected_2d_matrix = np.add(expected_2d_1_matrix, expected_2d_2_matrix)

        #FUNCTION CALL 
        output_ts_matrix, output_1d_matrix, output_2d_matrix = gu.preprocessDataset(sds, 
                                                                                    handle_mode, 
                                                                                    vars='all', 
                                                                                    dims='all', 
                                                                                    var_weightings=None) #TODO: test with different var_weightings


        # ASSERTION 
        ## ts 
        assert np.array_equal(output_ts_matrix, expected_ts_matrix)  
        ## 1d 
        assert np.array_equal(output_1d_matrix, expected_1d_matrix) 
        ## 2d 
        assert np.array_equal(output_2d_matrix, expected_2d_matrix)  
        

@pytest.mark.parametrize("reg_m, reg_n, dist_expected", [(0, 1, 16), (0, 2, 64), (1, 2, 48)])
def test_selfDistance(reg_m, reg_n, dist_expected):
    #TEST DATA 
    test_ts_dict = {}                  #TODO: add this data to a fixture as the next test also uses the same. 
                                         #check how you can pass this and parameters at the same time

    var_ts_1_c2_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3] ])

    var_ts_1_c4_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3]])

    test_ts_dict['var_ts_1'] = np.concatenate((var_ts_1_c2_matrix, var_ts_1_c4_matrix), axis=1)

    var_ts_2_c3_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3] ])

    var_ts_2_c4_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3]])

    test_ts_dict['var_ts_2'] = np.concatenate((var_ts_2_c3_matrix, var_ts_2_c4_matrix), axis=1)

    ## 1d dict
    test_1d_dict = {}

    test_1d_dict['var_1d_1'] = np.array([ [1, 1],
                                        [2, 2],
                                        [3, 3]])

    test_1d_dict['var_1d_2'] = np.array([ [1, 1],
                                        [2, 2],
                                        [3, 3]])

    ## 2d dict 
    test_2d_dict = {}

    var_2d_1_c1_array = np.array([1, 2, 3])
    var_2d_1_c3_array = np.array([1, 2, 3])
    test_2d_dict['var_2d_1'] = {0: var_2d_1_c1_array, 2: var_2d_1_c3_array}

    var_2d_2_c3_array = np.array([1, 2, 3])
    var_2d_2_c4_array = np.array([1, 2, 3])
    test_2d_dict['var_2d_2'] = {2: var_2d_2_c3_array, 3: var_2d_2_c4_array}          

    #FUNCTION CALL 
    n_regions = 3
    dist_output = gu.selfDistance(test_ts_dict, 
                                test_1d_dict, 
                                test_2d_dict, 
                                n_regions, reg_m, reg_n, 
                                var_weightings=None, part_weightings=None)  #TODO: test for different weightings

    #ASSERTION 
    assert dist_output == dist_expected

def test_selfDistanceMatrix():
    #TEST DATA                   #TODO: once a fixture is implemented, use that here 
    test_ts_dict = {}          

    var_ts_1_c2_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3] ])

    var_ts_1_c4_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3]])

    test_ts_dict['var_ts_1'] = np.concatenate((var_ts_1_c2_matrix, var_ts_1_c4_matrix), axis=1)

    var_ts_2_c3_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3] ])

    var_ts_2_c4_matrix = np.array([ [1, 1],
                                    [2, 2],
                                    [3, 3]])

    test_ts_dict['var_ts_2'] = np.concatenate((var_ts_2_c3_matrix, var_ts_2_c4_matrix), axis=1)

    ## 1d dict
    test_1d_dict = {}

    test_1d_dict['var_1d_1'] = np.array([ [1, 1],
                                        [2, 2],
                                        [3, 3]])

    test_1d_dict['var_1d_2'] = np.array([ [1, 1],
                                        [2, 2],
                                        [3, 3]])

    ## 2d dict 
    test_2d_dict = {}

    var_2d_1_c1_array = np.array([1, 2, 3])
    var_2d_1_c3_array = np.array([1, 2, 3])
    test_2d_dict['var_2d_1'] = {0: var_2d_1_c1_array, 2: var_2d_1_c3_array}

    var_2d_2_c3_array = np.array([1, 2, 3])
    var_2d_2_c4_array = np.array([1, 2, 3])
    test_2d_dict['var_2d_2'] = {2: var_2d_2_c3_array, 3: var_2d_2_c4_array}

    #EXPECTED DATA 
    expected_dist_matrix = np.array([ [0, 16, 64],
                                      [16, 0, 48],
                                      [64, 48, 0] ])
    
    #FUNCTION CALL 
    n_regions = 3
    output_dist_matrix = gu.selfDistanceMatrix(test_ts_dict, 
                                                test_1d_dict, 
                                                test_2d_dict, 
                                                 n_regions,              
                                                 var_weightings=None)        #TODO: test with different var weightings
    
    #ASSERTION 
    #floating points and ints need to be converted to decimals, otherwise they do not match 
    expected_dist_matrix = np.round(expected_dist_matrix, 1)
    output_dist_matrix = np.round(output_dist_matrix, 1)

    assert np.array_equal(expected_dist_matrix, output_dist_matrix)

@pytest.mark.parametrize("reg_m, reg_n, expected_bool", [(0, 1, True), (0, 2, True), (1, 2, False)])
def test_checkConnectivity(reg_m, reg_n, expected_bool):
    #TEST DATA
    test_dict = {}

    var1_c1_matrix = 0.1 * np.array([ [ 0,  1,  0],
                                      [ 1,  0,  0],
                                      [ 0,  0,  0] ])

    var1_c3_matrix = 0.1 * np.array([ [ 0,  0,  1],
                                      [ 0,  0,  0],
                                      [ 1,  0,  0] ])

    test_dict['var1'] = {0: var1_c1_matrix, 2: var1_c3_matrix}

    #FUNCTION CALL 
    connect_components = [0, 1, 2, 3]           
    output_bool = gu.checkConnectivity(reg_m, reg_n, test_dict, connect_components)

    #ASSERTION 
    assert output_bool == expected_bool



@pytest.mark.parametrize("component_list, expected_matrix", [(['c1','c2','c3','c4'], 
                                                               np.array([[ 1,  1,  1],
                                                                         [ 1,  1,  0],
                                                                         [ 1,  0,  1] ])
                                                              ), 
                                                              (['c1','c2','pipeline','c4'],
                                                                np.array([[ 1,  0,  1],
                                                                          [ 0,  1,  0],
                                                                          [ 1,  0,  1] ])
                                                              ) 
                                                            ])
def test_generateConnectivityMatrix(component_list, expected_matrix):

    #TEST DATA 
    space_list = ['01_reg','02_reg','03_reg']

    ## 2d variable data
    var_2d_1_data = np.array([ [[ 0,  1,  0],
                                [ 1,  0,  0],
                                [ 0,  0,  0]],

                                [[np.nan] * 3 for i in range(3)],

                                [[ 0,  0,  1],
                                 [ 0,  0,  0],
                                 [ 1,  0,  0]],
 
                                [[np.nan] * 3 for i in range(3)] ])

    var_2d_1_DataArray = xr.DataArray(var_2d_1_data, 
                                    coords=[component_list, space_list, space_list], 
                                    dims=['component', 'space', 'space_2'])

    ds = xr.Dataset({'var_2d_1': var_2d_1_DataArray})
    sds = spd.SpagatDataset()
    sds.xr_dataset = ds
    
    #FUNCTION CALL 
    output_matrix = gu.generateConnectivityMatrix(sds)

    #ASSERTION 
    assert np.array_equal(output_matrix, expected_matrix)


@pytest.mark.parametrize("regions_label_list, expected", [ ([0, 0, 0], 0.17), 
                                                           ([0, 0, 1],  0.06), 
                                                           ([0, 1, 1],  0.06), 
                                                           ([0, 1, 0],  0.06),
                                                           ([0, 1, 2],  0)
                                                           ])
def test_computeModularity(regions_label_list, expected):
    #NOTE: This test only checks for calcualtion errors. 
    # Cannot perform test similar to test_computeSilhouetteCoefficient(), 
    # Reason: sample data and labels obtained using make_blobs() is not suitable for modularity score check 
    # as sample_labels does not lead to high modularity score.
    # It was tested by taking the inverse of distance matrix to obtain adjacency_matrix   
           
    #TEST DATA 
    adjacency_matrix = np.array([ [ 10,  10,  10],
                                  [ 10,  10,  10],
                                  [ 10,  10,  10] ])
    
    #FUNCTION CALL 
    output = gu.computeModularity(adjacency_matrix, regions_label_list)
    
    #ASSERTION 
    #floating points need to be converted to decimals, otherwise they do not match 
    output = np.round(output, 2)

    assert output == expected                   



def test_computeSilhouetteCoefficient():
    #TEST DATA 
    sample_data, sample_labels = make_blobs(n_samples=5, centers=3, n_features=2, random_state=0) #NOTE: with random_state=0, sample data 
                                                                                                  # distribution is such that sample_labels is always 
                                                                                                  # array([2, 0, 0, 1, 1])
                                                                                                  

    test_dist_matrix = np.zeros((5,5))
    for i in range(5):
        for j in range(i+1, 5): 
            test_dist_matrix[i][j] = sum(np.power((sample_data[i] - sample_data[j]), 2))
    
    test_dist_matrix += test_dist_matrix.T

    regions_list = ['01_reg','02_reg','03_reg','04_reg','05_reg']
    aggregation_dict = {5: {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg'], '04_reg': ['04_reg'], '05_reg': ['05_reg']},
                        4: {'01_reg': ['01_reg'], '02_reg_03_reg_04_reg_05_reg': ['02_reg', '03_reg', '04_reg', '05_reg']},
                        3: {'01_reg': ['01_reg'], '02_reg_03_reg': ['02_reg', '03_reg'], '04_reg_05_reg': ['04_reg', '05_reg']},
                        2: {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg_04_reg_05_reg': ['03_reg', '04_reg', '05_reg']},
                        1: {'01_reg_02_reg_03_reg_04_reg_05_reg': ['01_reg','02_reg','03_reg','04_reg','05_reg']}}
    
    #FUNCTION CALL 
    output_list = gu.computeSilhouetteCoefficient(regions_list, test_dist_matrix, aggregation_dict)
    
    #ASSERTION 
    assert output_list[2] < output_list[0] < output_list[1]
    
    
    



