import pytest 

import numpy as np 
import xarray as xr 

import FINE.spagat.grouping_utils as gu 

@pytest.mark.parametrize("test_array", 
                        [ np.array([ [10, 9, 8], 
                                     [7, 4, 6], 
                                     [2, 1, 0] ]), 
                          np.array([10, 5, 0])
                        ])
def test_get_scaled_array(test_array):

    expected_array = 0.1 * test_array

    output_array = gu.get_scaled_array(test_array)

    assert np.isclose(output_array, expected_array).all()

def test_preprocess_time_series():
    #TEST DATA 
    test_dict = {}

    var_list = ['var1', 'var2']
    component_list = ['c1','c2','c3','c4']
    space_list = ['01_reg','02_reg','03_reg']
    time_list = ['T0','T1']

    var1_data = np.array([ [[ 0, 1],
                            [ 1, 1],
                            [ 1, 10]],
                            
                            [[ 0, 2],
                             [ 2, 2],
                             [ 2, 10]],

                            [[np.nan] * 2 for i in range(3)],

                            [[np.nan] * 2 for i in range(3)] ])

    test_dict['var1'] = xr.DataArray(var1_data, 
                                  coords=[component_list, space_list, time_list], 
                                  dims=['component', 'space', 'time'],
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
                                    coords=[component_list, space_list, time_list], 
                                    dims=['component', 'space', 'time'],
                                    name = 'var2')

    
    #EXPECTED DATA
    expected_dict = {}

    c1_matrix = 0.1 * np.array([ [0, 1], [ 1,  1], [ 1,  10] ])
    c2_matrix = 0.1 * np.array([ [ 0, 2], [ 2,  2], [ 2,  10]])

    expected_dict['var1'] = {'c1': c1_matrix, 'c2' : c2_matrix}

    c3_matrix = 0.1 * np.array([ [ 0,  8], [ 8,  8], [ 8,  10]])
    c4_matrix = 0.1 * np.array([ [ 0,  9], [ 9,  9], [ 9,  10]])

    expected_dict['var2'] = {'c3': c3_matrix, 'c4' : c4_matrix}

    # FUNCTION CALL 
    output_dict = gu.preprocess_time_series(test_dict)

    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in output_dict.keys():
        expected_var_dict = expected_dict[var]
        output_var_dict = output_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys() 

        for comp in output_var_dict.keys():
            assert np.isclose(output_var_dict.get(comp), expected_var_dict.get(comp)).all()

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

    c1_array = 0.1 * np.array([0,  1,  10])
    c2_array = 0.1 * np.array([0,  2,  10]) 

    expected_dict['var1'] = {'c1': c1_array, 'c2' : c2_array}

    c3_array = 0.1 * np.array([0,  8,  10])
    c4_array = 0.1 * np.array([0,  9,  10]) 

    expected_dict['var2'] = {'c3': c3_array, 'c4' : c4_array}

    #FUNCTION CALL 
    output_dict = gu.preprocess_1d_variables(test_dict)
    
    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in output_dict.keys():
        expected_var_dict = expected_dict[var]
        output_var_dict = output_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys() 

        for comp in output_var_dict.keys():
            assert np.isclose(output_var_dict.get(comp), expected_var_dict.get(comp)).all()

def test_preprocess_2d_variables():

    #TEST DATA 
    test_dict = {} 

    component_list = ['c1','c2','c3','c4']
    space_list = ['01_reg','02_reg','03_reg']  
    var_list = ['var1', 'var2']
    
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

    c1_array = np.array([0.9, 0., 0.9])
    c3_array = np.array([0.8, 0., 0.8])

    expected_dict['var1'] = {'c1': c1_array, 'c3': c3_array}

    c2_array = np.array([0.2, 0., 0.2])
    c4_array = np.array([0.1, 0., 0.1])

    expected_dict['var2'] = {'c2': c2_array, 'c4': c4_array}

    #FUNCTION CALL 
    output_dict = gu.preprocess_2d_variables(test_dict)

    # ASSERTION 
    assert output_dict.keys() == expected_dict.keys()

    for var in output_dict.keys():
        expected_var_dict = expected_dict[var]
        output_var_dict = output_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys() 

        for comp in output_var_dict.keys():
            assert np.isclose(output_var_dict.get(comp), expected_var_dict.get(comp)).all()


def test_preprocess_dataset():
    #TEST DATA 
    #var_list = ['var_ts_1', 'var_ts_2', 'var_1d_1', 'var_1d_2', 'var_2d_1', 'var_2d_2']
    component_list = ['c1','c2','c3','c4']

    space_list = ['01_reg','02_reg','03_reg']
    time_list = ['T0','T1']

    ## time series variable data
    var_ts_1_data = np.array([  [[np.nan, np.nan, np.nan] for i in range(2)],
                            
                                 [[0, 1,  1],
                                    [1, 1,  10]],
                                
                                 [[np.nan,np.nan, np.nan] for i in range(2)],
                                
                                 [[0,   2, 2],
                                    [2, 2, 10]]  ])

    var_ts_1_DataArray = xr.DataArray(var_ts_1_data, 
                                    coords=[component_list, time_list, space_list], 
                                    dims=['component', 'time', 'space'])

    var_ts_2_data = np.array([  [[np.nan, np.nan, np.nan] for i in range(2)],
                            
                                 [[np.nan,np.nan, np.nan] for i in range(2)],
                                
                                 [[0, 8,  8],
                                    [8, 8,  10]],
                                
                                 [[0,   9, 9],
                                    [9, 9, 10]] ])

    var_ts_2_DataArray = xr.DataArray(var_ts_2_data, 
                                    coords=[component_list, time_list, space_list], 
                                    dims=['component', 'time', 'space'])
        
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
    
    xr_ds = xr.Dataset({'var_ts_1': var_ts_1_DataArray, 
                    'var_ts_2': var_ts_2_DataArray,
                    'var_1d_1': var_1d_1_DataArray, 
                    'var_1d_2': var_1d_2_DataArray, 
                    'var_2d_1': var_2d_1_DataArray,
                    'var_2d_2': var_2d_2_DataArray})

    #EXPECTED DATA
    ## time series dict
    expected_ts_dict = {}

    var_ts_1_c2_matrix = 0.1 * np.array([ [0, 1], [ 1,  1], [ 1,  10] ])
    var_ts_1_c4_matrix = 0.1 * np.array([ [ 0, 2], [ 2,  2], [ 2,  10]])

    expected_ts_dict['var_ts_1'] = {'c2': var_ts_1_c2_matrix, 
                                    'c4': var_ts_1_c4_matrix}

    var_ts_2_c3_matrix = 0.1 * np.array([ [0, 8], [ 8,  8], [ 8,  10] ]) 
    var_ts_2_c4_matrix = 0.1 * np.array([ [ 0, 9], [ 9,  9], [ 9,  10]])

    expected_ts_dict['var_ts_2'] = {'c3': var_ts_2_c3_matrix, 
                                    'c4': var_ts_2_c4_matrix}

    ## 1d dict
    expected_1d_dict = {}
    var_1d_1_c1_array = 0.1 * np.array([0,  1,  10])
    var_1d_1_c2_array = 0.1 * np.array([0,  2,  10])
    expected_1d_dict['var_1d_1'] = {'c1': var_1d_1_c1_array, 'c2': var_1d_1_c2_array}

    var_1d_2_c1_array = 0.1 * np.array([0,  8,  10])
    var_1d_2_c4_array = 0.1 * np.array([0,  9,  10])
    expected_1d_dict['var_1d_2'] = {'c1': var_1d_2_c1_array, 'c4': var_1d_2_c4_array}

    ## 2d dict 
    expected_2d_dict = {}

    var_2d_1_c1_array = np.array([0.9, 0., 0.9])
    var_2d_1_c3_array = np.array([0.8, 0., 0.8])
    expected_2d_dict['var_2d_1'] = {'c1': var_2d_1_c1_array, 'c3': var_2d_1_c3_array}
    
    var_2d_2_c3_array = np.array([0.2, 0., 0.2])
    var_2d_2_c4_array = np.array([0.1, 0., 0.1])
    expected_2d_dict['var_2d_2'] = {'c3': var_2d_2_c3_array, 'c4': var_2d_2_c4_array}
        
    #FUNCTION CALL 
    output_ts_dict, output_1d_dict, output_2d_dict = gu.preprocess_dataset(xr_ds) 
    
    # ASSERTION 
    ## ts 
    assert output_ts_dict.keys() == expected_ts_dict.keys()
    for var in output_ts_dict.keys():
        expected_var_dict = expected_ts_dict[var]
        output_var_dict = output_ts_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys()
        for comp in output_var_dict.keys():
            assert np.isclose(output_var_dict.get(comp), expected_var_dict.get(comp)).all()

    ## 1d 
    assert output_1d_dict.keys() == expected_1d_dict.keys()
    for var in output_1d_dict.keys():
        expected_var_dict = expected_1d_dict[var]
        output_var_dict = output_1d_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys()  
        for comp in output_var_dict.keys():
            assert np.isclose(output_var_dict.get(comp), expected_var_dict.get(comp)).all()

    ## 2d 
    assert output_2d_dict.keys() == expected_2d_dict.keys()
    for var in output_2d_dict.keys():
        expected_var_dict = expected_2d_dict[var]
        output_var_dict = output_2d_dict[var]

        assert output_var_dict.keys() == expected_var_dict.keys() 
        for comp in output_var_dict.keys():
            assert np.isclose(output_var_dict.get(comp), expected_var_dict.get(comp)).all()
        

@pytest.mark.parametrize("weights, expected_dist_matrix", 
                        [ 
                            # no weights are given 
                            (None, np.array([ [0, 16, 64], [16, 0, 48], [64, 48, 0] ]) ),  

                            # particular component(s), particular variable(s) 
                            ({'components' : {'wind turbine' : 2}, 'variables' : ['operationRateMax', 'capacityMax'] },
                              np.array([ [0, 19, 76], [19, 0, 51], [76, 51, 0] ]) ),   

                            # particular component(s), all variables 
                            ({'components' : {'electricity demand' : 3}, 'variables' : 'all' } ,
                              np.array([ [0, 22, 88], [22, 0, 54], [88, 54, 0] ]) ),  

                            # all components, particular variable(s)
                            ({'components' : {'all' : 2.5}, 'variables' : ['distance', 'losses'] } ,
                              np.array([ [0, 22, 88], [22, 0, 102], [88, 102, 0] ]) ),

                            # skipping 'variables' key 
                            ({'components' : {'AC cables' : 2} } ,
                              np.array([ [0, 18, 72], [18, 0, 66], [72, 66, 0] ]) )
 
                        ])
def test_get_custom_distance_matrix(weights, 
                                    expected_dist_matrix,
                                    data_for_distance_measure):
    
    test_ts_dict, test_1d_dict, test_2d_dict = data_for_distance_measure 

    #FUNCTION CALL 
    n_regions = 3
    output_dist_matrix = gu.get_custom_distance_matrix(test_ts_dict, 
                                                    test_1d_dict, 
                                                    test_2d_dict, 
                                                    n_regions,
                                                    weights)     
    
    #ASSERTION 
    assert np.isclose(expected_dist_matrix, output_dist_matrix).all()


@pytest.mark.parametrize("weights", 
                        [ 
                            # 'components' key not present
                            {'variables' : ['capacityMax', 'capacityFix'] },  
                            
                            # dictionary not adhering to the template
                            {'variable_set' : ['capacityMax', 'capacityFix'], 'component' : 'all' }

                        ])
def test_get_custom_distance_matrix_with_unusual_weights(weights, 
                                                        data_for_distance_measure):
    
    test_ts_dict, test_1d_dict, test_2d_dict = data_for_distance_measure 

    #FUNCTION CALL 
    n_regions = 3
    with pytest.raises(ValueError):
        output_dist_matrix = gu.get_custom_distance_matrix(test_ts_dict, 
                                                        test_1d_dict, 
                                                        test_2d_dict, 
                                                        n_regions,
                                                        weights)     
    
def test_get_connectivity_matrix(xr_for_connectivity):
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
    output_matrix = gu.get_connectivity_matrix(xr_for_connectivity)

    #ASSERTION 
    assert np.array_equal(output_matrix, expected_matrix)


    
    
    



