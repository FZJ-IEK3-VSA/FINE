import os
import pytest

import numpy as np
import xarray as xr
from sklearn.datasets import make_blobs
from shapely.geometry import Point

import FINE.spagat.grouping as spg
import FINE.spagat.utils as spu


@pytest.mark.parametrize("string_list, expected, separator, position", 
                         [(['01_es', '02_es', '01_de', '02_de', '01_nl', '01_os'], ['es', 'de', 'nl', 'os'], '_', None),
                         (['abc123', 'abc456', 'def123', 'def456'], ['abc', 'def'], None, 3),
                         (['123abc456', '456abc345', '456def123', '897def456'], ['abc', 'def'], None, (3,6))
                         ]) 
def test_perform_string_based_grouping(string_list, expected, separator, position):
     clustered_regions_dict = spg.perform_string_based_grouping(string_list, separator=separator, position=position)
     assert sorted(clustered_regions_dict.keys()) == sorted(expected)    

def test_perform_distance_based_grouping():    
     #TEST DATA
     component_list = ['c1','c2']  
     space_list = ['01_reg','02_reg','03_reg','04_reg','05_reg']
     time_list = ['T0','T1']

     dummy_data = np.array([ [[np.nan for i in range(5)] for i in range(2)] ,
                             [[np.nan for i in range(5)] for i in range(2)] 
                           ])

     dummy_DataArray = xr.DataArray(dummy_data, 
                                   coords=[component_list, time_list, space_list], 
                                   dims=['component', 'time','space'])    

     dummy_ds = xr.Dataset({'var': dummy_DataArray})   

     sample_data, sample_labels = make_blobs(n_samples=5, centers=3, n_features=2, random_state=0)
     
     test_centroids = [np.nan for i in range(5)]
     for i, data_point in enumerate(sample_data):
          test_centroids[i] = Point(data_point)
     
     dummy_ds = spu.add_objects_to_xarray(dummy_ds, 
                                        description ='gpd_centroids',   
                                        dimension_list =['space'], 
                                        object_list = test_centroids)

     
     #FUNCTION CALL 
     output_dict = spg.perform_distance_based_grouping(dummy_ds)  
     

     #ASSERTION 
     assert output_dict == {'01_reg': ['01_reg'],                    ## Based on sample_labels ([2, 0, 0, 1, 1])       
                              '02_reg_03_reg': ['02_reg', '03_reg'],
                              '04_reg_05_reg': ['04_reg', '05_reg']}  
     

     
     
@pytest.mark.parametrize("aggregation_method", ['kmedoids_contiguity', 'hierarchical'])
@pytest.mark.parametrize("weights, expected_region_groups", 
                        [ 
                             # no weights 
                             (None, ['02_reg', '03_reg'] ),

                              # particular components, particular variables
                         ({'components' : {'AC cables' : 5,  'PV' : 10}, 'variables' : ['capacityMax'] }, 
                              ['01_reg', '02_reg'] ),

                              # particular component, all variables 
                         ({'components' : {'AC cables' : 10}, 'variables' : 'all' }, 
                              ['01_reg', '02_reg'] )
                        ])
def test_perform_parameter_based_grouping(aggregation_method,
                                        weights,
                                        expected_region_groups,
                                        xr_for_parameter_based_grouping): 


     regions_list = xr_for_parameter_based_grouping.space.values 

     #FUNCTION CALL
     output_dict = spg.perform_parameter_based_grouping(xr_for_parameter_based_grouping, 
                                                       n_groups = 2, 
                                                       aggregation_method = aggregation_method, 
                                                       weights=weights,
                                                       solver="glpk")  
     
     #ASSERTION
     for key, value in output_dict.items():   
          if len(value)==2:                            #NOTE: required to assert separately, because they are permuted
               assert (key == '_'.join(expected_region_groups)) & (value == expected_region_groups)
          else:
               remaining_region = np.setdiff1d(regions_list, expected_region_groups)
               assert (key == remaining_region.item()) & (value == remaining_region)
