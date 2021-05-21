import os
import pytest

import numpy as np
import xarray as xr
from sklearn.datasets import make_blobs
from shapely.geometry import Point, Polygon

import FINE.spagat.dataset as spd
import FINE.spagat.grouping as spg
import FINE.spagat.representation as spr

path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output/')  
file_name = 'test_fig'
expected_file = os.path.join(path_to_test_dir, f'{file_name}.png')

@pytest.mark.parametrize("string_list, expected", 
                         [(['01_es', '02_es', '01_de', '02_de', '03_de'], ['es', 'de']),
                         (['01_es', '02_es', '01_de', '02_de', '03_de', '01_nl'], ['es', 'de', 'nl']),
                         (['01_es', '02_es', '01_de', '02_de', '01_nl', '01_os'], ['es', 'de', 'nl', 'os'])]) 
def test_perform_string_based_grouping(string_list, expected):
     clustered_regions_dict = spg.perform_string_based_grouping(string_list)
     assert list(clustered_regions_dict.keys()).sort() == expected.sort()   
      #TODO: check values also       

@pytest.mark.parametrize("mode", ['sklearn_kmeans', 'sklearn_hierarchical', 'sklearn_spectral', 'scipy_kmeans', 'scipy_hierarchical'])
def test_perform_distance_based_grouping(mode):    
     #TEST DATA
     component_list = ['c1','c2']  
     space_list = ['01_reg','02_reg','03_reg','04_reg','05_reg']
     TimeStep_list = ['T0','T1']
     Period_list = [0]

     dummy_data = np.array([[ [[np.nan for i in range(5)] for i in range(2)] ],
                              [ [[np.nan for i in range(5)] for i in range(2)] ]
                           ])

     dummy_DataArray = xr.DataArray(dummy_data, 
                                   coords=[component_list, Period_list, TimeStep_list, space_list], 
                                   dims=['component', 'Period', 'TimeStep','space'])    

     dummy_ds = xr.Dataset({'var': dummy_DataArray}) 

     sds = spd.SpagatDataset()
     sds.xr_dataset = dummy_ds       

     sample_data, sample_labels = make_blobs(n_samples=5, centers=3, n_features=2, random_state=0)
     
     test_centroids = [np.nan for i in range(5)]
     for i, data_point in enumerate(sample_data):
          test_centroids[i] = Point(data_point)
     
     sds.add_objects(description ='gpd_centroids',   
                dimension_list =['space'], 
                object_list = test_centroids)

     
     #FUNCTION CALL 
     output_dict = spg.perform_distance_based_grouping(sds, 
                                                       agg_mode = mode, 
                                                       save_path = path_to_test_dir, 
                                                       fig_name=file_name)  
     

     #ASSERTION 
     ## Results for number of aggregated regions = 3 can be checked, because test data has 3 centers 
     #NOTE: 1 and 5 can also be tested but permutation is making it difficult to test these
     assert output_dict.get(3) == {'01_reg': ['01_reg'],                    ## Based on sample_labels ([2, 0, 0, 1, 1])       
                              '02_reg_03_reg': ['02_reg', '03_reg'],
                              '04_reg_05_reg': ['04_reg', '05_reg']}  
     
     if mode is not 'sklearn_spectral':
          assert os.path.isfile(expected_file) 
          os.remove(expected_file)
     

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
def test_perform_parameter_based_grouping(weights,
                                        expected_region_groups,
                                        sds_for_parameter_based_grouping): 
     
     regions_list = sds_for_parameter_based_grouping.xr_dataset.space.values 

     #FUNCTION CALL
     output_dict = spg.perform_parameter_based_grouping(sds_for_parameter_based_grouping, 
                                                       dimension_description='space',
                                                       weights=weights)  
     
     #ASSERTION
     for key, value in output_dict.get(2).items():   
          if len(value)==2:                            #NOTE: required to assert separately, because they are permuted
               assert (key == '_'.join(expected_region_groups)) & (value == expected_region_groups)
          else:
               remaining_region = np.setdiff1d(regions_list, expected_region_groups)
               assert (key == remaining_region.item()) & (value == remaining_region)
