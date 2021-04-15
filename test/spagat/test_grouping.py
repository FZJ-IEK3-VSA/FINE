import os
import pytest

import numpy as np
import xarray as xr
from shapely.geometry import Point, Polygon
from sklearn.datasets import make_blobs

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
def test_string_based_clustering(string_list, expected):
     clustered_regions_dict = spg.string_based_clustering(string_list)
     assert list(clustered_regions_dict.keys()).sort() == expected.sort()   
      #TODO: check values also       

@pytest.mark.parametrize("mode", ['sklearn_kmeans', 'sklearn_hierarchical', 'sklearn_spectral', 'scipy_kmeans', 'scipy_hierarchical'])
def test_distance_based_clustering(mode):    
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
     output_dict = spg.distance_based_clustering(sds, agg_mode = mode, save_path = path_to_test_dir, fig_name=file_name)  
     

     #ASSERTION 
     ## Results for number of aggregated regions = 3 can be checked, because test data has 3 centers 
     #NOTE: 1 and 5 can also be tested but permutation is making it difficult to test these
     assert output_dict.get(3) == {'01_reg': ['01_reg'],                    ## Based on sample_labels ([2, 0, 0, 1, 1])       
                              '02_reg_03_reg': ['02_reg', '03_reg'],
                              '04_reg_05_reg': ['04_reg', '05_reg']}  
     
     if mode is not 'sklearn_spectral':
          assert os.path.isfile(expected_file) 
          os.remove(expected_file)
     


def test_parameter_based_clustering(): #TODO: check what happens when you restrict connections. Hint: take more regions and make a complex structure 
     #TEST DATA     
     component_list = ['c1','c2', 'c3']  
     space_list = ['01_reg','02_reg','03_reg']
     TimeStep_list = ['T0','T1']
     Period_list = [0]

     ## time series variable data
     sample_ts_data, sample_ts_labels = make_blobs(n_samples=3, centers=2, n_features=2, random_state=0)
     var_ts_data = np.array([ [sample_ts_data.T],
                              [sample_ts_data.T], 
                              [[[np.nan]*3 for i in range(2)]]  ])

     var_ts_DataArray = xr.DataArray(var_ts_data, 
                                   coords=[component_list, Period_list, TimeStep_list, space_list], 
                                   dims=['component', 'Period', 'TimeStep','space'])
     
     
     ## 1d variable data
     var_1d_data = np.array([ [1, 1, 2],
                              [1, 1, 2],
                              [np.nan]*3 ])

     var_1d_DataArray = xr.DataArray(var_1d_data, 
                                        coords=[component_list, space_list], 
                                        dims=['component', 'space'])
     
     ## 2d variable data
     var_2d_data = np.array([ [[0, 2, 1], 
                              [2, 0, 1], 
                              [1, 1, 0]],
                              [[0, 2, 1], 
                              [2, 0, 1], 
                              [1, 1, 0]],
                         [[np.nan]*3 for i in range(3)]])

     var_2d_DataArray = xr.DataArray(var_2d_data, 
                                        coords=[component_list, space_list, space_list], 
                                        dims=['component', 'space', 'space_2'])
     
     ds = xr.Dataset({'ts_var': var_ts_DataArray,
                    '1d_var': var_1d_DataArray,  
                    '2d_var': var_2d_DataArray}) 

     sds = spd.SpagatDataset()
     sds.xr_dataset = ds

     #Geometries 
     test_geometries = [Polygon([(0,3), (1,3), (1,4), (0,4)]),
                         Polygon([(1,3), (2,3), (2,4), (4,1)]),
                         Polygon([(0,2), (1,2), (1,3), (0,3)]) ] 
                    

     sds.add_objects(description ='gpd_geometries',   
                    dimension_list =['space'], 
                    object_list = test_geometries)   
     
     spr.add_region_centroids(sds) 
     spr.add_centroid_distances(sds)
     
     #FUNCTION CALL
     output_dict = spg.parameter_based_clustering(sds,
                                                  dimension_description='space',
                                                  ax_illustration=None, 
                                                  save_path=path_to_test_dir, 
                                                  fig_name=file_name,  
                                                  verbose=False)  
     
     #ASSERTION
     for key, value in output_dict.get(2).items():     #sample labels array([0, 0, 1])
          if len(value)==2:                            #NOTE: required to assert separately, because they are permuted
               assert (key == '01_reg_02_reg') & (value == ['01_reg', '02_reg'])
          else:
               assert (key == '03_reg') & (value == ['03_reg'] )
