import pytest
import xarray as xr 
import geopandas
import numpy as np
import pandas as pd

import FINE.spagat.representation as spr
import FINE.spagat.grouping as spg
import FINE.spagat.dataset as spd

@pytest.mark.parametrize("string_list, expected", 
                         [(['01_es', '02_es', '01_de', '02_de', '03_de'], ['es', 'de']),
                         (['01_es', '02_es', '01_de', '02_de', '03_de', '01_nl'], ['es', 'de', 'nl']),
                         (['01_es', '02_es', '01_de', '02_de', '01_nl', '01_os'], ['es', 'de', 'nl', 'os'])]) 
def test_string_based_clustering(string_list, expected):
     clustered_regions = spg.string_based_clustering(string_list)
     assert list(clustered_regions.keys()).sort() == expected.sort()   #INFO: instead of unpermute, you can use .sort() 
            

def test_distance_based_clustering(sds):    #TODO: implement the test (hint for dataset -> makeblobs)
    #spg.distance_based_clustering(sds, mode='hierarchical', verbose=False, ax_illustration=None, save_fig=None)
    pass

@pytest.mark.skip(reason="TEST no implemented correctly")
def test_all_variable_based_clustering_hierarchical(test_dataset2):
     clustered_regions1 = spg.all_variable_based_clustering(test_dataset2,agg_mode='hierarchical2')
     assert len(clustered_regions1) == 3
     assert clustered_regions1.get(3) == {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']}
     
     dict2 = clustered_regions1.get(2)
     for sup_reg in dict2:                            #TODO: this is totally wrong. it's not asserting anything as len(sup_reg) is 13 and 6, use key, value in dict.items to iterate
          if len(sup_reg) == 2:
               assert sorted(sup_reg) == ['01_reg', '03_reg']  #TODO: assert the whole item, not just key or value
          if len(sup_reg) == 1:
               assert sorted(sup_reg) == ['02_reg']

     dict1 = clustered_regions1.get(1)
     for sup_reg in dict1:
          if len(sup_reg) == 3:
               assert sorted(sup_reg) == ['01_reg', '02_reg', '03_reg']

def test_all_variable_based_clustering_spectral(test_dataset2):
     clustered_regions1 = spg.all_variable_based_clustering(test_dataset2,agg_mode='spectral',weighting=[10,1,1])
     assert len(clustered_regions1) == 3

     dict1_2 = clustered_regions1.get(2)
     for sup_region in dict1_2.values():            #TODO: assert the whole item, not just key or value
          if len(sup_region) == 2:
               assert sorted(sup_region) ==  ['01_reg','03_reg']

     clustered_regions2 = spg.all_variable_based_clustering(test_dataset2,agg_mode='spectral',weighting=[1,1,10])
     assert len(clustered_regions2) == 3

     dict2_2 = clustered_regions2.get(2)
     for sup_region in dict2_2.values():
          if len(sup_region) == 2:
               assert sorted(sup_region) ==  ['02_reg','03_reg']

     clustered_regions3 = spg.all_variable_based_clustering(test_dataset2,agg_mode='spectral2')
     assert len(clustered_regions3) == 3

     dict3_2 = clustered_regions3.get(2)
     for sup_region in dict3_2.values():
          if len(sup_region) == 2:
               assert sorted(sup_region) ==  ['01_reg','03_reg']
