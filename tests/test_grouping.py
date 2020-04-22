import pytest
import xarray as xr 
import geopandas
import numpy as np
import pandas as pd

import spagat.representation as spr
import spagat.grouping as spg
import spagat.dataset as spd

@pytest.mark.parametrize("string_list, expected", 
                         [(['01_es', '02_es', '01_de', '02_de', '03_de'], ['es', 'de']),
                         (['01_es', '02_es', '01_de', '02_de', '03_de', '01_nl'], ['es', 'de', 'nl']),
                         (['01_es', '02_es', '01_de', '02_de', '01_nl', '01_os'], ['es', 'de', 'nl', 'os'])]) 
def test_string_based_clustering(string_list, expected):
     clustered_regions = spg.string_based_clustering(string_list)
     assert list(clustered_regions.keys()).sort() == expected.sort()
            
'''
def test_distance_based_clustering(sds):
    spg.distance_based_clustering(sds, mode='hierarchical', verbose=False, ax_illustration=None, save_fig=None)
    assert 1
'''

# Create a simple Test Xarray Dataset containing three variables (without component): opFix(time series var), 1d_cap, 2d_dist
@pytest.fixture()
def test_dataset1():
     space = ['01_reg','02_reg','03_reg']
     timestep = ['T0','T1']
     space_2 = space.copy()

     opFix = xr.DataArray(np.array([[1,1],
                                    [0.9,1],
                                    [2,2]]), coords=[space, timestep], dims=['space', 'TimeStep'])
     cap_1d = xr.DataArray(np.array([0.9,
                                       1,
                                       0.9]), coords=[space], dims=['space'])
     dist_2d = xr.DataArray(np.array([[0,1,2],
                                      [1,0,10],
                                      [2,10,0]]), coords=[space,space_2], dims=['space','space_2'])

     ds = xr.Dataset({'operationFixRate': opFix,'1d_capacity': cap_1d,'2d_distance': dist_2d})

     sds = spd.SpagatDataSet()
     sds.xr_dataset = ds
     return sds

def test_all_variable_based_clustering_hierarchical(test_dataset1):
     clustered_regions1 = spg.all_variable_based_clustering(test_dataset1,agg_mode='hierarchical')
     assert len(clustered_regions1) == 3
     assert clustered_regions1.get(3) == {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']}
     assert clustered_regions1.get(2) == {'03_reg': ['03_reg'], '01_reg_02_reg': ['01_reg', '02_reg']}
     assert clustered_regions1.get(1) == {'03_reg_01_reg_02_reg': ['03_reg', '01_reg', '02_reg']}

     clustered_regions2 = spg.all_variable_based_clustering(test_dataset1,agg_mode='hierarchical2')
     assert len(clustered_regions2) == 3
     assert clustered_regions2.get(3) == {'01_reg': ['01_reg'], '02_reg': ['02_reg'], '03_reg': ['03_reg']}
     assert clustered_regions2.get(2) == {'01_reg_02_reg': ['01_reg', '02_reg'], '03_reg': ['03_reg']}
     assert clustered_regions2.get(1) == {'01_reg_02_reg_03_reg': ['01_reg', '02_reg', '03_reg']}


