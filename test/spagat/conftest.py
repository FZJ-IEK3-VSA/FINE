import pathlib
import numpy as np
import pytest
import xarray as xr 
from shapely.geometry import Polygon
from collections import namedtuple

import FINE.spagat.dataset as spd
import FINE.spagat.representation as spr


@pytest.fixture(scope="package")
def sds():
    sds_folder_path_in = pathlib.Path("test/spagat/data/input")
    sds = spd.SpagatDataset()
    sds.read_dataset(sds_folder_path_in)
    spr.add_region_centroids(sds)

    return sds

@pytest.fixture()
def sds_and_dict_for_basic_representation():  
  '''
  sds data to test basic representation functions-
  1. test_aggregate_based_on_sub_to_sup_region_id_dict()
  2. test_aggregate_time_series()
  3. test_aggregate_values()
  4. test_aggregate_connections()
  5. test_create_grid_shapefile()
  5. test_aggregate_geometries()
  '''
  #DICT
  sub_to_sup_region_id_dict = {'01_reg_02_reg': ['01_reg','02_reg'], 
                                 '03_reg_04_reg': ['03_reg','04_reg']}
  
  #SDS
  component_list = ['c1','c2']  
  space_list = ['01_reg','02_reg','03_reg','04_reg']
  TimeStep_list = ['T0','T1']
  Period_list = [0]

  ## ts variable data
  test_ts_data = np.array([ [ [ [5, 5, 5, 5] for i in range(2)] ],

                          [ [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ]

                          ])

  test_ts_DataArray = xr.DataArray(test_ts_data, 
                                  coords=[component_list, Period_list, TimeStep_list, space_list], 
                                  dims=['component', 'Period', 'TimeStep','space'])

  ## 1d variable data
  test_1d_data = np.array([ [5,  5,  5, 5],
                            [np.nan] *4
                          ])

  test_1d_DataArray = xr.DataArray(test_1d_data, 
                              coords=[component_list, space_list], 
                              dims=['component', 'space'])

  ## 2d variable data
  test_2d_data = np.array([ [[ 0,  5,  5, 5],
                          [ 5,  0,  5, 5],
                          [ 5,  5,  0, 5],
                          [ 5,  5,  5, 0]],

                        [[np.nan] * 4 for i in range(4)] 
                        ])

  test_2d_DataArray = xr.DataArray(test_2d_data, 
                              coords=[component_list, space_list, space_list], 
                              dims=['component', 'space', 'space_2'])

  test_ds = xr.Dataset({'var_ts': test_ts_DataArray, 
                        'var_1d': test_1d_DataArray,
                        'var_2d': test_2d_DataArray})    

  sds = spd.SpagatDataset()
  sds.xr_dataset = test_ds            

  test_geometries = [Polygon([(0,0), (2,0), (2,2), (0,2)]),
                    Polygon([(2,0), (4,0), (4,2), (2,2)]),
                    Polygon([(0,0), (4,0), (4,4), (0,4)]),
                    Polygon([(0,0), (1,0), (1,1), (0,1)])]   

  sds.add_objects(description ='gpd_geometries',   #NOTE: not sure if it is ok to call another function here
                dimension_list =['space'], 
                object_list = test_geometries)   

  return namedtuple("dict_and_sds", "sub_to_sup_region_id_dict sds")(sub_to_sup_region_id_dict, sds)    



  
  









