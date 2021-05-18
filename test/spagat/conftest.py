import pathlib
import pytest

import numpy as np
import pandas as pd
import geopandas as gpd 
import xarray as xr 
from shapely.geometry import Polygon, MultiPolygon
from collections import namedtuple

import FINE.spagat.dataset as spd
import FINE.spagat.representation as spr

#============================================Fixtures for Grouping==================================================#


@pytest.fixture()
def sds_for_connectivity():  

  component_list = ['source_comp','sink_comp', 'transmission_comp']  
  space_list = ['01_reg','02_reg','03_reg','04_reg', '05_reg','06_reg', '07_reg','08_reg']
  TimeStep_list = ['T0','T1']
  Period_list = [0]

  ## ts variable data
  operationRateMax = np.array([ [ [[1] * 8 for i in range(2)] ], 

                                [ [[np.nan] * 8 for i in range(2)] ], 

                                [ [[np.nan] * 8 for i in range(2)] ] 

                          ])

  operationRateMax_da = xr.DataArray(operationRateMax, 
                                  coords=[component_list, Period_list, TimeStep_list, space_list], 
                                  dims=['component', 'Period', 'TimeStep','space'])


  ## 1d variable data
  capacityMax_1d = np.array([ [14] * 8, 
                            [np.nan] *8, 
                            [np.nan] *8
                          ])

  capacityMax_1d_da = xr.DataArray(capacityMax_1d, 
                              coords=[component_list, space_list], 
                              dims=['component', 'space'])


  ## 2d variable data
  capacityMax_2d = np.array([ [[np.nan] * 8 for i in range(8)], 

                        [[np.nan] * 8 for i in range(8)],

                         [[ 0, 0, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 0],
                          [ 0, 0, 0, 0, 0, 0, 0, 3],
                          [ 0, 0, 0, 0, 0, 0, 0, 5],
                          [ 0, 0, 0, 0, 0, 3, 5, 0]]
                        ])

  capacityMax_2d_da = xr.DataArray(capacityMax_2d, 
                              coords=[component_list, space_list, space_list], 
                              dims=['component', 'space', 'space_2'])
  
  locationalEligibility_2d = np.array([ [[np.nan] * 8 for i in range(8)], 

                                        [[np.nan] * 8 for i in range(8)],

                                        [[ 0, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0, 0, 0.2, 0],
                                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0.2, 0, 0, 0, 0],
                                        [ 0, 0, 0, 0, 0, 0, 0, 0]]
                                        ])

  locationalEligibility_2d_da = xr.DataArray(locationalEligibility_2d, 
                              coords=[component_list, space_list, space_list], 
                              dims=['component', 'space', 'space_2'])
  
  test_ds = xr.Dataset({'ts_operationRateMax': operationRateMax_da, 
                        '1d_capacityMax': capacityMax_1d_da, 
                        '2d_capacityMax': capacityMax_2d_da,
                        '2d_locationalEligibility': locationalEligibility_2d_da
                        })    

  sds = spd.SpagatDataset()
  sds.xr_dataset = test_ds 

  #Geometries 
  test_geometries = [Polygon([(0,3), (1,3), (1,4), (0,4)]),
                  Polygon([(1,3), (2,3), (2,4), (4,1)]),
                  Polygon([(2,3), (3,3), (3,4), (2,4)]),
                  Polygon([(0,2), (1,2), (1,3), (0,3)]),
                  Polygon([(1,2), (2,2), (2,3), (1,3)]),
                  Polygon([(2,2), (3,2), (3,3), (2,3)]),
                  Polygon([(1,0), (2,0), (2,1), (1,1)]),
                  Polygon([(2.5,0), (3.5,0), (3.5,1), (2.5,1)])] 
                

  sds.add_objects(description ='gpd_geometries',   
                dimension_list =['space'], 
                object_list = test_geometries)   
  
  spr.add_region_centroids(sds) 
  spr.add_centroid_distances(sds)

  return sds  

@pytest.fixture()
def data_for_distance_measure():  

  test_ts_dict = {}            

  var_ts_1_c2_matrix = np.array([ [1, 1],
                                  [2, 2],
                                  [3, 3] ])

  var_ts_1_c4_matrix = np.array([ [1, 1],
                                  [2, 2],
                                  [3, 3]])

  test_ts_dict['ts_operationRateMax'] = np.concatenate((var_ts_1_c2_matrix, var_ts_1_c4_matrix), axis=1)

  var_ts_2_c3_matrix = np.array([ [1, 1],
                                  [2, 2],
                                  [3, 3] ])

  var_ts_2_c4_matrix = np.array([ [1, 1],
                                  [2, 2],
                                  [3, 3]])

  test_ts_dict['ts_operationRateFix'] = np.concatenate((var_ts_2_c3_matrix, var_ts_2_c4_matrix), axis=1)

  ## 1d dict
  test_1d_dict = {}

  test_1d_dict['1d_capacityMax'] = np.array([ [1, 1],
                                      [2, 2],
                                      [3, 3]])

  test_1d_dict['1d_capacityFix'] = np.array([ [1, 1],
                                      [2, 2],
                                      [3, 3]])

  ## 2d dict 
  test_2d_dict = {}

  var_2d_1_c1_array = np.array([1, 2, 3])
  var_2d_1_c3_array = np.array([1, 2, 3])
  test_2d_dict['2d_distance'] = {0: var_2d_1_c1_array, 2: var_2d_1_c3_array}

  var_2d_2_c3_array = np.array([1, 2, 3])
  var_2d_2_c4_array = np.array([1, 2, 3])
  test_2d_dict['2d_losses'] = {2: var_2d_2_c3_array, 3: var_2d_2_c4_array}    

  return namedtuple("test_ts_1d_2s_dicts", "test_ts_dict test_1d_dict test_2d_dict")(test_ts_dict, test_1d_dict, test_2d_dict)  


@pytest.fixture()
def sds_for_parameter_based_grouping(): 

  component_list = ['c1','c2', 'c3']  
  space_list = ['01_reg','02_reg','03_reg']
  TimeStep_list = ['T0','T1']
  Period_list = [0]

  ## time series variables data
  operationRateMax = np.array([ [[[0.2, 0.1, 0.1] for i in range(2)]],
                                [[[np.nan]*3 for i in range(2)]], 
                                [[[0.2, 0.1, 0.1] for i in range(2)]]  ])

  operationRateMax = xr.DataArray(operationRateMax, 
                                coords=[component_list, Period_list, TimeStep_list, space_list], 
                                dims=['component', 'Period', 'TimeStep','space'])
  
  
  ## 1d variable data
  capacityMax = np.array([ [1, 1, 0.2],
                          [1, 1, 0.2],
                          [1, 1, 0.2] ])

  capacityMax = xr.DataArray(capacityMax, 
                            coords=[component_list, space_list], 
                            dims=['component', 'space'])
  
  ## 2d variable data
  transmissionDistance = np.array([ [[0, 0.2, 0.7], 
                                    [0.2, 0, 0.2], 
                                    [0.7, 0.2, 0]],
                                  [[0, 0.2, 0.7], 
                                    [0.2, 0, 0.2], 
                                    [0.7, 0.2, 0]],
                      [[np.nan]*3 for i in range(3)]])

  transmissionDistance = xr.DataArray(transmissionDistance, 
                                    coords=[component_list, space_list, space_list], 
                                    dims=['component', 'space', 'space_2'])
  
  ds = xr.Dataset({'ts_operationRateMax': operationRateMax,
                '1d_capacityMax': capacityMax,  
                '2d_transmissionDistance': transmissionDistance}) 

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

  return sds 
#============================================Fixtures for Basic Representation==================================================#

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
  component_list = ['source_comp','sink_comp', 'transmission_comp']  
  space_list = ['01_reg','02_reg','03_reg','04_reg']
  TimeStep_list = ['T0','T1']
  Period_list = [0]

  ## ts variable data
  operationRateMax = np.array([ [ [ [3, 3, 3, 3] for i in range(2)] ],

                          [ [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ], 

                          [ [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ]

                          ])

  operationRateMax_da = xr.DataArray(operationRateMax, 
                                  coords=[component_list, Period_list, TimeStep_list, space_list], 
                                  dims=['component', 'Period', 'TimeStep','space'])

  operationRateFix = np.array([ [ [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ], 

                          [ [ [5, 5, 5, 5] for i in range(2)] ],

                          [ [[np.nan, np.nan, np.nan, np.nan] for i in range(2)] ]

                          ])

  operationRateFix_da = xr.DataArray(operationRateFix, 
                                  coords=[component_list, Period_list, TimeStep_list, space_list], 
                                  dims=['component', 'Period', 'TimeStep','space'])

  ## 1d variable data
  capacityMax_1d = np.array([ [15,  15,  15, 15],
                            [np.nan] *4, 
                            [np.nan] *4, 
                          ])

  capacityMax_1d_da = xr.DataArray(capacityMax_1d, 
                              coords=[component_list, space_list], 
                              dims=['component', 'space'])

  capacityFix_1d = np.array([ [np.nan] *4, 
                           [5,  5,  5, 5],
                            [np.nan] *4, 
                          ])

  capacityFix_1d_da = xr.DataArray(capacityFix_1d, 
                              coords=[component_list, space_list], 
                              dims=['component', 'space'])

  ## 2d variable data
  capacityMax_2d = np.array([ [[np.nan] * 4 for i in range(4)], 

                        [[np.nan] * 4 for i in range(4)],

                         [[ 0,  5,  5, 5],
                          [ 5,  0,  5, 5],
                          [ 5,  5,  0, 5],
                          [ 5,  5,  5, 0]]
                        ])

  capacityMax_2d_da = xr.DataArray(capacityMax_2d, 
                              coords=[component_list, space_list, space_list], 
                              dims=['component', 'space', 'space_2'])
  
  locationalEligibility_2d = np.array([ [[np.nan] * 4 for i in range(4)], 

                        [[np.nan] * 4 for i in range(4)],

                         [[ 0,  1,  1, 1],
                          [ 0,  0,  1, 1],
                          [ 0,  0,  0, 1],
                          [ 0,  0,  0, 0]]
                        ])

  locationalEligibility_2d_da = xr.DataArray(locationalEligibility_2d, 
                              coords=[component_list, space_list, space_list], 
                              dims=['component', 'space', 'space_2'])

  test_ds = xr.Dataset({'ts_operationRateMax': operationRateMax_da, 
                        'ts_operationRateFix': operationRateFix_da,
                        '1d_capacityMax': capacityMax_1d_da, 
                        '1d_capacityFix': capacityFix_1d_da, 
                        '2d_capacityMax': capacityMax_2d_da, 
                        '2d_locationalEligibility': locationalEligibility_2d_da
                        })    

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

#============================================Fixtures for RE Representation==================================================#

@pytest.fixture
def gridded_RE_data(scope="session"):
  time_steps = 10
  x_coordinates = 5
  y_coordinates = 3

  time = np.arange(time_steps)
  x_locations = [1, 2, 3, 4, 5]
  y_locations = [1, 2, 3]

  #capacity factor time series 
  capfac_xr_da = xr.DataArray(coords=[x_locations, y_locations, time], 
                              dims=['x', 'y','time'])

  capfac_xr_da.loc[[1, 2, 5], :, :] = [np.full((3, 10), 1) for x in range(3)]
  capfac_xr_da.loc[3:4, :, :] = [np.full((3, 10), 2) for x in range(2)]

  #capacities
  test_data = np.ones((x_coordinates, y_coordinates))
  capacity_xr_da = xr.DataArray(test_data, 
                              coords=[x_locations, y_locations], 
                              dims=['x', 'y'])

  test_xr_ds = xr.Dataset({'capacity': capacity_xr_da,
                          'capfac': capfac_xr_da}) 

  test_xr_ds.attrs['SRS'] = 'epsg:3035'

  return test_xr_ds


@pytest.fixture
def sample_shapefile(scope="session"):
  polygon1 = Polygon([(0,0), (4,0), (4,4), (0,4)])
  polygon2 = Polygon([(4,0), (7,0), (7,4), (4,4)])

  test_geometries = [MultiPolygon([polygon1]),
                  MultiPolygon([polygon2])] 

  df = pd.DataFrame({'region_ids': ['reg_01', 'reg_02']})

  gdf = gpd.GeoDataFrame(df, geometry=test_geometries, crs='epsg:3035') 

  return gdf







