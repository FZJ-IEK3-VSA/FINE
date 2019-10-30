import pytest
import numpy as np 
import xarray as xr 

import spagat.dataset as spd
import spagat.representation as spr
import spagat.grouping as spg


@pytest.mark.skip()
def test_add_region_centroids_and_distances():

    sds = spd.SpagatDataSet()

    # ...

    spr.add_region_centroids(sds)
    spr.add_centroid_distances(sds)


@pytest.mark.skip()
def test_aggregate_based_on_sub_to_sup_region_id_dict():
    sds = spd.SpagatDataSet()

    # ...


@pytest.mark.skip()
def test_aggregate_geometries():
    pass


#@pytest.mark.skip()
def test_aggregate_time_series():
     
     #import dataset 
     ds = xr.open_dataset('tests/data/input/sds_xr_dataset.nc4')

     #get the dictonary output by string_based_clustering function
     dict_ds = spg.string_based_clustering(ds['region_ids'].values)
    
     #A test_xr_DataArray is created with dummy values, coordinates and dimensions being region_ids and time
     region_ids = ds['region_ids'].values
     time = ds['time'].values

     temp_num = 60*8760 
     xr_DataArray_values = np.ones(temp_num)*5  #create array of 5s
     xr_DataArray_values = np.reshape(xr_DataArray_values, (60,8760))

     test_xr_DataArray = xr.DataArray(xr_DataArray_values, coords=[region_ids,time], dims=['region_ids','time'])
     
     #call the function 
     ##modes have to be changed, xr_weight_array has to be provided in case of weighted mean  
     mode = 'weighted mean'
     xr_weight_array = test_xr_DataArray   #for weighted mean
     time_series_aggregated = spr.aggregate_time_series(test_xr_DataArray, dict_ds, mode=mode, xr_weight_array=xr_weight_array)

     #correct values for mean = 5, weighted mean = 5, sum = 25  for loc[4,'de']
     assert time_series_aggregated.loc[4,'de'].values == 5   


@pytest.mark.skip()
def test_create_grid_shapefile():
    pass
