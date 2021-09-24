import os
import shutil

import pandas as pd
import numpy as np 
import xarray as xr 
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

import FINE.spagat.utils as spu


def test_plt_savefig():
    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output/')  
    #TEST PLOT 
    x = [1, 2, 3, 4]
    plt.plot(x, x)
    
    #FUNCTION CALL 
    spu.plt_savefig(path = path_to_test_dir)  

    #ASSERTION
    expected_file = os.path.join(path_to_test_dir, 'test.png')
    assert os.path.isfile(expected_file) 
    
    #Delete test plot
    os.remove(expected_file)


def test_create_gdf():
    #TEST DATA 
    geometries = [Point(1,2), Point(2,1)]
    df = pd.DataFrame({'space': ['reg_01', 'reg_02']})

    crs=3035
    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output/test_dir')   
    file_name = 'test_file'

    #FUNCTION CALL 
    spu.create_gdf(df, geometries, crs, file_path=path_to_test_dir, files_name = file_name)
    
    #EXPECTED 
    output_shp = gpd.read_file(os.path.join(path_to_test_dir, f'{file_name}.shp'))
    assert list(output_shp.columns) == ['space', 'geometry']

    #Delete test_dir 
    shutil.rmtree(path_to_test_dir)  


def test_add_objects_and_space_coords_to_xarray():

    xarray_dataset = xr.Dataset()

    #TEST add_objects_to_xarray()
    data = [1, 2, 3]
    xarray_dataset = spu.add_objects_to_xarray(xarray_dataset,
                                            description ='data_var', 
                                            dimension_list =['space'], 
                                            object_list = data)

    assert np.array_equal(xarray_dataset['data_var'].values, np.array(data))                 
    
    #TEST add_space_coords_to_xarray()          
    region_id_list = ['reg_0', 'reg_1', 'reg_2']
    xarray_dataset = spu.add_space_coords_to_xarray(xarray_dataset, 
                                                    region_id_list)

    assert list(xarray_dataset.coords['space'].values) == region_id_list
    assert list(xarray_dataset.coords['space_2'].values) == region_id_list  


def test_add_region_centroids_and_distances_to_xarray():
    test_geometries = [Polygon([(0,0), (2,0), (2,2), (0,2)]),
                       Polygon([(2,0), (4,0), (4,2), (2,2)])] 

    expected_centroids = [Point(1, 1), Point(3,1)]
    expected_centroid_distances = .001 * np.array([ [0, 2], [2, 0] ]) #Distances in km

    #FUNCTION CALL                                         
    xarray_dataset = xr.Dataset()
    xarray_dataset = spu.add_objects_to_xarray(xarray_dataset, 
                                            description ='gpd_geometries',  
                                            dimension_list =['space'], 
                                            object_list = test_geometries)
    
    xarray_dataset = spu.add_region_centroids_to_xarray(xarray_dataset) 
    xarray_dataset = spu.add_centroid_distances_to_xarray(xarray_dataset)

    output_centroids = xarray_dataset['gpd_centroids'].values
    output_centroid_distances = xarray_dataset['centroid_distances'].values

    #ASSERTION 
    for output, expected in zip(output_centroids, expected_centroids):
        assert output.coords[:] == expected.coords[:]
    
    assert np.array_equal(output_centroid_distances, expected_centroid_distances)


def test_create_grid_shapefile(xr_and_dict_for_basic_representation):
    test_xr = xr_and_dict_for_basic_representation.test_xr

    
    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output')
    files_name="test_ac_lines"

    spu.create_grid_shapefile(test_xr,
                            variable_description='2d_capacityMax',
                            component_description='source_comp',
                            file_path=path_to_test_dir,
                            files_name=files_name)
    
    #EXPECTED 
    ## File extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    #ASSERTION
    for file_extension in file_extensions_list:
        expected_file = os.path.join(path_to_test_dir, f'{files_name}{file_extension}')
        assert os.path.isfile(expected_file)

        os.remove(expected_file)