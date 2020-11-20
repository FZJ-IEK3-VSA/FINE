import os
import pytest

import numpy as np

import pathlib
import FINE.spagat.dataset as spd



def test_add_objects_and_add_region_data():

    sds = spd.SpagatDataset()

    #TEST add_objects()
    data = [1, 2, 3]
    sds.add_objects(description ='data_var', 
                    dimension_list =['space'], 
                    object_list = data)

    assert np.array_equal(sds.xr_dataset['data_var'].values, np.array(data))                 
    
    #TEST add_region_data()          
    region_id_list = ['reg_0', 'reg_1', 'reg_2']
    sds.add_region_data(region_id_list, spatial_dim = 'space')

    assert list(sds.xr_dataset.coords['space'].values) == region_id_list
    assert list(sds.xr_dataset.coords['space_2'].values) == region_id_list
    

def test_read_and_save_sds_and_shapefiles():
    test_folder_path_in =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","input")
    test_folder_path_out =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","output")

    sds = spd.SpagatDataset()

    sds.read_dataset(test_folder_path_in, 
                    sds_regions_filename= "sds_regions.shp",
                    sds_xr_dataset_filename= "sds_xr_dataset.nc4")   
    
    #1. TEST save_sds_regions()
    sds.save_sds_regions(test_folder_path_out, shape_output_files_name = 'sds_regions') 
    
    ## Expected file extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    ## Assertion
    for file_extension in file_extensions_list:
        expected_file_path = os.path.join(test_folder_path_out, f'sds_regions{file_extension}')
        assert os.path.isfile(expected_file_path)

        os.remove(expected_file_path)
    
    #TODO: - check why permission is denied
    # #2. TEST save_data()  
    # sds.save_data(test_folder_path_out)
    # expected_file_path = os.path.join(test_folder_path_out, 'sds_xr_dataset.nc4')
    # assert os.path.isfile(expected_file_path)

    # os.remove(expected_file_path)

    #3. TEST save_sds()
    # sds.save_sds(test_folder_path_out) 

    # ## Expected file extensions 
    # file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    # ## Assertion
    # for file_extension in file_extensions_list:
    #     expected_file_path = os.path.join(test_folder_path_out, f'sds_regions{file_extension}')
    #     assert os.path.isfile(expected_file_path)

    #     os.remove(expected_file_path)
    
    # # #TEST save_data()
    # sds.save_data(test_folder_path_out)
    # expected_file_path = os.path.join(test_folder_path_out, 'sds_xr_dataset.nc4')
    # assert os.path.isfile(expected_file_path)

    # os.remove(expected_file_path)




