import pytest 
import os
from pathlib import Path
import shutil

import pandas as pd
import numpy as np
from shapely.geometry import Point
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


@pytest.mark.skip(reason="Not implemented.")
def test_timer():
    pass #NOTE: does it make sense to implement this ? 


def test_create_dir():

    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output/test_dir/')

    #Test if it can create the folder
    spu.create_dir(path_to_test_dir)
    assert os.path.isdir(path_to_test_dir)
    
    #Test if it can skip if the folder is already present 
    spu.create_dir(path_to_test_dir)
    assert os.path.isdir(path_to_test_dir)

    #Delete test_dir 
    os.rmdir(path_to_test_dir)


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
    ## File extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    #ASSERTION
    for file_extension in file_extensions_list:
        expected_file_path = os.path.join(path_to_test_dir, f'{file_name}.shp')
        assert os.path.isfile(expected_file_path)

    #Delete test_dir 
    shutil.rmtree(path_to_test_dir)   #INFO: os.rmdir() does not delete folder that is not empty 
   
