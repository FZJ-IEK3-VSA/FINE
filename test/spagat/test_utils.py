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
    #TEST PLOT 
    x = [1, 2, 3, 4]
    plt.plot(x, x)
    
    spu.plt_savefig()  #NOTE: only testing for default arguments 

    assert os.path.isfile('test.png') 
    
    #Delete test plot
    os.remove('test.png')




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
    path_to_test_dir = os.path.join(os.path.dirname(__file__), 'data/output/test_dir')  #TODO: maybe create directly in output 
    files_name = 'test_file'

    #FUNCTION CALL 
    spu.create_gdf(df, geometries, crs, file_path=path_to_test_dir, files_name = files_name)
    
    #EXPECTED 
    ## File extensions 
    file_extensions_list = ['.cpg', '.dbf', '.prj', '.shp', '.shx']

    #ASSERTION
    for file_extension in file_extensions_list:
        expected_file_path = os.path.join(path_to_test_dir, f'{files_name}{file_extension}')
        assert os.path.isfile(expected_file_path)

    #Delete test_dir 
    shutil.rmtree(path_to_test_dir)   #INFO: os.rmdir() does not delete folder that is not empty 
   
