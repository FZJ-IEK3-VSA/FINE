import pytest
import xarray as xr 
import geopandas

import spagat.representation as spr
import spagat.grouping as spg
import spagat.dataset as spd

testdata = [(['01_es', '02_es', '01_de', '02_de', '03_de'], ['es', 'de']),
            (['01_es', '02_es', '01_de', '02_de', '03_de', '01_nl'], ['es', 'de', 'nl']),
            (['01_es', '02_es', '01_de', '02_de', '01_nl', '01_os'], ['es', 'de', 'nl', 'os'])]
@pytest.mark.parametrize("string_list, expected", testdata) 
def test_string_based_clustering(string_list, expected):
     clustered_regions = spg.string_based_clustering(string_list)
     assert list(clustered_regions.keys()).sort() == expected.sort()
            

def test_distance_based_clustering(sds):
    '''Test whether the distance-based clustering works'''
    spg.distance_based_clustering(sds, mode='hierarchical', verbose=False, ax_illustration=None, save_fig=None)
