import pytest
import xarray as xr 
import geopandas
import geoplot 

import spagat.representation as spr
import spagat.grouping as spg
import spagat.dataset as spd

# read e-highway shapefile
# TODO: replace with relative paths to tests/data directory
path_to_ehighwayfile = "/home/s-patil/data/e-highway"
ehighway_shapefile = geopandas.read_file(path_to_ehighwayfile, layer='e-highway')

def test_string_based_clustering():
     clustered_regions = spg.string_based_clustering(ehighway_shapefile['e-id'])
     assert len(clustered_regions.keys()) == 42
      

def test_distance_based_clustering(sds):
    '''Test whether the distance-based clustering works'''

    spg.distance_based_clustering(sds, mode='hierarchical', verbose=False, ax_illustration=None, save_fig=None)
