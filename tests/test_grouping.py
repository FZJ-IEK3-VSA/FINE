import pytest
import xarray as xr 
import geopandas
import geoplot 

import spagat.grouping as spg
import spagat.dataset as spd

#read e-highway shapefile 
path_to_ehighwayfile = "/home/s-patil/data/e-highway"
ehighway_shapefile = geopandas.read_file(path_to_ehighwayfile, layer='e-highway')
#@pytest.mark.skip()

def test_string_based_clustering():
     clustered_regions = spg.string_based_clustering(ehighway_shapefile['e-id'])
     assert len(clustered_regions.keys()) == 42
      