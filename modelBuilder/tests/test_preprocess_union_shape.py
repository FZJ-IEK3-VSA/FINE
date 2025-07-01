import pytest
import geokit as gk
import geopandas as gpd
import os

from modelBuilder.inputDataHandler import preprocess_union_shape

from .test_data import test_data_folder

#@pytest.fixture
def test_shape_gpd():
    location_shape_path = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    shape_gpd = gpd.read_file(location_shape_path)

    return shape_gpd

#@pytest.fixture
def test_shape_gk():
    location_shape_path = os.path.join(test_data_folder, "input_data", "test_regions.shp")
    shape_gk = gk.vector.extractFeatures(location_shape_path)

    return shape_gk


@pytest.mark.skip("Not implemented, tbd")
def test_preprocess_union_shape(test_shape_gk):
    shape_gk = test_shape_gk()
    # shape_gpd = shape_gpd[shape_gpd["Union_No"] == 23]

    #%%
    
    shape_aggregated_gpd = preprocess_union_shape(
        location_shape=shape_gk,
        max_regions=2,
        #as_gk=True,
    )
    assert False

if __name__ == "__main__":
    test_preprocess_union_shape(test_shape_gk)