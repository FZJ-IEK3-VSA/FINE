import os
import shutil

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

import FINE.aggregations.spatialAggregation.managerUtils as manUtils


def test_create_gdf():
    # TEST DATA
    geometries = [Point(1, 2), Point(2, 1)]
    df = pd.DataFrame({"space": ["reg_01", "reg_02"]})

    crs = 3035
    path_to_test_dir = os.path.join(
        os.path.dirname(__file__), "../data/output/test_dir"
    )
    file_name = "test_file"

    # FUNCTION CALL
    manUtils.create_gdf(
        df, geometries, crs, file_path=path_to_test_dir, files_name=file_name
    )

    # EXPECTED
    output_shp = gpd.read_file(os.path.join(path_to_test_dir, f"{file_name}.shp"))
    assert list(output_shp.columns) == ["space", "geometry"]

    # Delete test_dir
    shutil.rmtree(path_to_test_dir)


def test_create_geom_xarray(sample_shapefile):

    expected_centroids = [Point(2, 2), Point(5.5, 2)]
    expected_centroid_distances = 0.001 * np.array(
        [[0, 3.5], [3.5, 0]]
    )  # Distances in km

    # FUNCTION CALL
    output_xr = manUtils.create_geom_xarray(
        sample_shapefile, geom_id_col_name="region_ids"
    )

    # ASSERTION
    output_centroids = output_xr["centroids"].values
    output_centroid_distances = output_xr["centroid_distances"].values

    for output, expected in zip(output_centroids, expected_centroids):
        assert output.coords[:] == expected.coords[:]

    assert np.array_equal(output_centroid_distances, expected_centroid_distances)
