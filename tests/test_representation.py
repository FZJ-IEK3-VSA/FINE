import os
import pathlib

import pytest

import spagat.representation as spr

# paths
cfd = os.path.dirname(os.path.realpath(__file__))
data_path = pathlib.Path(cfd, 'data/InputData/SpatialData/')


def test_add_region_centroids_and_distances(sds):

    spr.add_region_centroids(sds)
    spr.add_centroid_distances(sds)


@pytest.mark.skip(reason='not yet implemented')
def test_aggregate_based_on_sub_to_sup_region_id_dict():
    sds = spd.SpagatDataSet()

    # ...


@pytest.mark.skip(reason='not yet implemented')
def test_aggregate_geometries():
    pass


@pytest.mark.skip(reason='not yet implemented')
def test_aggregate_time_series():
    pass


# spagat.output
@pytest.mark.skip(reason='not yet implemented')
def test_create_grid_shapefile():
    pass
