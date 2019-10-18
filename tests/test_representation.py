import pytest

import spagat.dataset as spd
import spagat.representation as spr


@pytest.mark.skip()
def test_add_region_centroids_and_distances():

    sds = spd.SpagatDataSet()

    # ...

    spr.add_region_centroids(sds)
    spr.add_centroid_distances(sds)


@pytest.mark.skip()
def test_aggregate_based_on_sub_to_sup_region_id_dict():
    sds = spd.SpagatDataSet()

    # ...


@pytest.mark.skip()
def test_aggregate_geometries():
    pass


@pytest.mark.skip()
def test_aggregate_time_series():
    pass


# spagat.output
@pytest.mark.skip()
def test_create_grid_shapefile():
    pass
