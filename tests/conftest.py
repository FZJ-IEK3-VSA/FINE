import spagat.dataset as spd
import spagat.representation as spr

import pathlib
import pytest


@pytest.fixture(scope="package")
def sds():
    sds_folder_path_in = pathlib.Path("spagat/tests/data/input")
    sds = spd.SpagatDataSet()
    sds.read_dataset(sds_folder_path_in)
    spr.add_region_centroids(sds)

    return sds
