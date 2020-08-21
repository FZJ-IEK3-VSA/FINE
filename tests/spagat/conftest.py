import FINE.spagat.dataset as spd
import FINE.spagat.representation as spr

import pathlib
import pytest


@pytest.fixture(scope="package")
def sds():
    sds_folder_path_in = pathlib.Path("tests/data/input")
    sds = spd.SpagatDataset()
    sds.read_dataset(sds_folder_path_in)
    spr.add_region_centroids(sds)

    return sds
