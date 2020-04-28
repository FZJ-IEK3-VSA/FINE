import spagat.dataset as spd
import spagat.representation as spr
import metis_utils.io_tools as ito
import pytest

@pytest.fixture(scope='package')
def sds():
    sds_folder_path_in = ito.Path('tests/data/input')
    sds = spd.SpagatDataSet()
    sds.read_dataset(sds_folder_path_in)
    spr.add_region_centroids(sds)

    return sds
