
import pytest

import metis_utils.io_tools as ito
import spagat.dataset as spd


def test_read_and_save_sds():
    sds_folder_path_in = ito.Path('tests/data/input')
    sds_folder_path_out = ito.Path('tests/data/output')

    sds = spd.SpagatDataSet()

    sds.read_dataset(sds_folder_path_in)
    sds.save_sds(sds_folder_path_out)
