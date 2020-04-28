
import pytest

import metis_utils.io_tools as ito
import spagat.dataset as spd


def test_read_and_save_sds():
    sds_folder_path_in = ito.Path('tests/data/input')
    sds_folder_path_out = ito.Path('tests/data/output')

    sds = spd.SpagatDataSet()

    sds.read_dataset(sds_folder_path_in)
    sds.save_sds(sds_folder_path_out)

def test_add_time_series_from_csv():
    '''Region time series from a csv file are read and added to the dataset'''

