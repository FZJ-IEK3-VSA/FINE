import os
import pytest

import pathlib
import spagat.dataset as spd


def test_read_and_save_sds():
    sds_folder_path_in =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","input")
    sds_folder_path_out =  os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","output")

    sds = spd.SpagatDataset()

    sds.read_dataset(sds_folder_path_in)
    sds.save_sds(sds_folder_path_out)

@pytest.mark.skip('not implemented')
def test_add_time_series_from_csv():
    '''Region time series from a csv file are read and added to the dataset'''


