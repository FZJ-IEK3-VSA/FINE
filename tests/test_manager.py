import pytest

import pathlib
import spagat.manager as spm
import spagat.utils as spu


def test_workflow():
    sds_folder_path_in = pathlib.Path("tests/data/input")
    sds_folder_path_out = pathlib.Path("tests/data/output/aggregated/33")
    spu.create_dir(sds_folder_path_out)

    n_regions = 33

    spagat_manager = spm.SpagatManager()

    spagat_manager.analysis_path = sds_folder_path_out

    spagat_manager.read_data(sds_folder_path=sds_folder_path_in)

    spagat_manager.grouping()

    spagat_manager.representation(number_of_regions=n_regions)

    spagat_manager.save_data(sds_folder_path_out)
