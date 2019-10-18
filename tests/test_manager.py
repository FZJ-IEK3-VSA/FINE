import pytest

import metis_utils.io_tools as ito
import spagat.manager as spm


def test_workflow():
    sds_folder_path_in = ito.Path('tests/data/input')
    sds_folder_path_out = ito.Path('tests/data/output/aggregated/33')
    ito.create_dir(sds_folder_path_out)

    n_regions = 33

    spagat_manager = spm.SpagatManager()

    spagat_manager.analysis_path = sds_folder_path_out

    spagat_manager.read_data(sds_folder=sds_folder_path_in)

    spagat_manager.grouping()

    spagat_manager.representation(number_of_regions=n_regions)

    spagat_manager.save_data(sds_folder_path_out)
