import pytest

import spagat.manager as spm
import spagat.utils as spu
import spagat.representation as spr

import os

def test_workflow():
    sds_folder_path_in = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","input")
    sds_folder_path_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","output","aggregated","33")
    spu.create_dir(sds_folder_path_out)

    n_regions = 33

    spagat_manager = spm.SpagatManager()

    spagat_manager.analysis_path = sds_folder_path_out

    spagat_manager.read_data(sds_folder_path=sds_folder_path_in)

    # TODO: correct test data after naming decision for either 'space' or 'region_ids'
    spagat_manager.sds.xr_dataset = spagat_manager.sds.xr_dataset.rename(
        {"region_ids": "space", "region_ids_2": "space_2",}
    )

    spagat_manager.grouping()

    spagat_manager.representation(number_of_regions=n_regions)

    spagat_manager.save_data(
        sds_folder_path_out,
        eligibility_variable="AC_cable_incidence",
        eligibility_component=None,
    )


if __name__ == "__main__":
    test_workflow()