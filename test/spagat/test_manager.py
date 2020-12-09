#TODO: remove this file

# import pytest

# import FINE.spagat.manager as spm
# import FINE.spagat.utils as spu
# import FINE.spagat.representation as spr

# import os

# @pytest.mark.skip("reason: changes required")
# def test_workflow():
#     sds_folder_path_in = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","input")
#     sds_folder_path_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","output","aggregated","33")
#     spu.create_dir(sds_folder_path_out)

#     n_regions = 2

#     spagat_manager = spm.SpagatManager()

#     spagat_manager.analysis_path = sds_folder_path_out

#     spagat_manager.read_data(sds_folder_path=sds_folder_path_in)

#     spagat_manager.grouping()

#     spagat_manager.representation(number_of_regions=n_regions)

#     spagat_manager.save_data(
#         sds_folder_path_out,
#         eligibility_variable="2d_locationalEligibility",
#         eligibility_component="Transmission, h2pipeline",
#     )


# if __name__ == "__main__":
#     test_workflow()