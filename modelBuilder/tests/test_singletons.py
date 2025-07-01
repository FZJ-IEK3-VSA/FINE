import pytest
import os
import shutil

from modelBuilder.singletons import ModelLocations,ModelPaths,ModelTechnoEconomicData,InputDataInfo
from .test_data import test_data_folder

## Test ModelPath

#@pytest.mark.skip(reason="to be done")
def test__test1(ModelPaths_default):
    #print(ModelPaths().base_folder)
    assert ModelPaths().base_folder == os.path.join(test_data_folder, "test_output_data")