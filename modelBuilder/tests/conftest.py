import pytest
import os
import shutil

from modelBuilder.singletons import ModelLocations,ModelPaths,ModelTechnoEconomicData,InputDataInfo
from .test_data import test_data_folder

## Test ModelPath

@pytest.fixture
def ModelPaths_default():
    model_base_folder = os.path.join(test_data_folder, "test_output_data")
    
    ModelPaths.reset()
    ModelPaths(
        base_folder=model_base_folder, 
        techno_economic_data_fp = None,
        default_paths_fp = None,
        intermediates_folder=None,
    )

    yield ModelPaths
    shutil.rmtree(model_base_folder)
    ModelPaths.reset()

