import FINE as fn
import numpy as np
import pandas as pd


def test_perfectForesight_mini(perfectForesight_test_esM):
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="gurobi")
    print(perfectForesight_test_esM)
    expected_obj = perfectForesight_test_esM.pyM.Obj()

    # DICT-IO
    # export to netcdf and import again
    esm_dict, comp_dict = fn.dictIO.exportToDict(perfectForesight_test_esM)
    output_esM_dict = fn.dictIO.importFromDict(esm_dict, comp_dict)
    # run with the reloaded esM
    output_esM_dict.optimize(timeSeriesAggregation=False, solver="gurobi")
    output_obj_dict = output_esM_dict.pyM.Obj()
    # test if objective values are the same
    np.testing.assert_almost_equal(
        expected_obj, output_obj_dict
    ), "The expected objective value and the output objective value differ"

    # XARRAY-IO
    esm_datasets = fn.xrIO.writeEnergySystemModelToDatasets(perfectForesight_test_esM)
    output_esM_xarray = fn.xrIO.convertDatasetsToEnergySystemModel(esm_datasets)
    # run with the reloaded esM
    output_esM_xarray.optimize(timeSeriesAggregation=False, solver="gurobi")
    output_obj_xarray = output_esM_xarray.pyM.Obj()
    # test if objective values are the same
    np.testing.assert_almost_equal(
        expected_obj, output_obj_xarray
    ), "The expected objective value and the output objective value differ"
