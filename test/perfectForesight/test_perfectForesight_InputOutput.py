import numpy as np
import pandas as pd
import os
import shutil

import fine as fn
from fine.IOManagement.standardIO import writeOptimizationOutputToExcel
from pathlib import Path


def test_perfectForesight_excel(perfectForesight_test_esM):
    # optimize perfect foresight model
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")

    # create empty directory to save results
    parent__directory = Path(__file__).parent.absolute()
    dataPath = os.path.join(parent__directory, "data")
    resultPath = os.path.join(dataPath, "perfect_foresight_results")
    os.makedirs(resultPath, exist_ok=True)

    # write excel output to results folder
    files = os.path.join(resultPath, "pf_results")
    writeOptimizationOutputToExcel(
        perfectForesight_test_esM,
        outputFileName=files,
        optSumOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
            "LOPFModel": 0,
        },
        optValOutputLevel={
            "SourceSinkModel": 0,
            "ConversionModel": 0,
            "StorageModel": 0,
            "TransmissionModel": 0,
            "LOPFModel": 0,
        },
    )

    # check results and excel output
    for ip in perfectForesight_test_esM.investmentPeriodNames:
        filePath = files + f"_{ip}.xlsx"

        # check if all files are in folder
        if not os.path.isfile(filePath):
            raise ValueError(f"Result excel missing for {ip}.")

        # check if results (which are different between the ips) are correctly saved
        expected_PV_operation = perfectForesight_test_esM.getOptimizationSummary(
            "SourceSinkModel", ip=ip
        ).loc["PV", "operation", "[kW$_{el}$*h/a]"]["ForesightLand"]
        expected_PV_opexCap = perfectForesight_test_esM.getOptimizationSummary(
            "SourceSinkModel", ip=ip
        ).loc["PV", "opexCap", "[1 Euro/a]"]["ForesightLand"]
        expected_PV_npv = perfectForesight_test_esM.getOptimizationSummary(
            "SourceSinkModel", ip=ip
        ).loc["PV", "NPVcontribution", "[1 Euro]"]["PerfectLand"]
        savedExcel = pd.read_excel(
            filePath, sheet_name="SourceSinkOptSummary_1dim", index_col=[0, 1, 2]
        )
        output_PV_operation = savedExcel.loc["PV", "operation", "[kW$_{el}$*h/a]"][
            "ForesightLand"
        ]
        output_PV_opexCap = savedExcel.loc["PV", "opexCap", "[1 Euro/a]"][
            "ForesightLand"
        ]
        output_PV_npv = savedExcel.loc["PV", "NPVcontribution", "[1 Euro]"][
            "PerfectLand"
        ]
        np.testing.assert_almost_equal(expected_PV_operation, output_PV_operation)
        np.testing.assert_almost_equal(expected_PV_opexCap, output_PV_opexCap)
        np.testing.assert_almost_equal(expected_PV_npv, output_PV_npv)

    # delete folder with result files
    shutil.rmtree(resultPath)


def test_perfectForesight_netcdf_ipConversionFactors(perfectForesight_test_esM):
    perfectForesight_test_esM.add(
        fn.Conversion(
            esM=perfectForesight_test_esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                2020: {"electricity": -1, "hydrogen": 0.7},
                2025: {"electricity": -1, "hydrogen": 0.71},
                2030: {"electricity": -1, "hydrogen": 0.72},
                2035: {"electricity": -1, "hydrogen": 0.73},
                2040: {"electricity": -1, "hydrogen": 0.74},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    expected_obj = perfectForesight_test_esM.pyM.Obj().copy()

    # DICT-IO
    # export to netcdf and import again
    esm_dict, comp_dict = fn.dictIO.exportToDict(perfectForesight_test_esM)
    output_esM_dict = fn.dictIO.importFromDict(esm_dict, comp_dict)
    # run with the reloaded esM
    output_esM_dict.optimize(timeSeriesAggregation=False, solver="glpk")
    output_obj_dict = output_esM_dict.pyM.Obj()
    # test if objective values are the same
    np.testing.assert_almost_equal(
        expected_obj, output_obj_dict
    ), "The expected objective value and the output objective value differ"

    # XARRAY-IO
    esm_datasets = fn.xrIO.writeEnergySystemModelToDatasets(perfectForesight_test_esM)
    output_esM_xarray = fn.xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    # 1. test if results are saved correctly
    for ip in perfectForesight_test_esM.investmentPeriodNames:
        for mdl in perfectForesight_test_esM.componentModelingDict.keys():
            expected_OptSum = perfectForesight_test_esM.getOptimizationSummary(
                mdl, ip=ip
            )
            output_OptSum = output_esM_xarray.getOptimizationSummary(mdl, ip=ip)

            # see test/IOManagement/test_xarrayio.py: "Only total operation is
            # saved in netCDF not the yearly value, so we drop the
            # operation value. This needs to be fixed in the future."
            drop_rows_condition = [
                x
                for x in expected_OptSum.index
                if x[1] == "operation" and "h/a" in x[2]
            ]
            expected_OptSum = expected_OptSum.drop(drop_rows_condition)

            expected_OptSum = expected_OptSum.astype(float).round(2).sort_index()
            output_OptSum = output_OptSum.astype(float).round(2).sort_index()

            from pandas.testing import assert_frame_equal

            assert_frame_equal(expected_OptSum, output_OptSum, check_dtype=False)

    # 2.check result for reloaded esM from netcdf
    output_esM_xarray.optimize(timeSeriesAggregation=False, solver="glpk")
    output_obj_xarray = output_esM_xarray.pyM.Obj()
    # test if objective values are the same
    np.testing.assert_almost_equal(
        expected_obj, output_obj_xarray
    ), "The expected objective value and the output objective value differ"


def test_perfectForesight_netcdf_commisConversionFactors(perfectForesight_test_esM):
    perfectForesight_test_esM.add(
        fn.Conversion(
            esM=perfectForesight_test_esM,
            name="Electrolyzers",
            physicalUnit=r"kW$_{el}$",
            commodityConversionFactors={
                (2020, 2020): {"electricity": -1, "hydrogen": 0.7},
                (2020, 2025): {"electricity": -1, "hydrogen": 0.7},
                (2025, 2025): {"electricity": -1, "hydrogen": 0.71},
                (2025, 2030): {"electricity": -1, "hydrogen": 0.71},
                (2030, 2030): {"electricity": -1, "hydrogen": 0.72},
                (2030, 2035): {"electricity": -1, "hydrogen": 0.72},
                (2035, 2035): {"electricity": -1, "hydrogen": 0.73},
                (2035, 2040): {"electricity": -1, "hydrogen": 0.73},
                (2040, 2040): {"electricity": -1, "hydrogen": 0.74},
            },
            hasCapacityVariable=True,
            investPerCapacity=500,  # euro/kW
            opexPerCapacity=500 * 0.025,
            interestRate=0.08,
            economicLifetime=10,
        )
    )

    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    expected_obj = perfectForesight_test_esM.pyM.Obj()

    # DICT-IO
    # export to netcdf and import again
    esm_dict, comp_dict = fn.dictIO.exportToDict(perfectForesight_test_esM)
    output_esM_dict = fn.dictIO.importFromDict(esm_dict, comp_dict)
    # run with the reloaded esM
    output_esM_dict.optimize(timeSeriesAggregation=False, solver="glpk")
    output_obj_dict = output_esM_dict.pyM.Obj()
    # test if objective values are the same
    np.testing.assert_almost_equal(
        expected_obj, output_obj_dict
    ), "The expected objective value and the output objective value differ"

    # XARRAY-IO
    esm_datasets = fn.xrIO.writeEnergySystemModelToDatasets(perfectForesight_test_esM)
    output_esM_xarray = fn.xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    # 1. test if results are saved correctly
    for ip in perfectForesight_test_esM.investmentPeriodNames:
        for mdl in perfectForesight_test_esM.componentModelingDict.keys():
            expected_OptSum = perfectForesight_test_esM.getOptimizationSummary(
                mdl, ip=ip
            )
            output_OptSum = output_esM_xarray.getOptimizationSummary(mdl, ip=ip)

            # see test/IOManagement/test_xarrayio.py: "Only total operation is
            # saved in netCDF not the yearly value, so we drop the
            # operation value. This needs to be fixed in future."
            drop_rows_condition = [
                x
                for x in expected_OptSum.index
                if x[1] == "operation" and "h/a" in x[2]
            ]
            expected_OptSum = expected_OptSum.drop(drop_rows_condition)

            expected_OptSum = expected_OptSum.astype(float).round(2).sort_index()
            output_OptSum = output_OptSum.astype(float).round(2).sort_index()

            from pandas.testing import assert_frame_equal

            assert_frame_equal(expected_OptSum, output_OptSum, check_dtype=False)

    # 2.check result for reloaded esM from netcdf
    output_esM_xarray.optimize(timeSeriesAggregation=False, solver="glpk")
    output_obj_xarray = output_esM_xarray.pyM.Obj()
    # test if objective values are the same
    np.testing.assert_almost_equal(
        expected_obj, output_obj_xarray
    ), "The expected objective value and the output objective value differ"

    # 3. check consistency in commodity conversion factors
    assert (
        output_esM_xarray.getComponent("Electrolyzers").commodityConversionFactors[
            (2025, 2030)
        ]["hydrogen"]
        == perfectForesight_test_esM.getComponent(
            "Electrolyzers"
        ).commodityConversionFactors[(2025, 2030)]["hydrogen"]
    )
