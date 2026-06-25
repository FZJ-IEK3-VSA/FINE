import os
from pathlib import Path

import pytest

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from fine.utils import ImplementedSolvers
import fine as fn
import fine.IOManagement.xarrayIO as xrIO
from fine.IOManagement.dictIO import exportToDict
import xarray as xr


GOLDEN_DIR = Path(__file__).resolve().parents[1] / "data" / "golden"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true", "yes"}


def _assert_same_keys(actual, expected, path):
    actual_keys = set(actual.keys())
    expected_keys = set(expected.keys())
    assert actual_keys == expected_keys, (
        f"{path}: different keys\n"
        f"Only in actual: {sorted(actual_keys - expected_keys)}\n"
        f"Only in expected: {sorted(expected_keys - actual_keys)}"
    )


def _assert_attrs_equal(actual_attrs, expected_attrs, path):
    assert dict(actual_attrs) == dict(expected_attrs), (
        f"{path}: different attrs\n"
        f"Actual attrs: {dict(actual_attrs)}\n"
        f"Expected attrs: {dict(expected_attrs)}"
    )


def _assert_data_array_matches(actual, expected, path):
    assert actual.dims == expected.dims, (
        f"{path}: different dimensions\n"
        f"Actual dims: {actual.dims}\n"
        f"Expected dims: {expected.dims}"
    )

    assert actual.dtype == expected.dtype, (
        f"{path}: different dtype\n"
        f"Actual dtype: {actual.dtype}\n"
        f"Expected dtype: {expected.dtype}"
    )

    _assert_attrs_equal(actual.attrs, expected.attrs, f"{path}.attrs")

    if np.issubdtype(actual.dtype, np.number):
        xr.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    else:
        xr.testing.assert_identical(actual, expected)


def assert_xarray_dataset_matches(actual, expected, path):
    """Compare two xarray datasets including structure, attrs, dtypes, coords, and values."""
    assert isinstance(actual, xr.Dataset)
    assert isinstance(expected, xr.Dataset)

    assert dict(actual.sizes) == dict(expected.sizes), (
        f"{path}: different sizes\n"
        f"Actual sizes: {dict(actual.sizes)}\n"
        f"Expected sizes: {dict(expected.sizes)}"
    )

    _assert_attrs_equal(actual.attrs, expected.attrs, f"{path}.attrs")

    assert set(actual.coords) == set(expected.coords), (
        f"{path}: different coordinates\n"
        f"Actual coords: {set(actual.coords)}\n"
        f"Expected coords: {set(expected.coords)}"
    )

    for coord in actual.coords:
        _assert_data_array_matches(
            actual.coords[coord],
            expected.coords[coord],
            f"{path}.coords[{coord}]",
        )

    assert set(actual.data_vars) == set(expected.data_vars), (
        f"{path}: different data variables\n"
        f"Actual data variables: {set(actual.data_vars)}\n"
        f"Expected data variables: {set(expected.data_vars)}"
    )

    for variable in actual.data_vars:
        _assert_data_array_matches(
            actual[variable],
            expected[variable],
            f"{path}.data_vars[{variable}]",
        )


def assert_nested_xarray_dict_matches(actual, expected, path="Results"):
    """Recursively compare the nested xarray dictionary returned by readNetCDFToDatasets()."""
    _assert_same_keys(actual, expected, path)

    for key in actual:
        key_path = f"{path}/{key}"
        actual_value = actual[key]
        expected_value = expected[key]

        if isinstance(actual_value, dict):
            assert isinstance(expected_value, dict), f"{key_path}: expected a dict"
            assert_nested_xarray_dict_matches(actual_value, expected_value, key_path)
        else:
            assert_xarray_dataset_matches(actual_value, expected_value, key_path)


def assert_optimization_results_match_golden(esM, golden_file_name, tmp_path):
    """Write optimized esM results to netCDF and compare them with a committed golden file.

    Set UPDATE_GOLDEN=1 to regenerate the committed golden file intentionally.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / golden_file_name

    if UPDATE_GOLDEN:
        xrIO.writeEnergySystemModelToNetCDF(
            esM,
            outputFilePath=str(golden_path),
            overwriteExisting=True,
        )
        return

    assert golden_path.exists(), (
        f"Missing golden file: {golden_path}\n"
        "Generate it intentionally with:\n"
        f"UPDATE_GOLDEN=1 pytest {Path(__file__).as_posix()} "
        "-k golden_minimal_optimization_results"
    )

    actual_path = tmp_path / golden_file_name
    xrIO.writeEnergySystemModelToNetCDF(
        esM,
        outputFilePath=str(actual_path),
        overwriteExisting=True,
    )

    actual = xrIO.readNetCDFToDatasets(str(actual_path))
    expected = xrIO.readNetCDFToDatasets(str(golden_path))

    assert "Results" in actual
    assert "Results" in expected
    assert_nested_xarray_dict_matches(actual["Results"], expected["Results"])



def compare_values(value_1, value_2):
    """Apply assert functions from pandas if values are pandas.DataFrame or
    pandas.Series, else compare with `==` operator.
    """
    # Dataframes and Series need a special treatment.
    if isinstance(value_1, DataFrame) and isinstance(value_2, DataFrame):
        # Reset index names
        number_of_index_level_value_1 = value_1.index.nlevels
        value_1.index.set_names(
            names=[None] * number_of_index_level_value_1, inplace=True
        )
        # if isinstance(value_1.index, Index):
        #     value_1.index.set_names(None,inplace=True)
        # elif isinstance(value_1.index, MultiIndex):
        #     value_1.index.set_names(None,inplace=True)
        number_of_index_level_value_2 = value_2.index.nlevels
        value_2.index.set_names(
            names=[None] * number_of_index_level_value_2, inplace=True
        )
        # if isinstance(value_2.index, Index):
        #     value_2.index.set_names(None,inplace=True)
        # elif isinstance(value_2.index, MultiIndex):
        #     value_2.index.set_names(None,inplace=True)

        value_1.columns.set_names(names=[None], inplace=True)
        value_2.columns.set_names(names=[None], inplace=True)

        assert_frame_equal(
            value_1.sort_index(), value_2.sort_index(), check_dtype=False
        )

    elif isinstance(value_1, Series) and isinstance(value_2, Series):
        value_1.index.set_names(names=[None], inplace=True)
        value_2.index.set_names(names=[None], inplace=True)

        assert_series_equal(
            value_1.sort_index(), value_2.sort_index(), check_dtype=False
        )

    else:
        assert value_1 == value_2


def compare_dicts(dict_1: dict, dict_2: dict):
    """Iterate over the dict key-value pairs and compare those with
    `compare_values().
    """
    for (key_1, value_1), (key_2, value_2) in zip(dict_1.items(), dict_2.items()):
        if isinstance(value_1, dict):
            compare_dicts(value_1, value_2)
        else:
            assert key_1 == key_2
            compare_values(value_1, value_2)


def compare_esm_inputs(esm_1: fn.EnergySystemModel, esm_2: fn.EnergySystemModel):
    """Assert if two esM instances have equal input parameters. It
    uses exportToDict() and compares all attributes.

    :param esm1:
    :type esm1: FINE.EnergySystemModel
    :param esm2:
    :type esm2: FINE.EnergySystemModel
    """
    # Create (esm_dict, comp_dict) tuples
    esm_tuple_1 = exportToDict(esm_1)
    esm_tuple_2 = exportToDict(esm_2)

    for dict_1, dict_2 in zip(esm_tuple_1, esm_tuple_2):
        compare_dicts(dict_1, dict_2)


def compare_esm_outputs(esm_1: fn.EnergySystemModel, esm_2: fn.energySystemModel):
    """Compare `optimizationSummary` of two instances of fn.EnergySystemModel."""
    for ip in esm_1.investmentPeriodNames:
        results_1 = {}
        results_2 = {}
        for model in esm_1.componentModelingDict.keys():
            results_1[model] = esm_1.getOptimizationSummary(model, outputLevel=0, ip=ip)
        for model in esm_2.componentModelingDict.keys():
            results_2[model] = esm_2.getOptimizationSummary(model, outputLevel=0, ip=ip)

        assert results_1.keys() == results_2.keys()

        for model_key, model_results_1 in results_1.items():
            model_results_2 = results_2[model_key]

            model_results_1_sorted = model_results_1.sort_index()
            model_results_2_sorted = model_results_2.sort_index()

            assert_frame_equal(
                model_results_1_sorted, model_results_2_sorted, check_dtype=False
            )


def test_golden_minimal_optimization_results(minimal_test_esM, tmp_path):
    """Regression test for committed golden optimization output of the minimal esM."""
    esM = minimal_test_esM
    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

    assert_optimization_results_match_golden(
        esM,
        "minimal_test_esM.nc",
        tmp_path,
    )


def test_esm_input_to_dataset_and_back(minimal_test_esM):
    esm_original = minimal_test_esM

    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esm_original)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    compare_esm_inputs(esm_original, esm_from_datasets)


def test_esm_output_to_dataset_and_back(minimal_test_esM):
    esm_original = minimal_test_esM
    esm_original.optimize()
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esm_original)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    compare_esm_inputs(esm_original, esm_from_datasets)
    compare_esm_outputs(esm_original, esm_from_datasets)

    for mdl in esm_original.componentModelingDict.keys():
        compare_dicts(
            esm_original.componentModelingDict[mdl].getOptimalValues(),
            esm_from_datasets.componentModelingDict[mdl].getOptimalValues(),
        )


def test_input_esm_to_netcdf_and_back(minimal_test_esM, tmp_path):
    """Write an esM to netCDF, then load the esM from this file. Compare if both
    esMs are identical.
    """
    test_esM = str(tmp_path / "test_esM.nc")

    esm_original = minimal_test_esM
    xrIO.writeEnergySystemModelToNetCDF(
        esm_original, outputFilePath=test_esM, overwriteExisting=True
    )
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)

    compare_esm_inputs(esm_original, esm_from_netcdf)


def test_output_esm_to_netcdf_and_back(minimal_test_esM, tmp_path):
    """Optimize an esM, write it to  netCDF, then load the esM from this file.
    Compare if both esMs are identical. Inputs are compared with exportToDict,
    outputs are compared with optimizationSummary.
    """
    test_esM = str(tmp_path / "test_esM.nc")

    esm_original = minimal_test_esM
    esm_original.optimize()
    xrIO.writeEnergySystemModelToNetCDF(
        esm_original, outputFilePath=test_esM, overwriteExisting=True
    )
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)

    compare_esm_inputs(esm_original, esm_from_netcdf)
    compare_esm_outputs(esm_original, esm_from_netcdf)


def test_output_esm_to_netcdf_and_back_perfectForesight(
    perfectForesight_test_esM, tmp_path
):
    """Optimize an esM, write it to  netCDF, then load the esM from this file.
    Compare if both esMs are identical. Inputs are compared with exportToDict,
    outputs are compared with optimizationSummary.
    """
    test_esM = str(tmp_path / "test_esM_pf.nc")

    esm_original_pf = perfectForesight_test_esM
    esm_original_pf.optimize()

    xrIO.writeEnergySystemModelToNetCDF(esm_original_pf, outputFilePath=test_esM)
    esm_pf_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)
    compare_esm_inputs(esm_original_pf, esm_pf_from_netcdf)
    compare_esm_outputs(esm_original_pf, esm_pf_from_netcdf)


def test_capacityFix_subset(multi_node_test_esM_init, tmp_path):
    """Optimize esM, set optimal capacity values for every component as capacity Fix.
    Then, save the esM to netCDF and read out the same netCDF to esM.
    Assert that capacityFix values do not have to be provided for every location when saving to NetCDF.
    Assert that capacityFix index can be a subset of locationalEligibility when reading in NetCDF.
    """
    esM = multi_node_test_esM_init

    capacityFix = Series(0, index=esM.locations)
    capacityFix["cluster_1"] = 3
    with pytest.warns(
        UserWarning,
        match="Component identifier New CCGT plants \\(biogas\\) already exists",
    ):
        multi_node_test_esM_init.updateComponent(
            componentName="New CCGT plants (biogas)",
            updateAttrs={
                "opexPerOperation": 0.01,
                "locationalEligibility": Series(1, index=esM.locations),
                "capacityFix": capacityFix,
                "capacityMax": Series(3, index=esM.locations),
            },
        )

    test_esM = str(tmp_path / "test_cdf_error.nc")

    xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath=test_esM)
    _ = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)


def test_esm_to_datasets_with_processed_values(minimal_test_esM):
    esm_original = minimal_test_esM

    xr_dss = xrIO.convertOptimizationInputToDatasets(
        esm_original, useProcessedValues=True
    )
    assert (
        xr_dss.get("Input")
        .get("Transmission")
        .get("Pipelines")["0d_investPerCapacity.0"]
        .item()
        == 0.177
    )


def test_transmission_dims(minimal_test_esM):
    esM = minimal_test_esM
    capacityMin = pd.DataFrame(
        [[0, 1], [1, 0]], index=list(esM.locations), columns=list(esM.locations)
    )

    # update Pipeline component
    with pytest.warns(
        UserWarning, match="Component identifier Pipelines already exists"
    ):
        esM.updateComponent(
            componentName="Pipelines",
            updateAttrs={"capacityMin": capacityMin},
        )

    time_index = pd.date_range(start="2020-01-01", periods=4, freq="h")
    _locs = pd.MultiIndex.from_product([["ElectrolyzerLocation"], ["IndustryLocation"]])
    columns = [f"{idx0}_{idx1}" for idx0, idx1 in _locs]
    column2 = [f"{idx1}_{idx0}" for idx0, idx1 in _locs]
    columns = columns + column2
    operationRateMax = pd.DataFrame(1, index=time_index, columns=columns).reset_index(
        drop=True
    )
    with pytest.warns(
        UserWarning, match="Component identifier Pipelines already exists"
    ):
        esM.updateComponent(
            componentName="Pipelines",
            updateAttrs={"operationRateMax": operationRateMax},
        )

    esM.optimize()
    xr_dss = xrIO.convertOptimizationInputToDatasets(esM)
    assert esM.totalTimeSteps == list(
        xr_dss["Input"]["Transmission"]["Pipelines"].time.to_numpy()
    )

    esM2 = xrIO.convertDatasetsToEnergySystemModel(xr_dss)

    operationRateMax = esM2.getComponentAttribute("Pipelines", "operationRateMax")
    assert operationRateMax.index.name == "time"

    esM2.optimize()


def test_saving_clustered_timeseries_to_xarray(perfectForesight_test_esM, tmp_path):
    """Optimize an esM, write it to  netCDF, then load the esM from this file.
    Compare if both esMs are identical. Inputs are compared with exportToDict,
    outputs are compared with optimizationSummary.
    """
    esm_original_pf = perfectForesight_test_esM
    esm_original_pf.aggregateTemporally(
        numberOfTypicalPeriods=1, numberOfTimeStepsPerPeriod=2
    )
    esm_original_pf.optimize()
    path_to_output_file_str = str(tmp_path / "test_esM_pf.nc")
    xrIO.writeEnergySystemModelToNetCDF(
        esm_original_pf, outputFilePath=path_to_output_file_str
    )
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esm_original_pf)
    assert "ts_aggregatedOperationRateMax" in esm_datasets["Input"]["Source"]["PV"]

    esm_pf_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(
        filePath=path_to_output_file_str
    )

    compare_esm_inputs(esm_original_pf, esm_pf_from_netcdf)


def test_operation_export_to_xarray(multi_node_test_esM_init):
    """Optimize an esM, write it to  xarray datasets, then load the esM from this file.
    Check that the results of the transmission model are identical to the initial ones.

    Info: This test will fail as soon the annual operation will not be part of the
    optimization summary anymore. In that case convertOptimizationOutputToDatasets()
    in xarrayIO.py needs to be adapted.
    """
    esM = multi_node_test_esM_init
    esM.aggregateTemporally(
        numberOfTypicalPeriods=5,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )
    esM.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    xrds = xrIO.writeEnergySystemModelToDatasets(esM)
    optSum = (
        esM.getOptimizationSummary("TransmissionModel")
        .loc["DC cables", "operation", "[GW$_{el}$*h]"]
        .dropna(how="all")
    )
    xrRes = (
        xrds["Results"][0]["TransmissionModel"]["DC cables"]
        .operation.to_series()
        .unstack()
        .dropna(how="all")
    )
    xrRes.columns.name = None

    assert_frame_equal(optSum, xrRes, check_dtype=False)


def test_coordinates(multi_node_test_esM_init):
    """Optimize an esM, write it to  xarray datasets, then load the esM from this file.
    Check that the coordinates of the results of the ESM model are as expected.
    """
    esM = multi_node_test_esM_init
    esM.aggregateTemporally(
        numberOfTypicalPeriods=5,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )
    esM.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

    xrds = xrIO.writeEnergySystemModelToDatasets(esM)

    required_coord_1dim = {"location", "time"}
    xrRes = xrds["Results"][0]["SourceSinkModel"]["Wind (onshore)"]
    assert set(xrRes.coords) == required_coord_1dim, (
        f"Expected {required_coord_1dim}; got {set(xrRes.coords)}."
    )

    required_coord_2dim = {"locationIn", "locationOut", "time"}
    xrRes = xrds["Results"][0]["TransmissionModel"]["Pipelines (biogas)"]
    assert set(xrRes.coords) == required_coord_2dim, (
        f"Expected {required_coord_2dim}; got {set(xrRes.coords)}"
    )


def test_shadow_price_data_exists_in_xarray(multi_node_test_esM_init):
    """Optimize an esM, write it to  xarray datasets, then load the esM from this file.
    Check that the shadow price data is part of the xarray datasets.
    """
    esM = multi_node_test_esM_init
    esM.aggregateTemporally(
        numberOfTypicalPeriods=3,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )
    esM.optimize(
        timeSeriesAggregation=True, solver=ImplementedSolvers.STANDARD_SOLVER.value
    )

    xrds = xrIO.writeEnergySystemModelToDatasets(esM, includeShadowPrices=True)
    assert "ShadowPrices" in xrds.keys()
    assert isinstance(xrds["ShadowPrices"], xr.DataArray)
    # assert that "ip", "component", "space" and "time" are dimensions of the ShadowPrices DataArray
    assert set(["ip", "component", "space", "time"]).issubset(
        set(xrds["ShadowPrices"].dims)
    )

    # Test fail behaviour if nonexistent constraint is given: msg = f"Constraint '{constraint_str}' not found in model."
    with pytest.raises(
        ValueError, match="Constraint 'non_existent_constraint' not found in model."
    ):
        xrIO.writeEnergySystemModelToDatasets(
            esM,
            includeShadowPrices=True,
            shadowPriceConstraintStr="non_existent_constraint",
        )


def test_shadow_price_with_multiple_ip(perfectForesight_test_esM):
    """Test that shadow prices are written correctly for a model with multiple investment periods.
    Specifically exercises the xr.concat path in getShadowPriceXarray (hit from ip=1 onward).
    """
    esM = perfectForesight_test_esM

    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

    xrds = xrIO.writeEnergySystemModelToDatasets(esM, includeShadowPrices=True)

    assert "ShadowPrices" in xrds.keys()
    sp = xrds["ShadowPrices"]
    assert isinstance(sp, xr.DataArray)
    assert set(["ip", "component", "space", "time"]).issubset(set(sp.dims))
    assert list(sp.coords["ip"].values) == esM.investmentPeriodNames
