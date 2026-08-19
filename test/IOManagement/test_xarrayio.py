import io
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
        if "StorageModel/Salt caverns (hydrogen)" in path and path.endswith(
            ".data_vars[stateOfChargeOperationVariablesOptimum]"
        ):
            # A lossless cyclic storage can have a free absolute state-of-charge
            # offset. Its trajectory, rather than that solver-dependent offset,
            # is the invariant result.
            actual = actual - actual.isel(time=0)
            expected = expected - expected.isel(time=0)
        xr.testing.assert_allclose(actual, expected, rtol=2e-2, atol=1e-8)
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


# Numeric tolerance for the golden value comparison, per leaf kind.
GOLDEN_TOLERANCE = {
    "frame": {"rtol": 2e-2, "atol": 1e-8},
    "series": {"rtol": 1e-7, "atol": 1e-9},
    "scalar": {"rtol": 0.0, "atol": 0.0},
}

# Columns that describe the leaf itself instead of its position.
GOLDEN_METADATA_COLUMNS = ["kind", "dtype", "index_names", "col_name"]


def _format_label(value):
    """Convert an index or column label into a stable string."""
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def _join_names(names):
    """Join index or column level names into one string."""
    return "|".join(_format_label(name) for name in names)


def _index_level_labels(index):
    """Return one formatted label array for every level of an index."""
    if isinstance(index, pd.MultiIndex):
        levels = [index.get_level_values(level) for level in range(index.nlevels)]
    else:
        levels = [index]
    return [
        np.array([_format_label(label) for label in level], dtype=object)
        for level in levels
    ]


def _frame_to_long_table(frame, path):
    """Convert a DataFrame into one row per cell, keeping dtype and level names."""
    values = frame.to_numpy()
    number_of_rows, number_of_columns = values.shape

    table = {}
    for number, level in enumerate(_index_level_labels(frame.index)):
        table[f"index{number}"] = np.repeat(level, number_of_columns)

    column_labels = np.array(
        [_format_label(column) for column in frame.columns], dtype=object
    )
    column_dtypes = np.array([str(dtype) for dtype in frame.dtypes], dtype=object)

    table["col"] = np.tile(column_labels, number_of_rows)
    table["kind"] = "frame"
    table["dtype"] = np.tile(column_dtypes, number_of_rows)
    table["index_names"] = _join_names(frame.index.names)
    table["col_name"] = _join_names(frame.columns.names)
    table["value"] = values.ravel(order="C")

    long_table = pd.DataFrame(table)
    _add_path_columns(long_table, path)
    return long_table


def _series_to_long_table(series, path):
    """Convert a Series into one row per element."""
    table = {}
    for number, level in enumerate(_index_level_labels(series.index)):
        table[f"index{number}"] = level

    table["col"] = ""
    table["kind"] = "series"
    table["dtype"] = str(series.dtype)
    table["index_names"] = _join_names(series.index.names)
    table["col_name"] = ""
    table["value"] = series.to_numpy()

    long_table = pd.DataFrame(table)
    _add_path_columns(long_table, path)
    return long_table


def _scalar_to_long_table(value, path):
    """Convert a scalar leaf into a single row."""
    long_table = pd.DataFrame(
        {
            "col": [""],
            "kind": ["scalar"],
            "dtype": [type(value).__name__],
            "index_names": [""],
            "col_name": [""],
            "value": pd.Series([value], dtype=object),
        }
    )
    _add_path_columns(long_table, path)
    return long_table


def _add_path_columns(long_table, path):
    """Add one column for every key of the dict path that leads to the leaf."""
    for number, key in enumerate(path):
        long_table.insert(number, f"path{number}", key)


def _numbered_columns(long_table, prefix):
    """Return the columns named ``<prefix><number>``, in numeric order."""
    names = [
        name
        for name in long_table.columns
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
    ]
    return sorted(names, key=lambda name: int(name[len(prefix) :]))


def _flatten_golden(obj):
    """Convert a nested dict of pandas objects and scalars into one long table.

    Every cell becomes one row. The dict path becomes ``path*`` columns, the
    index levels become ``index*`` columns. The ``dtype``, ``index_names`` and
    ``col_name`` columns keep the metadata that a plain CSV would lose.
    """
    tables = []

    def walk(node, path):
        if isinstance(node, dict):
            for key in node:
                walk(node[key], path + [_format_label(key)])
        elif isinstance(node, pd.DataFrame):
            tables.append(_frame_to_long_table(node, path))
        elif isinstance(node, pd.Series):
            tables.append(_series_to_long_table(node, path))
        else:
            tables.append(_scalar_to_long_table(node, path))

    walk(obj, [])
    assert tables, "The golden object holds no leaves."

    long_table = pd.concat(tables, ignore_index=True, sort=False)

    path_columns = _numbered_columns(long_table, "path")
    index_columns = _numbered_columns(long_table, "index")
    ordered = path_columns + index_columns + GOLDEN_METADATA_COLUMNS + ["col", "value"]
    long_table = long_table[ordered]

    key_columns = path_columns + index_columns + ["col"]
    long_table[key_columns] = long_table[key_columns].fillna("")
    return long_table.sort_values(key_columns, kind="stable").reset_index(drop=True)


def _normalize_values(long_table):
    """Return the long table with the value column as golden CSV strings.

    The values keep their original Python types until here. Send them through
    the same CSV writer that makes the golden file. Both sides of the
    comparison then hold identical strings. All other columns are strings
    already, because the flatten step formats them.
    """
    buffer = io.StringIO()
    long_table[["value"]].to_csv(buffer, index=False)
    buffer.seek(0)
    values = pd.read_csv(buffer, dtype=str, keep_default_na=False)["value"]

    normalized = long_table.copy()
    normalized["value"] = values.to_numpy()
    return normalized


def _assert_long_tables_match(actual, expected, golden_file_name):
    """Compare two golden long tables, with numeric tolerance on the values."""
    assert list(actual.columns) == list(expected.columns), (
        f"{golden_file_name}: different columns\n"
        f"Actual columns: {list(actual.columns)}\n"
        f"Expected columns: {list(expected.columns)}"
    )

    key_columns = [
        name
        for name in actual.columns
        if name not in GOLDEN_METADATA_COLUMNS + ["value"]
    ]

    actual_keys = pd.MultiIndex.from_frame(actual[key_columns])
    expected_keys = pd.MultiIndex.from_frame(expected[key_columns])

    assert not actual_keys.has_duplicates, (
        f"{golden_file_name}: the actual result holds duplicate keys."
    )

    missing = expected_keys.difference(actual_keys)
    extra = actual_keys.difference(expected_keys)
    assert missing.empty and extra.empty, (
        f"{golden_file_name}: different rows\n"
        f"Missing in actual ({len(missing)}): {missing[:5].tolist()}\n"
        f"Only in actual ({len(extra)}): {extra[:5].tolist()}"
    )

    for column in GOLDEN_METADATA_COLUMNS:
        different = actual[column].to_numpy() != expected[column].to_numpy()
        assert not different.any(), (
            f"{golden_file_name}: different {column} on {int(different.sum())} rows\n"
            f"First difference at key {actual_keys[different][0]}: "
            f"actual {actual[column][different].iloc[0]!r}, "
            f"expected {expected[column][different].iloc[0]!r}"
        )

    actual_numbers = pd.to_numeric(actual["value"], errors="coerce").to_numpy()
    expected_numbers = pd.to_numeric(expected["value"], errors="coerce").to_numpy()
    numeric = ~np.isnan(actual_numbers) & ~np.isnan(expected_numbers)

    rtol = actual["kind"].map(lambda kind: GOLDEN_TOLERANCE[kind]["rtol"]).to_numpy()
    atol = actual["kind"].map(lambda kind: GOLDEN_TOLERANCE[kind]["atol"]).to_numpy()

    close = np.abs(actual_numbers - expected_numbers) <= (
        atol + rtol * np.abs(expected_numbers)
    )
    identical = actual["value"].to_numpy() == expected["value"].to_numpy()
    bad = np.where(numeric, ~close, ~identical)

    assert not bad.any(), (
        f"{golden_file_name}: {int(bad.sum())} values differ\n"
        f"First difference at key {actual_keys[bad][0]}:\n"
        f"Actual: {actual['value'][bad].iloc[0]!r}\n"
        f"Expected: {expected['value'][bad].iloc[0]!r}"
    )


def assert_csv_golden(actual, golden_file_name):
    """Compare a nested dict of pandas objects with a committed golden CSV file.

    The golden file holds one row per cell, gzip compressed. Set UPDATE_GOLDEN=1
    to intentionally regenerate the golden reference.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden_path = GOLDEN_DIR / golden_file_name
    long_table = _flatten_golden(actual)

    if UPDATE_GOLDEN:
        # Keep the file uncompressed. Git delta-compresses text between
        # commits, but cannot delta a compressed blob. Plain CSV therefore
        # grows the repository more slowly, and it gives a readable diff
        # when somebody regenerates a golden file.
        long_table.to_csv(golden_path, index=False)
        return

    assert golden_path.exists(), (
        f"Missing golden file: {golden_path}\n"
        "Generate it intentionally with:\n"
        f"UPDATE_GOLDEN=1 pytest {Path(__file__).as_posix()} "
        "-k <matching_golden_test_name>"
    )

    expected = pd.read_csv(golden_path, dtype=str, keep_default_na=False)
    _assert_long_tables_match(_normalize_values(long_table), expected, golden_file_name)


def collect_optimization_summaries(esM):
    """Collect getOptimizationSummary() for every model, investment period, and output level."""
    summaries = {}
    for ip in esM.investmentPeriodNames:
        summaries[ip] = {}
        for model in esM.componentModelingDict:
            summaries[ip][model] = {}
            for output_level in (0, 1, 2):
                summaries[ip][model][output_level] = esM.getOptimizationSummary(
                    model, outputLevel=output_level, ip=ip
                )
    return summaries


def _summarize_time_dependent_values(variable):
    """Replace a time-dependent frame with its shape, dtypes and level names.

    The netCDF goldens already hold every cell of these time series. Do not
    duplicate them here. Keep only the frame structure, because the netCDF
    export flattens the index into variable names and attrs, and loses it.
    """
    values = variable.get("values")
    if not variable.get("timeDependent") or not isinstance(values, pd.DataFrame):
        return variable

    summarized = {key: variable[key] for key in variable if key != "values"}
    summarized["valuesSummary"] = (
        f"shape={values.shape[0]}x{values.shape[1]} "
        f"index={_join_names(values.index.names)} "
        f"columns={_join_names(values.columns.names)} "
        f"dtypes={'|'.join(sorted({str(dtype) for dtype in values.dtypes}))}"
    )
    return summarized


def collect_optimal_values(esM):
    """Collect getOptimalValues() for every model and investment period.

    Time-dependent values become a shape summary. See
    _summarize_time_dependent_values() for the reason.
    """
    optimal_values = {}
    for ip in esM.investmentPeriodNames:
        optimal_values[ip] = {}
        for model in esM.componentModelingDict:
            values = esM.componentModelingDict[model].getOptimalValues(ip=ip)
            optimal_values[ip][model] = {
                name: _summarize_time_dependent_values(variable)
                for name, variable in values.items()
            }
    return optimal_values


def assert_pandas_optimization_results_match_golden(esM, golden_prefix):
    """Compare OptimizationSummary and OptimalValues against committed golden files."""
    assert_csv_golden(
        collect_optimization_summaries(esM),
        f"{golden_prefix}_optimization_summaries.csv",
    )
    assert_csv_golden(
        collect_optimal_values(esM),
        f"{golden_prefix}_optimal_values.csv",
    )


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
        "-k <matching_golden_test_name>"
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


def test_golden_multi_node_optimization_results(
    multi_node_test_esM_optimized, tmp_path
):
    """Regression test for committed golden optimization output of the multi-node esM."""
    assert_optimization_results_match_golden(
        multi_node_test_esM_optimized,
        "multi_node_test_esM_optimized.nc",
        tmp_path,
    )


def test_golden_perfect_foresight_optimization_results(
    perfectForesight_test_esM, tmp_path
):
    """Regression test for committed golden optimization output of the perfect-foresight esM."""
    esM = perfectForesight_test_esM
    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

    assert_optimization_results_match_golden(
        esM,
        "perfectForesight_test_esM.nc",
        tmp_path,
    )


def test_golden_minimal_pandas_optimization_results(minimal_test_esM):
    """Regression test for getOptimizationSummary() and getOptimalValues()."""
    esM = minimal_test_esM
    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

    assert_pandas_optimization_results_match_golden(esM, "minimal_test_esM")


def test_golden_multi_node_pandas_optimization_results(multi_node_test_esM_optimized):
    """Regression test for multi-node getOptimizationSummary() and getOptimalValues()."""
    assert_pandas_optimization_results_match_golden(
        multi_node_test_esM_optimized, "multi_node_test_esM_optimized"
    )


def test_golden_perfect_foresight_pandas_optimization_results(
    perfectForesight_test_esM,
):
    """Regression test for perfect-foresight getOptimizationSummary() and getOptimalValues()."""
    esM = perfectForesight_test_esM
    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

    assert_pandas_optimization_results_match_golden(esM, "perfectForesight_test_esM")


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

    removed_xarray_duplicates = {
        "capacityVariablesOptimum",
        "commissioningVariablesOptimum",
        "decommissioningVariablesOptimum",
    }

    for mdl in esm_original.componentModelingDict.keys():
        original_opt_values = esm_original.componentModelingDict[mdl].getOptimalValues()
        roundtrip_opt_values = esm_from_datasets.componentModelingDict[
            mdl
        ].getOptimalValues()

        original_opt_values = {
            key: value
            for key, value in original_opt_values.items()
            if key not in removed_xarray_duplicates
        }
        roundtrip_opt_values = {
            key: value
            for key, value in roundtrip_opt_values.items()
            if key not in removed_xarray_duplicates
        }

        compare_dicts(original_opt_values, roundtrip_opt_values)


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


def test_export_without_raw_results_raises(minimal_test_esM, tmp_path):
    """An esM read back from netCDF holds the summary but not the raw results dict, so
    re-exporting its results must fail with an explanatory error instead of an AttributeError.
    """
    test_esM = str(tmp_path / "test_esM.nc")

    esm_original = minimal_test_esM
    esm_original.optimize()
    xrIO.writeEnergySystemModelToNetCDF(
        esm_original, outputFilePath=test_esM, overwriteExisting=True
    )
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)

    with pytest.raises(RuntimeError, match="re-optimize"):
        xrIO.convertOptimizationOutputToDatasets(esm_from_netcdf)


def test_optSumOutputLevel_is_deprecated(minimal_test_esM):
    """The export cannot apply the summary's output-level filtering anymore, but the parameter
    is still accepted (as a no-op) so existing calls do not break.
    """
    esM = minimal_test_esM
    esM.optimize()

    with pytest.warns(FutureWarning, match="optSumOutputLevel"):
        withParam = xrIO.convertOptimizationOutputToDatasets(esM, optSumOutputLevel=2)

    without = xrIO.convertOptimizationOutputToDatasets(esM)

    assert withParam["Results"].keys() == without["Results"].keys()
    for ip, models in without["Results"].items():
        assert withParam["Results"][ip].keys() == models.keys()
        for model, components in models.items():
            assert withParam["Results"][ip][model].keys() == components.keys()
            for component, dataset in components.items():
                # identical() also compares the unit attributes, not only the values
                assert withParam["Results"][ip][model][component].identical(dataset), (
                    f"optSumOutputLevel changed the export of {model}/{component}"
                )


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


def test_operation_export_to_xarray(multi_node_test_esM_optimized):
    """Optimize an esM, write it to  xarray datasets, then load the esM from this file.
    Check that the results of the transmission model are identical to the initial ones.

    Info: This test will fail as soon the annual operation will not be part of the
    optimization summary anymore. In that case convertOptimizationOutputToDatasets()
    in xarrayIO.py needs to be adapted.
    """
    esM = multi_node_test_esM_optimized

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


def test_coordinates(multi_node_test_esM_optimized):
    """Optimize an esM, write it to  xarray datasets, then load the esM from this file.
    Check that the coordinates of the results of the ESM model are as expected.
    """
    esM = multi_node_test_esM_optimized

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


def test_shadow_price_data_exists_in_xarray(multi_node_test_esM_optimized):
    """Optimize an esM, write it to  xarray datasets, then load the esM from this file.
    Check that the shadow price data is part of the xarray datasets.
    """
    esM = multi_node_test_esM_optimized

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
