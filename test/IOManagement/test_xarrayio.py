from copy import deepcopy
from pathlib import Path
from pandas import DataFrame, Series, MultiIndex, Index
from pandas.testing import assert_frame_equal, assert_series_equal
import FINE.IOManagement.xarrayIO as xrIO
from FINE.IOManagement.dictIO import exportToDict


def compare_values(value_1, value_2):
    # Dataframes and Series need a special treatment.
    if isinstance(value_1, DataFrame):

        # Reset index names
        if isinstance(value_1.index, Index):
            value_1.index.name = None
        elif isinstance(value_1.index, MultiIndex):
            value_1.index.names = None

        if isinstance(value_2.index, Index):
            value_2.index.name = None
        elif isinstance(value_2.index, MultiIndex):
            value_2.index.names = None

        value_1.columns.name = None
        value_2.columns.name = None

        assert_frame_equal(
            value_1.sort_index(), value_2.sort_index(), check_dtype=False
        )

    elif isinstance(value_1, Series):
        value_1.index.name = None
        value_2.index.name = None

        assert_series_equal(
            value_1.sort_index(), value_2.sort_index(), check_dtype=False
        )

    else:
        assert value_1 == value_2


def compare_dicts(dict_1, dict_2):
    for ((key_1, value_1), (key_2, value_2)) in zip(dict_1.items(), dict_2.items()):
        # If the values are dicts we iterate over the dict key-value pairs
        # and compare those.
        if isinstance(value_1, dict):
            compare_dicts(value_1, value_2)
        else:
            assert key_1 == key_2
            compare_values(value_1, value_2)


def compare_esm_inputs(esm_1, esm_2):
    """A method to assert if two esM instances have equal input parameters. It
    uses exportToDict and compares all attributes.

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


def compare_esm_outputs(esm_1, esm_2):

    results_original = {}
    results_from_netcdf = {}
    for model in esm_1.componentModelingDict.keys():
        results_original[model] = esm_1.getOptimizationSummary(model, outputLevel=0)
    for model in esm_2.componentModelingDict.keys():
        results_from_netcdf[model] = esm_2.getOptimizationSummary(model, outputLevel=0)

    assert results_original.keys() == results_from_netcdf.keys()

    for model_key in results_original.keys():
        model_results_original = results_original[model_key]
        model_results_from_netcdf = results_from_netcdf[model_key]

        # Only total operation is saved in netCDF not the yearly value so we drop the
        # opreation value. This needs to be fixed in future.
        switch = False
        labels = set()
        for label in list(model_results_original.index.get_level_values(1).unique()):
            if label.startswith("operation"):
                switch = True
                labels.add(label)
        if switch:
            for label in labels:
                model_results_original.drop(
                    index=model_results_original.xs(
                        label, axis=0, level=1, drop_level=False
                    ).index.tolist(),
                    inplace=True,
                )
                model_results_from_netcdf.drop(
                    index=model_results_from_netcdf.xs(
                        label, axis=0, level=1, drop_level=False
                    ).index.tolist(),
                    inplace=True,
                )

        # Reading from netCDF creates a column name `space_1`. This needs to be
        # fixed in future.
        model_results_original.columns.name = None
        model_results_from_netcdf.columns.name = None

        assert_frame_equal(
            model_results_original, model_results_from_netcdf, check_dtype=False
        )


def test_esm_input_to_dataset_and_back(minimal_test_esM):

    esm_original = deepcopy(minimal_test_esM)

    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esm_original)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    compare_esm_inputs(esm_original, esm_from_datasets)


def test_esm_output_to_dataset_and_back(minimal_test_esM):

    esm_original = deepcopy(minimal_test_esM)
    esm_original.optimize()
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esm_original)
    esm_from_datasets = xrIO.convertDatasetsToEnergySystemModel(esm_datasets)

    compare_esm_inputs(esm_original, esm_from_datasets)
    compare_esm_outputs(esm_original, esm_from_datasets)


def test_input_esm_to_netcdf_and_back(minimal_test_esM):
    """Write an esM to netCDF, then load the esM from this file. Compare if both
    esMs are identical.
    """

    esm_original = deepcopy(minimal_test_esM)
    _ = xrIO.writeEnergySystemModelToNetCDF(esm_original, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")

    compare_esm_inputs(esm_original, esm_from_netcdf)

    Path("test_esM.nc").unlink()


def test_output_esm_to_netcdf_and_back(minimal_test_esM):
    """Optimize an esM, write it to  netCDF, then load the esM from this file.
    Compare if both esMs are identical. Inputs are compared with exportToDict,
    outputs are compared with optimizationSummary.
    """

    esm_original = deepcopy(minimal_test_esM)
    esm_original.optimize()

    _ = xrIO.writeEnergySystemModelToNetCDF(esm_original, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")

    compare_esm_inputs(esm_original, esm_from_netcdf)
    compare_esm_outputs(esm_original, esm_from_netcdf)

    Path("test_esM.nc").unlink()
