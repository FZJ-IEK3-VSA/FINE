from copy import deepcopy
from pathlib import Path
from pandas import DataFrame, Series, MultiIndex, Index
from pandas.testing import assert_frame_equal, assert_series_equal
import fine.IOManagement.xarrayIO as xrIO
from fine.IOManagement.dictIO import exportToDict
import fine as fn
import pandas as pd


def compare_values(value_1, value_2):
    """Apply assert functions from pandas if values are pandas.DataFrame or
    pandas.Series, else compare with `==` operator."""
    # Dataframes and Series need a special treatment.
    if isinstance(value_1, DataFrame) and isinstance(value_2, DataFrame):
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

    elif isinstance(value_1, Series) and isinstance(value_2, Series):
        value_1.index.name = None
        value_2.index.name = None

        assert_series_equal(
            value_1.sort_index(), value_2.sort_index(), check_dtype=False
        )

    else:
        assert value_1 == value_2


def compare_dicts(dict_1: dict, dict_2: dict):
    """Iterate over the dict key-value pairs and compare those with
    `compare_values()."""
    for (key_1, value_1), (key_2, value_2) in zip(dict_1.items(), dict_2.items()):
        if isinstance(value_1, dict):
            compare_dicts(value_1, value_2)
        else:
            assert key_1 == key_2
            compare_values(value_1, value_2)


def compare_esm_inputs(esm_1: fn.EnergySystemModel, esm_2: fn.EnergySystemModel):
    """A method to assert if two esM instances have equal input parameters. It
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

            # Only total operation is saved in netCDF not the yearly value so we drop the
            # opreation value. This needs to be fixed in future.
            switch = False
            labels = set()
            for label in list(model_results_1.index.get_level_values(1).unique()):
                if label.startswith("operation"):
                    switch = True
                    labels.add(label)
            if switch:
                for label in labels:
                    model_results_1.drop(
                        index=model_results_1.xs(
                            label, axis=0, level=1, drop_level=False
                        ).index.tolist(),
                        inplace=True,
                    )
                    model_results_2.drop(
                        index=model_results_2.xs(
                            label, axis=0, level=1, drop_level=False
                        ).index.tolist(),
                        inplace=True,
                    )

            # Reading from netCDF creates a column name `space_1`. This needs to be
            # fixed in future.
            model_results_1.columns.name = None
            model_results_2.columns.name = None

            model_results_1_sorted = model_results_1.sort_index()
            model_results_2_sorted = model_results_2.sort_index()

            assert_frame_equal(model_results_1_sorted, model_results_2_sorted, check_dtype=False)


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
    xrIO.writeEnergySystemModelToNetCDF(esm_original, outputFilePath="test_esM.nc")
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

    xrIO.writeEnergySystemModelToNetCDF(esm_original, outputFilePath="test_esM.nc")
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM.nc")

    compare_esm_inputs(esm_original, esm_from_netcdf)
    compare_esm_outputs(esm_original, esm_from_netcdf)

    Path("test_esM.nc").unlink()


def test_output_esm_to_netcdf_and_back_perfectForesight(perfectForesight_test_esM):
    """Optimize an esM, write it to  netCDF, then load the esM from this file.
    Compare if both esMs are identical. Inputs are compared with exportToDict,
    outputs are compared with optimizationSummary.
    """

    esm_original_pf = deepcopy(perfectForesight_test_esM)
    esm_original_pf.optimize()

    xrIO.writeEnergySystemModelToNetCDF(
        esm_original_pf, outputFilePath="test_esM_pf.nc"
    )
    esm_pf_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath="test_esM_pf.nc")

    compare_esm_inputs(esm_original_pf, esm_pf_from_netcdf)
    compare_esm_outputs(esm_original_pf, esm_pf_from_netcdf)

    Path("test_esM_pf.nc").unlink()


def test_capacityFix_subset(multi_node_test_esM_init):
    """
    Optimize esM, set optimal capacity values for every component as capacity Fix.
    Then, save the esM to netCDF and read out the same netCDF to esM.
    Assert that capacityFix values do not have to be provided for every location when saving to NetCDF.
    Assert that capacityFix index can be a subset of locationalEligibility when reading in NetCDF.
    """
    esM = multi_node_test_esM_init

    capacityFix = Series(0, index=esM.locations)
    capacityFix["cluster_1"] = 3
    esM.add(
        fn.Conversion(
            esM=esM,
            name="New CCGT plants (biogas)",
            physicalUnit=r"GW$_{el}$",
            commodityConversionFactors={"electricity": 1, "biogas": -1 / 0.635},
            hasCapacityVariable=True,
            investPerCapacity=0.7,
            opexPerCapacity=0.021,
            interestRate=0.08,
            economicLifetime=33,
            opexPerOperation=0.01,
            locationalEligibility=Series(1, index=esM.locations),
            capacityFix=capacityFix,
            capacityMax=Series(3, index=esM.locations),
        )
    )

    fileName = "test_cdf_error.nc"
    xrIO.writeEnergySystemModelToNetCDF(esM, outputFilePath=fileName)
    _ = xrIO.readNetCDFtoEnergySystemModel(filePath=fileName)

    Path("test_cdf_error.nc").unlink()


def test_esm_to_datasets_with_processed_values(minimal_test_esM):
    esm_original = deepcopy(minimal_test_esM)

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
    esM.updateComponent(
        componentName="Pipelines",
        updateAttrs={"capacityMin": capacityMin},
    )

    time_index = pd.date_range(start="2020-01-01", periods=4, freq="H")
    _locs = pd.MultiIndex.from_product([["ElectrolyzerLocation"], ["IndustryLocation"]])
    columns = [f"{idx0}_{idx1}" for idx0, idx1 in _locs]
    column2 = [f"{idx1}_{idx0}" for idx0, idx1 in _locs]
    columns = columns + column2
    operationRateMax = pd.DataFrame(1, index=time_index, columns=columns).reset_index(
        drop=True
    )
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
