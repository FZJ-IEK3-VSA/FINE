import json

import pytest

import pandas as pd
from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal, assert_series_equal

from fine.utils import ImplementedSolvers
import fine as fn
import fine.IOManagement.xarrayIO as xrIO
from fine.IOManagement.dictIO import exportToDict
import xarray as xr


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


def _roundtrip_through_netcdf_folder(esm, base_path, **kwargs):
    """Write an optimized esM to a netCDF folder and read it back into an esM."""
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(esm)
    xrIO.writeDatasetsToNetCDFfolder(esm_datasets, base_path=base_path)
    read_datasets = xrIO.readNetCDFfolderToDatasets(base_path=base_path, **kwargs)
    return xrIO.convertDatasetsToEnergySystemModel(read_datasets)


@pytest.mark.parametrize("parallel", [False, True])
def test_esm_output_to_netcdf_folder_and_back(minimal_test_esM, tmp_path, parallel):
    """Optimize an esM, write it to a netCDF folder and read it back.

    The folder format writes one file per dataset instead of one large file.
    Both the serial and the parallel read have to give the same esM back.
    """
    esm_original = minimal_test_esM
    esm_original.optimize()

    esm_from_folder = _roundtrip_through_netcdf_folder(
        esm_original, str(tmp_path / "test_esM"), parallel=parallel
    )

    compare_esm_inputs(esm_original, esm_from_folder)
    compare_esm_outputs(esm_original, esm_from_folder)


def test_esm_output_to_netcdf_folder_and_back_lazy(minimal_test_esM, tmp_path):
    """The same round trip, with the data read on demand instead of at once."""
    esm_original = minimal_test_esM
    esm_original.optimize()

    esm_from_folder = _roundtrip_through_netcdf_folder(
        esm_original, str(tmp_path / "test_esM"), lazy_load=True, parallel=False
    )

    compare_esm_inputs(esm_original, esm_from_folder)
    compare_esm_outputs(esm_original, esm_from_folder)


def test_netcdf_folder_structure_file_holds_relative_paths(minimal_test_esM, tmp_path):
    """structure.json has to stay free of absolute paths, so the tree can be moved."""
    base_path = tmp_path / "test_esM"
    esm_datasets = xrIO.writeEnergySystemModelToDatasets(minimal_test_esM)
    xrIO.writeDatasetsToNetCDFfolder(esm_datasets, base_path=str(base_path))

    with (base_path / "structure.json").open() as structureFile:
        structure = json.load(structureFile)

    paths = []

    def collect(item):
        if isinstance(item, dict):
            for value in item.values():
                collect(value)
        else:
            paths.append(item)

    collect(structure)
    assert paths, "the structure file has to name at least one dataset"
    for path in paths:
        assert not path.startswith("/")
        assert (base_path / path).is_file()


def _esM_with_component_limit():
    """Build a small two-region esM that uses every componentLimit argument."""
    locations = {"R1", "R2"}
    esM = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        componentLimit=pd.DataFrame(
            index=["capLimit", "opLimit"],
            columns=["value", "bound", "type", "commodity", "ip", "ipEnd"],
            data=[
                [100.0, "upper", "capacity", None, 0, None],
                [5000.0, "upper", "operation", None, 0, None],
            ],
        ),
        componentLimitEligibility=pd.DataFrame(
            index=sorted(locations),
            columns=["capLimit", "opLimit"],
            data=[[1, 1], [1, 0]],
        ),
        componentLimitEligibility2dim=pd.DataFrame(
            index=pd.MultiIndex.from_tuples([("R1", "R2"), ("R2", "R1")]),
            columns=["gridLimit"],
            data=[[1], [1]],
        ),
        componentLimitGrouping=pd.DataFrame(
            index=sorted(locations),
            columns=["capLimit"],
            data=[["DE"], ["FR"]],
        ),
    )
    esM.add(
        fn.Source(
            esM=esM,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            # a multi-entry componentLimitID has to survive the round trip too
            componentLimitID=["capLimit", "opLimit"],
        )
    )
    esM.add(
        fn.Sink(
            esM=esM,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame([[1.0, 1.0]] * 4, columns=sorted(locations)),
        )
    )
    return esM


def test_componentLimit_netcdf_roundtrip(tmp_path):
    """Write an esM that uses componentLimit to netCDF and read it back.

    The four esM level componentLimit arguments and the per component
    componentLimitID all have to come back unchanged.
    """
    test_esM = str(tmp_path / "test_esM.nc")

    esm_original = _esM_with_component_limit()
    xrIO.writeEnergySystemModelToNetCDF(
        esm_original, outputFilePath=test_esM, overwriteExisting=True
    )
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)

    compare_esm_inputs(esm_original, esm_from_netcdf)

    assert_frame_equal(esm_original.componentLimit, esm_from_netcdf.componentLimit)
    assert_frame_equal(
        esm_original.componentLimitEligibility,
        esm_from_netcdf.componentLimitEligibility,
    )
    assert_frame_equal(
        esm_original.componentLimitEligibility2dim,
        esm_from_netcdf.componentLimitEligibility2dim,
    )
    assert_frame_equal(
        esm_original.componentLimitGrouping, esm_from_netcdf.componentLimitGrouping
    )
    assert esm_from_netcdf.getComponent("PV").componentLimitID == [
        "capLimit",
        "opLimit",
    ]


def test_balanceLimit_netcdf_roundtrip(tmp_path):
    """Write an esM that uses balanceLimit to netCDF and read it back.

    balanceLimit used to come back as None, because it was only rebuilt when the
    file held no balanceLimit at all.
    """
    test_esM = str(tmp_path / "test_esM.nc")
    locations = {"R1", "R2"}

    esm_original = fn.EnergySystemModel(
        locations=locations,
        commodities={"electricity"},
        numberOfTimeSteps=4,
        commodityUnitsDict={"electricity": r"kW$_{el}$"},
        hoursPerTimeStep=2190,
        costUnit="1 Euro",
        lengthUnit="km",
        verboseLogLevel=2,
        balanceLimit=pd.DataFrame(
            index=["CO2"],
            columns=["R1", "R2", "lowerBound"],
            data=[[100.0, 200.0, False]],
        ),
    )
    esm_original.add(
        fn.Source(
            esM=esm_original,
            name="PV",
            commodity="electricity",
            hasCapacityVariable=True,
            balanceLimitID="CO2",
        )
    )
    esm_original.add(
        fn.Sink(
            esM=esm_original,
            name="Demand",
            commodity="electricity",
            hasCapacityVariable=False,
            operationRateFix=pd.DataFrame([[1.0, 1.0]] * 4, columns=sorted(locations)),
        )
    )

    xrIO.writeEnergySystemModelToNetCDF(
        esm_original, outputFilePath=test_esM, overwriteExisting=True
    )
    esm_from_netcdf = xrIO.readNetCDFtoEnergySystemModel(filePath=test_esM)

    assert esm_from_netcdf.balanceLimit is not None
    assert_frame_equal(esm_original.balanceLimit, esm_from_netcdf.balanceLimit)


def test_esm_input_to_zarr_and_back(minimal_test_esM, tmp_path):
    """Write an esM to a Zarr store and read it back.

    The Zarr layout records the shape of every parameter in a dimension mask and
    stacks the components of a class into one dataset, so this exercises both the
    mask layer and the concatenation.
    """
    store = str(tmp_path / "test_esM.zarr")

    esm_original = minimal_test_esM
    datasets = xrIO.convertOptimizationInputToDatasetsZarr(esm_original)
    xrIO.writeDatasetsToZarr(datasets, output_zarr_path=store)
    esm_from_zarr = xrIO.readZarrToEnergySystemModel(store)

    compare_esm_inputs(esm_original, esm_from_zarr)


def test_esm_input_with_componentLimit_to_zarr_and_back(tmp_path):
    """The Zarr round trip has to carry componentLimit as well."""
    store = str(tmp_path / "test_esM.zarr")

    esm_original = _esM_with_component_limit()
    datasets = xrIO.convertOptimizationInputToDatasetsZarr(esm_original)
    xrIO.writeDatasetsToZarr(datasets, output_zarr_path=store)
    esm_from_zarr = xrIO.readZarrToEnergySystemModel(store)

    compare_esm_inputs(esm_original, esm_from_zarr)
    assert_frame_equal(esm_original.componentLimit, esm_from_zarr.componentLimit)
    assert esm_from_zarr.getComponent("PV").componentLimitID == [
        "capLimit",
        "opLimit",
    ]


def test_writeEnergySystemModelToDatasetsBoth_agrees_with_the_separate_writers(
    minimal_test_esM,
):
    """One export has to give the same two views as two separate exports.

    writeEnergySystemModelToDatasetsBoth shares a single dictIO.exportToDict
    between the netCDF and the Zarr conversion. If the Zarr assembler changed the
    datasets it was handed, the netCDF view would silently differ from the one the
    netCDF writer builds on its own.
    """
    esm = minimal_test_esM
    esm.optimize()

    netcdf_both, zarr_both = xrIO.writeEnergySystemModelToDatasetsBoth(esm)
    netcdf_alone = xrIO.writeEnergySystemModelToDatasets(esm)

    assert set(netcdf_both) == set(netcdf_alone)
    assert set(zarr_both) == set(netcdf_alone)

    for model, components in netcdf_alone["Input"].items():
        for component, dataset in components.items():
            xr.testing.assert_identical(dataset, netcdf_both["Input"][model][component])


def test_zarr_compressor_encoding_matches_the_installed_zarr():
    """The compressor encoding has to follow the zarr major version.

    zarr 2 takes a numcodecs codec under "compressor". zarr 3 renamed the key to
    "compressors" and rejects a raw numcodecs object. FINE supports both, and CI
    and the cluster environment do not run the same one, so this asserts against
    whichever is installed rather than against a fixed expectation.
    """
    zarr = pytest.importorskip("zarr")

    encoding = xrIO._zarrCompressorEncoding("zstd", 5)

    if int(zarr.__version__.split(".")[0]) >= 3:
        assert list(encoding) == ["compressors"]
        assert len(encoding["compressors"]) == 1
    else:
        assert list(encoding) == ["compressor"]
        assert encoding["compressor"].cname == "zstd"
        assert encoding["compressor"].clevel == 5
