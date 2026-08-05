import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import logging

from fine import utils
from fine.enums import Dimension
from fine.IOManagement import dictIO, utilsIO

logger = logging.getLogger(__name__)


def convertOptimizationInputToDatasets(esM, useProcessedValues=False):
    """Take esM instance input and convert it into xarray datasets.

    :param esM: EnergySystemModel instance in which the model is held
    :type esM: EnergySystemModel instance

    **Default arguments:**

        :param useProcessedValues: True if the raw values should be over-written by processed values, False otherwise.
            A requirement for perfect-foresight and by extension for spatial and technology aggregations
            |br| * the default value is False
        :type useProcessedValues: bool

    :return: xr_ds - esM instance data in xarray dataset format
    :rtype: xarray.dataset
    """
    # STEP 1. Get the esm and component dicts
    esm_dict, component_dict = dictIO.exportToDict(esM, useProcessedValues)

    # STEP 2. get _mapC for all transmission components
    _mapC_dict = {}
    for transmission_class in ["LinearOptimalPowerFlow", "Transmission"]:
        for tech in component_dict[transmission_class].keys():
            _mapC_dict[tech] = esM.getComponent(tech)._mapC

    # STEP 3. Convert component_dict into per-component xarray datasets
    xr_dss = utilsIO.convertComponentDictToXarrayDict(
        component_dict, _mapC_dict, sorted(esm_dict["locations"])
    )

    # STEP 4. Add the data present in esm_dict as xarray attributes
    # (These attributes contain esM init info).
    attributes_xr = xr.Dataset()
    attributes_xr.attrs = esm_dict

    return {"Input": xr_dss, "Parameters": attributes_xr}


def convertPerformanceSummaryToDatasets(esM):  # noqa D103
    df = esM.performanceSummary.squeeze()
    df = df.droplevel("Category")
    df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
    # convert datetime to string
    for idx, value in df.items():
        if isinstance(value, pd.Timestamp):
            logger.debug("Converting timestamp: %s", value)
            df.loc[idx] = value.strftime("%Y-%m-%d %H:%M:%S")
        # a netCDF attribute cannot hold a dict, so write its repr. This is one
        # way: the summary is a record of the run and is never read back into a
        # model, so the dict is not rebuilt on read.
        elif isinstance(value, dict):
            df.loc[idx] = str(value)
    summary_dict = df.to_dict()
    summary_xr = xr.Dataset()
    summary_xr.attrs = summary_dict

    return {"PerformanceSummary": summary_xr}


def convertOptimizationOutputToDatasets(esM, optSumOutputLevel=0):
    """Take esM instance output and convert it into an xarray dataset.

    :param esM: EnergySystemModel instance in which the optimized model is held
    :type esM: EnergySystemModel instance

    :param optSumOutputLevel: Output level of the optimization summary (see
        EnergySystemModel). Either an integer (0,1,2) which holds for all model
        classes or a dictionary with model class names as keys and an integer
        (0,1,2) for each key (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2) or dict

    :return: xr_ds - EnergySystemModel instance output data in xarray dataset format
    :rtype: xarray.dataset
    """
    # Create the netCDF file and the xr.Dataset dict for all ips and components
    xr_dss = dict.fromkeys(esM.investmentPeriodNames)
    for ip in esM.investmentPeriodNames:
        xr_dss[ip] = dict.fromkeys(esM.componentModelingDict.keys())
        for model_dict in esM.componentModelingDict.keys():
            xr_dss[ip][model_dict] = {
                key: xr.Dataset()
                for key in esM.componentModelingDict[model_dict].componentsDict.keys()
            }
    for ip in esM.investmentPeriodNames:
        # Write output from esM.getOptimizationSummary to datasets
        for name in esM.componentModelingDict.keys():
            utils.output("\tProcessing " + name + " ...", esM.verboseLogLevel, 0)
            oL = optSumOutputLevel
            oL_ = oL[name] if isinstance(oL, dict) else oL
            optSum = esM.getOptimizationSummary(name, ip=ip, outputLevel=oL_)
            if esM.componentModelingDict[name].dimension == Dimension.ONE:
                for component in optSum.index.get_level_values(0).unique():
                    variables = optSum.loc[component].index.get_level_values(0)
                    units = optSum.loc[component].index.get_level_values(1)
                    variables_unit = dict(zip(variables, units))
                    for variable in (
                        optSum.loc[component].index.get_level_values(0).unique()
                    ):
                        df = optSum.loc[(component, variable)]
                        df = df.iloc[-1]
                        df.name = variable
                        df.index.rename("location", inplace=True)
                        df = pd.to_numeric(df)
                        xr_da = df.to_xarray()
                        # add variable [e.g. 'TAC'] and units to attributes of xarray
                        unit = variables_unit[variable]
                        xr_da.attrs[variable] = unit

                        # merge to overall xr_ds
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            combine_attrs="drop_conflicts",
                            join="outer",
                        )
            elif esM.componentModelingDict[name].dimension == Dimension.TWO:
                for component in optSum.index.get_level_values(0).unique():
                    variables = optSum.loc[component].index.get_level_values(0)
                    units = optSum.loc[component].index.get_level_values(1)
                    variables_unit = dict(zip(variables, units))
                    for variable in (
                        optSum.loc[component].index.get_level_values(0).unique()
                    ):
                        df = optSum.loc[(component, variable)]
                        if len(df.index.get_level_values(0).unique()) > 1:
                            idx = df.index.get_level_values(0).unique()[-1]
                            df = df.xs(idx, level=0)

                        else:
                            df.index = df.index.droplevel(0)

                        # df = df.iloc[-1]
                        df = df.stack()
                        # df.name = (name, component, variables
                        df.name = variable
                        df.index.rename(["locationIn", "locationOut"], inplace=True)
                        df = pd.to_numeric(df)
                        xr_da = df.to_xarray()

                        # add variable [e.g. 'TAC'] and units to attributes of xarray
                        unit = variables_unit[variable]
                        xr_da.attrs[variable] = unit
                        # merge to overall xr_ds
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            combine_attrs="drop_conflicts",
                            join="outer",
                        )

            # Write output from esM.esM.componentModelingDict[name].getOptimalValues() to datasets
            data = esM.componentModelingDict[name].getOptimalValues(ip=ip)
            dataTD1dim, indexTD1dim, dataTD2dim, indexTD2dim = [], [], [], []
            dataTI, indexTI = [], []

            duplicate_optimum_variables = {
                "capacityVariablesOptimum",
                "commissioningVariablesOptimum",
                "decommissioningVariablesOptimum",
            }
            rename_optimum_variables = {
                "operationVariablesOptimum": "operationTimeSeries",
            }
            for key, d in data.items():
                if key in duplicate_optimum_variables:
                    continue

                if d["values"] is None:
                    continue
                if d["timeDependent"]:
                    if d["dimension"] == Dimension.ONE:
                        dataTD1dim.append(d["values"]), indexTD1dim.append(key)
                    elif d["dimension"] == Dimension.TWO:
                        dataTD2dim.append(d["values"]), indexTD2dim.append(key)
                else:
                    dataTI.append(d["values"]), indexTI.append(key)
            # One dimensional time dependent data
            if dataTD1dim:
                names = ["Variable", "Component", "Location"]
                dfTD1dim = pd.concat(dataTD1dim, keys=indexTD1dim, names=names)
                for variable in dfTD1dim.index.get_level_values(0).unique():
                    # for component in dfTD1dim.index.get_level_values(1).unique():
                    for component in (
                        dfTD1dim.loc[variable].index.get_level_values(0).unique()
                    ):
                        df = dfTD1dim.loc[(variable, component)].T.stack()
                        df.name = rename_optimum_variables.get(variable, variable)
                        df.index.rename(["time", "location"], inplace=True)
                        xr_da = df.to_xarray()
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            join="outer",
                        )
            # Two dimensional time dependent data
            if dataTD2dim:
                names = ["Variable", "Component", "locationIn", "locationOut"]
                dfTD2dim = pd.concat(dataTD2dim, keys=indexTD2dim, names=names)
                for variable in dfTD2dim.index.get_level_values(0).unique():
                    # for component in dfTD2dim.index.get_level_values(1).unique():
                    for component in (
                        dfTD2dim.loc[variable].index.get_level_values(0).unique()
                    ):
                        df = dfTD2dim.loc[(variable, component)].stack()

                        df.name = rename_optimum_variables.get(variable, variable)
                        df.index.rename(
                            ["locationIn", "locationOut", "time"], inplace=True
                        )
                        df.index = df.index.reorder_levels([2, 0, 1])
                        xr_da = df.to_xarray()
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da], join="outer"
                        )
            # Time independent data
            if dataTI:
                # One dimensional
                if esM.componentModelingDict[name].dimension == Dimension.ONE:
                    names = ["Variable type", "Component"]
                    dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                    for variable in dfTI.index.get_level_values(0).unique():
                        # for component in dfTI.index.get_level_values(1).unique():
                        for component in (
                            dfTI.loc[variable].index.get_level_values(0).unique()
                        ):
                            df = dfTI.loc[(variable, component)].T
                            df.name = variable
                            df.index.rename("location", inplace=True)
                            xr_da = df.to_xarray()
                            xr_dss[ip][name][component] = xr.merge(
                                [xr_dss[ip][name][component], xr_da], join="outer"
                            )
                # Two dimensional
                elif esM.componentModelingDict[name].dimension == Dimension.TWO:
                    names = ["Variable type", "Component", "Location"]
                    dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                    for variable in dfTI.index.get_level_values(0).unique():
                        # for component in dfTI.index.get_level_values(1).unique():
                        for component in (
                            dfTI.loc[variable].index.get_level_values(0).unique()
                        ):
                            df = dfTI.loc[(variable, component)].T.stack()
                            df.name = variable
                            df.index.rename(["locationIn", "locationOut"], inplace=True)
                            xr_da = df.to_xarray()
                            xr_dss[ip][name][component] = xr.merge(
                                [xr_dss[ip][name][component], xr_da], join="outer"
                            )

        for name in esM.componentModelingDict.keys():
            for component in esM.componentModelingDict[name].componentsDict.keys():
                if list(xr_dss[ip][name][component].data_vars) == []:
                    # Delete components that have not been built.
                    del xr_dss[ip][name][component]
                elif esM.componentModelingDict[name].dimension == Dimension.TWO:
                    xr_dss[ip][name][component].coords["locationOut"] = (
                        xr_dss[ip][name][component].coords["locationOut"].astype(str)
                    )

                    xr_dss[ip][name][component].coords["locationIn"] = (
                        xr_dss[ip][name][component].coords["locationIn"].astype(str)
                    )
                else:
                    xr_dss[ip][name][component].coords["location"] = (
                        xr_dss[ip][name][component].coords["location"].astype(str)
                    )

    return {"Results": xr_dss}


def serialiseDatasetsForWriting(datasets):
    """Convert the attributes of every dataset in the tree into writable types.

    :func:`writeDatasetsToNetCDF` does this per group as it writes. The folder
    writer hands whole datasets to worker processes, so it has to be done up
    front instead.

    :param datasets: nested dictionary of xarray datasets holding the esM data
    :type datasets: dict

    :return: the same dictionary, with the attributes converted in place
    :rtype: dict
    """
    for group in datasets:
        if group in ("Parameters", "PerformanceSummary"):
            utilsIO.serialiseDatasetAttributes(datasets[group])
    return datasets


# Name of the file that records the folder layout written by
# writeDatasetsToNetCDFfolder, so the reader can rebuild the nested dictionary.
STRUCTURE_FILE_NAME = "structure.json"

# Name of the netCDF file inside each leaf directory.
DATASET_FILE_NAME = "data.nc"


def _saveSingleDataset(task):
    """Write one dataset to one file. Top level, so it can be sent to a worker process.

    :param task: the file path, the dataset, the per variable encoding and the write mode
    :type task: tuple

    :return: the file path that was written
    :rtype: string
    """
    filePath, dataset, encoding, mode = task
    dataset.to_netcdf(
        filePath,
        encoding={var: encoding for var in dataset.data_vars},
        mode=mode,
    )
    return filePath


def writeDatasetsToNetCDFfolder(
    datasets,
    base_path="my_esm",
    compression=True,
    parallel=False,
    chunks=None,
    mode="w",
):
    """Write a nested dictionary of xarray datasets to a folder tree.

    Each dataset goes into its own netCDF file, one directory level per
    dictionary level, with the leaf file named ``data.nc``. A ``structure.json``
    next to them records the layout as paths relative to ``base_path``, which is
    what :func:`readNetCDFfolderToDatasets` reads back.

    Compared with one large netCDF file this writes and reads much faster on a
    parallel file system, because the files are independent.

    **Required arguments:**

    :param datasets: nested dictionary of xarray datasets holding the esM data
    :type datasets: dict

    **Default arguments:**

    :param base_path: directory the tree is written into. It is created if it does not exist.
        |br| * the default value is "my_esm"
    :type base_path: string or pathlib.Path

    :param compression: states if the variables are written with zlib compression
        |br| * the default value is True
    :type compression: boolean

    :param parallel: states if the files are written by a pool of worker processes.
        Worth it for many large datasets, wasteful for a small model.
        |br| * the default value is False
    :type parallel: boolean

    :param chunks: dask chunk sizes applied to every dataset before writing, e.g. {"time": 100}
        |br| * the default value is None
    :type chunks: None or dict

    :param mode: netCDF write mode, "w" to overwrite or "a" to append
        |br| * the default value is "w"
    :type mode: string

    :return: the same nested structure, with the relative file path in place of each dataset
    :rtype: dict
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    encoding = {"zlib": True, "complevel": 5, "shuffle": True} if compression else {}
    save_tasks = []

    def collect_save_tasks(item, current_path):
        if isinstance(item, dict):
            structure = {}
            for key, value in item.items():
                new_path = current_path / str(key)
                new_path.mkdir(exist_ok=True)
                structure[key] = collect_save_tasks(value, new_path)
            return structure

        if isinstance(item, xr.Dataset):
            filePath = current_path / DATASET_FILE_NAME
            if chunks is not None:
                item = item.chunk(chunks)
            save_tasks.append((str(filePath), item, encoding, mode))
            # the structure file holds relative paths, so the tree can be moved
            return str(filePath.relative_to(base_path))

        raise ValueError(
            f"Cannot write an object of type {type(item)} to a netCDF folder. "
            "Only nested dictionaries of xarray datasets are supported."
        )

    datasets = serialiseDatasetsForWriting(datasets)
    structure = collect_save_tasks(datasets, base_path)

    with (base_path / STRUCTURE_FILE_NAME).open("w") as structureFile:
        json.dump(structure, structureFile, indent=2)

    if parallel and save_tasks:
        with ProcessPoolExecutor() as executor:
            list(executor.map(_saveSingleDataset, save_tasks))
    else:
        for task in save_tasks:
            _saveSingleDataset(task)

    return structure


def _loadSingleDataset(path, chunks=None, lazy_load=False):
    """Read one dataset from one file. Top level, so it can be sent to a worker process.

    :param path: path of the netCDF file
    :type path: string

    :param chunks: dask chunk sizes, e.g. {"time": 100}
    :type chunks: None or dict

    :param lazy_load: states if the file stays open and the data is read on demand
    :type lazy_load: boolean

    :return: the dataset
    :rtype: xr.Dataset
    """
    if lazy_load:
        return xr.open_dataset(path, chunks=chunks)
    return xr.load_dataset(path, chunks=chunks)


def _rebuildFolderStructure(item, loaded_datasets):
    """Put the loaded datasets back into the nested structure that was written.

    :param item: the structure as read from structure.json
    :type item: dict or string

    :param loaded_datasets: the datasets, keyed by their relative path
    :type loaded_datasets: dict

    :return: the nested dictionary of datasets
    :rtype: dict or xr.Dataset
    """
    if isinstance(item, dict):
        return {
            key: _rebuildFolderStructure(value, loaded_datasets)
            for key, value in item.items()
        }
    if isinstance(item, str):
        return loaded_datasets[item]
    raise ValueError(f"Unsupported entry in {STRUCTURE_FILE_NAME}: {type(item)}")


def readNetCDFfolderToDatasets(base_path, parallel=True, chunks=None, lazy_load=False):
    """Read back a folder tree written by :func:`writeDatasetsToNetCDFfolder`.

    **Required arguments:**

    :param base_path: directory holding the tree and its structure.json
    :type base_path: string or pathlib.Path

    **Default arguments:**

    :param parallel: states if the files are read by a pool of worker processes.
        Not compatible with lazy_load, which needs the files to stay open in this process.
        |br| * the default value is True
    :type parallel: boolean

    :param chunks: dask chunk sizes, e.g. {"time": 100}
        |br| * the default value is None
    :type chunks: None or dict

    :param lazy_load: states if the data is read on demand instead of at once. Keeps the
        memory use down for a model that does not fit in memory.
        |br| * the default value is False
    :type lazy_load: boolean

    :return: the nested dictionary of datasets that was written
    :rtype: dict
    """
    base_path = Path(base_path)

    with (base_path / STRUCTURE_FILE_NAME).open() as structureFile:
        structure = json.load(structureFile)

    paths_to_load = []

    def collect_all_paths(item):
        if isinstance(item, dict):
            for value in item.values():
                collect_all_paths(value)
        elif isinstance(item, str):
            paths_to_load.append(str(base_path / item))

    collect_all_paths(structure)

    load_fn = partial(_loadSingleDataset, chunks=chunks, lazy_load=lazy_load)

    if lazy_load and parallel:
        # a lazily read dataset keeps a file handle in the process that opened it,
        # so it cannot be handed back from a worker process
        logger.debug("lazy_load is set, reading in this process instead of in parallel")
        parallel = False

    if parallel and paths_to_load:
        with ProcessPoolExecutor() as executor:
            results = executor.map(load_fn, paths_to_load)
            loaded_datasets = dict(
                zip(
                    (str(Path(path).relative_to(base_path)) for path in paths_to_load),
                    results,
                )
            )
    else:
        loaded_datasets = {
            str(Path(path).relative_to(base_path)): load_fn(path)
            for path in paths_to_load
        }

    return _rebuildFolderStructure(structure, loaded_datasets)


def writeDatasetsToNetCDF(
    datasets,
    outputFilePath="my_esm.nc",
    removeExisting=False,
    mode="a",
    groupPrefix=None,
):
    """Save dictionary of xarray datasets (with esM instance data) to a netCDF
    file.

    **Required arguments:**

    :param datasets: The xarray datasets holding all data required to set up an esM instance.
    :type datasets: Dict[xr.Dataset]

    **Default arguments:**

    :param outputFilePath: output file name of the netCDF file (can include full path)
        |br| * the default value is "my_esm.nc"
    :type outputFilePath: string

    :param removeExisting: indicates if an existing netCDF file should be removed
        |br| * the default value is False
    :type removeExisting: boolean

    :param mode: Write (‘w’) or append (‘a’) mode.

        * If mode=’w’, any existing file at this location will be overwritten.
        * If mode=’a’, existing variables will be overwritten.

        |br| * the default value is 'a'
    :type mode: string

    :param groupPrefix: if specified, multiple xarray datasets (with esM
        instance data) are saved to the same netcdf file. The dictionary
        structure is then {group_prefix}/{group}/{...} instead of {group}/{...}
        |br| * the default value is None
    :type groupPrefix: string

    """
    # Create netCDF file, remove existant
    if removeExisting:
        if Path(outputFilePath).is_file():
            Path(outputFilePath).unlink()

    if not Path(outputFilePath).is_file():
        with Dataset(outputFilePath, "w", format="NETCDF4") as _:
            pass

    for group in datasets.keys():
        if group in ("Parameters", "PerformanceSummary"):
            xarray_dataset = datasets[group]
            utilsIO.serialiseDatasetAttributes(xarray_dataset)

            if groupPrefix:
                group_path = f"{groupPrefix}/{group}"
            else:
                group_path = f"{group}"

            xarray_dataset.to_netcdf(
                path=f"{outputFilePath}",
                # Datasets per component will be reflectes as groups in the NetCDF file.
                group=group_path,
                # Use mode='a' to append datasets to existing file. Variables will be overwritten.
                mode=mode,
            )

        elif group == "Results":
            for ip in datasets[group].keys():
                for model, comps in datasets[group][ip].items():
                    for component in comps.keys():
                        if component is not None:
                            if groupPrefix:
                                group_path = (
                                    f"{groupPrefix}/{group}/{ip}/{model}/{component}"
                                )
                            else:
                                group_path = f"{group}/{ip}/{model}/{component}"
                            datasets[group][ip][model][component].to_netcdf(
                                path=f"{outputFilePath}",
                                # Datasets per component will be reflectes as groups in the NetCDF file.
                                group=group_path,
                                # Use mode='a' to append datasets to existing file. Variables will be overwritten.
                                mode=mode,
                                # Use zlib variable compression to reduce filesize with little performance loss
                                # for our use-case. Complevel 9 for best compression.
                                encoding={
                                    var: {"zlib": True, "complevel": 5}
                                    for var in list(
                                        datasets[group][ip][model][component].data_vars
                                    )
                                },
                            )
        else:
            for model, comps in datasets[group].items():
                for component in comps.keys():
                    if component is not None:
                        if groupPrefix:
                            group_path = f"{groupPrefix}/{group}/{model}/{component}"
                        else:
                            group_path = f"{group}/{model}/{component}"
                        datasets[group][model][component].to_netcdf(
                            path=f"{outputFilePath}",
                            # Datasets per component will be reflectes as groups in the NetCDF file.
                            group=group_path,
                            # Use mode='a' to append datasets to existing file. Variables will be overwritten.
                            mode=mode,
                            # Use zlib variable compression to reduce filesize with little performance loss
                            # for our use-case. Complevel 9 for best compression.
                            encoding={
                                var: {"zlib": True, "complevel": 5}
                                for var in list(
                                    datasets[group][model][component].data_vars
                                )
                            },
                        )


def convertDatasetsToEnergySystemModel(datasets):
    """Take dictionary of xarray datasets (with esM instance data) and convert
    it to an esM instance.

    :param datasets: The xarray datasets holding all data required to set up an esM instance.
    :type datasets: Dict[xr.Dataset]

    :return: esM - EnergySystemModel instance
    :rtype: EnergySystemModel instance
    """
    # Read parameters
    xarray_dataset = utilsIO.processXarrayAttributes(datasets["Parameters"])
    esm_dict = xarray_dataset.attrs

    # Read input
    # Iterate through each component-variable pair, depending on the variable's
    # prefix restructure the data and add it to component_dict
    component_dict = utilsIO.PowerDict()

    for model, comps in datasets["Input"].items():
        for component_name, comp_xr in comps.items():
            for variable, comp_var_xr in comp_xr.data_vars.items():
                if not pd.isnull(comp_var_xr.values).all():  # Skip if all are NAs
                    component = f"{model}; {component_name}"

                    # STEP 4 (i). Set regional time series (region, time)
                    if variable[:3] == "ts_":
                        component_dict = utilsIO.addTimeSeriesVariableToDict(
                            component_dict,
                            comp_var_xr,
                            component,
                            variable,
                            drop_component=False,
                        )

                    # STEP 4 (ii). Set 2d data (region, region)
                    elif variable[:3] == "2d_":
                        component_dict = utilsIO.add2dVariableToDict(
                            component_dict,
                            comp_var_xr,
                            component,
                            variable,
                            drop_component=False,
                        )

                    # STEP 4 (iii). Set 1d data (region)
                    elif variable[:3] == "1d_":
                        component_dict = utilsIO.add1dVariableToDict(
                            component_dict,
                            comp_var_xr,
                            component,
                            variable,
                            drop_component=False,
                        )

                    # STEP 4 (iv). Set 0d data
                    elif variable[:3] == "0d_":
                        component_dict = utilsIO.add0dVariableToDict(
                            component_dict, comp_var_xr, component, variable
                        )

    # Create esm from esm_dict and component_dict
    esM = dictIO.importFromDict(esm_dict, component_dict)

    # Read output
    if "Results" in datasets:
        # get startyear to find model classes
        startyear = list(datasets["Results"].keys())[0]
        for model, comps in datasets["Results"][startyear].items():
            componentModel = esM.componentModelingDict[model]
            optSum = {}
            operationVariablesOptimum_dict = {}
            capacityVariablesOptimum_dict = {}
            isBuiltVariablesOptimum_dict = {}
            commissioningVariablesOptimum_dict = {}
            decommissioningVariablesOptimum_dict = {}
            chargeOperationVariablesOptimum_dict = {}
            dischargeOperationVariablesOptimum_dict = {}
            stateOfChargeOperationVariablesOptimum_dict = {}

            # variables that only hold optimum values (no corresponding
            # optSummary property), even though their name doesn't contain
            # "Optimum" (renamed to avoid duplicate data in the datasets)
            optimum_only_variables = {"operationTimeSeries"}

            for ip in datasets["Results"].keys():
                # read opt Summary
                optSum_df = pd.DataFrame([])
                for component in datasets["Results"][ip][model]:
                    optSum_df_comp = pd.DataFrame([])
                    for variable in datasets["Results"][ip][model][component]:
                        if "Optimum" in variable or variable in optimum_only_variables:
                            continue
                        if "locationOut" in list(
                            datasets["Results"][ip][model][component].coords
                        ):
                            _optSum_df = (
                                datasets["Results"][ip][model][component][variable]
                                .to_dataframe()
                                .unstack()
                            )
                            iterables = [
                                [component, variable, unit]
                                for variable, unit in datasets["Results"][ip][model][
                                    component
                                ][variable].attrs.items()
                            ]
                            iterables2 = [
                                [iterables[0] + [location]][0]
                                for location in datasets["Results"][ip][model][
                                    component
                                ][variable]["locationIn"].values
                            ]
                            idx = pd.MultiIndex.from_tuples(tuple(iterables2))
                            _optSum_df.index = idx
                            _optSum_df.index.set_names(
                                names=[
                                    "Component",
                                    "Property",
                                    "Unit",
                                    "locationIn",
                                ],
                                inplace=True,
                            )
                            _optSum_df = _optSum_df.droplevel(0, axis=1)
                            if isinstance(_optSum_df, pd.Series):
                                _optSum_df = _optSum_df.to_frame().T
                            optSum_df_comp = pd.concat(
                                [optSum_df_comp, _optSum_df],
                                axis=0,
                            )

                        else:
                            _optSum_df = (
                                datasets["Results"][ip][model][component][variable]
                                .to_dataframe()
                                .T
                            )
                            iterables = [
                                [component, variable, unit]
                                for variable, unit in datasets["Results"][ip][model][
                                    component
                                ][variable].attrs.items()
                            ]
                            _optSum_df.index = pd.MultiIndex.from_tuples(
                                iterables, names=["Component", "Property", "Unit"]
                            )

                            if isinstance(_optSum_df, pd.Series):
                                _optSum_df = _optSum_df.to_frame().T
                            optSum_df_comp = pd.concat(
                                [optSum_df_comp, _optSum_df],
                                axis=0,
                            )

                    if isinstance(optSum_df_comp, pd.Series):
                        optSum_df_comp = optSum_df_comp.to_frame().T
                    optSum_df = pd.concat(
                        [optSum_df, optSum_df_comp],
                        axis=0,
                    )
                optSum[int(ip)] = optSum_df

                componentModel._optSummary = optSum

                # read optimal Values (3 types exist)
                operationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                capacityVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                isBuiltVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                commissioningVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                decommissioningVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                chargeOperationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                dischargeOperationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                stateOfChargeOperationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])

                for component in datasets["Results"][ip][model]:
                    _operationVariablesOptimum_df = pd.DataFrame([])
                    _capacityVariablesOptimum_df = pd.DataFrame([])
                    _isBuiltVariablesOptimum_df = pd.DataFrame([])
                    _commissioningVariablesOptimum_df = pd.DataFrame([])
                    _decommissioningVariablesOptimum_df = pd.DataFrame([])
                    _chargeOperationVariablesOptimum_df = pd.DataFrame([])
                    _dischargeOperationVariablesOptimum_df = pd.DataFrame([])
                    _stateOfChargeOperationVariablesOptimum_df = pd.DataFrame([])

                    summary_optimum_mapping = {
                        "capacity": "capacityVariablesOptimum",
                        "commissioning": "commissioningVariablesOptimum",
                        "decommissioning": "decommissioningVariablesOptimum",
                        "operationTimeSeries": "operationVariablesOptimum",
                    }

                    for variable in datasets["Results"][ip][model][component]:
                        if (
                            "Optimum" not in variable
                            and variable not in summary_optimum_mapping
                        ):
                            continue

                        opt_variable = summary_optimum_mapping.get(variable, variable)
                        xr_opt = datasets["Results"][ip][model][component][variable]

                        if opt_variable == "operationVariablesOptimum":
                            if "locationOut" in list(xr_opt.coords):
                                df = (
                                    xr_opt.to_dataframe()
                                    .unstack(level=0)
                                    .droplevel(0, axis=1)
                                )
                                _operationVariablesOptimum_df = pd.DataFrame([])
                                for item in df.index.get_level_values(0).unique():
                                    _df = df.loc[item]
                                    _df = _df.drop(item)
                                    idx = pd.MultiIndex.from_product(
                                        [[component], [item], list(_df.index)]
                                    )
                                    _df = _df.set_index(idx)
                                    if isinstance(_df, pd.Series):
                                        _df = _df.to_frame().T
                                    _operationVariablesOptimum_df = pd.concat(
                                        [_operationVariablesOptimum_df, _df],
                                        axis=0,
                                    ).rename_axis(len(idx.names) * [None], axis=0)

                            else:
                                _operationVariablesOptimum_df = (
                                    xr_opt.to_dataframe()
                                    .unstack(level=0)
                                    .droplevel(0, axis=1)
                                )
                                _operationVariablesOptimum_df = (
                                    _operationVariablesOptimum_df.dropna(axis=0)
                                )
                                idx = pd.MultiIndex.from_product(
                                    [[component], _operationVariablesOptimum_df.index]
                                )
                                _operationVariablesOptimum_df = (
                                    _operationVariablesOptimum_df.set_index(idx)
                                ).rename_axis(len(idx.names) * [None], axis=0)

                        if opt_variable == "capacityVariablesOptimum":
                            if "locationOut" in list(xr_opt.coords):
                                df = (
                                    xr_opt.to_dataframe()
                                    .unstack(level=0)
                                    .droplevel(0, axis=1)
                                )
                                idx = pd.MultiIndex.from_product(
                                    [[component], list(df.index)]
                                )
                                _df = df.set_index(idx)
                                _capacityVariablesOptimum_df = _df.rename_axis(
                                    None, axis=1
                                )
                            else:
                                _capacityVariablesOptimum_df = xr_opt.to_dataframe().T
                                _capacityVariablesOptimum_df = (
                                    _capacityVariablesOptimum_df.set_axis([component])
                                ).rename_axis(None, axis=1)

                        if opt_variable == "isBuiltVariablesOptimum":
                            _isBuiltVariablesOptimum_df = (
                                xr_opt.to_dataframe()
                                .unstack(level=0)
                                .droplevel(0, axis=1)
                            )
                            idx = pd.MultiIndex.from_product(
                                [[component], _isBuiltVariablesOptimum_df.index]
                            )
                            _isBuiltVariablesOptimum_df = (
                                _isBuiltVariablesOptimum_df.set_index(idx)
                            ).rename_axis(None, axis=1)
                        if opt_variable == "commissioningVariablesOptimum":
                            if "locationOut" in list(xr_opt.coords):
                                df = (
                                    xr_opt.to_dataframe()
                                    .unstack(level=0)
                                    .droplevel(0, axis=1)
                                )
                                idx = pd.MultiIndex.from_product(
                                    [[component], list(df.index)]
                                )
                                _df = df.set_index(idx)
                                _commissioningVariablesOptimum_df = _df.rename_axis(
                                    None, axis=1
                                )
                            else:
                                _commissioningVariablesOptimum_df = (
                                    xr_opt.to_dataframe().T
                                )
                                _commissioningVariablesOptimum_df = (
                                    _commissioningVariablesOptimum_df.set_axis(
                                        [component]
                                    )
                                ).rename_axis(None, axis=1)
                        if opt_variable == "decommissioningVariablesOptimum":
                            if "locationOut" in list(xr_opt.coords):
                                df = (
                                    xr_opt.to_dataframe()
                                    .unstack(level=0)
                                    .droplevel(0, axis=1)
                                )
                                idx = pd.MultiIndex.from_product(
                                    [[component], list(df.index)]
                                )
                                _df = df.set_index(idx)
                                _decommissioningVariablesOptimum_df = _df.rename_axis(
                                    None, axis=1
                                )
                            else:
                                _decommissioningVariablesOptimum_df = (
                                    xr_opt.to_dataframe().T
                                )
                                _decommissioningVariablesOptimum_df = (
                                    _decommissioningVariablesOptimum_df.set_axis(
                                        [component]
                                    )
                                ).rename_axis(None, axis=1)

                        if opt_variable == "chargeOperationVariablesOptimum":
                            _chargeOperationVariablesOptimum_df = (
                                xr_opt.to_dataframe()
                                .unstack(level=0)
                                .droplevel(0, axis=1)
                            )
                            idx = pd.MultiIndex.from_product(
                                [[component], _chargeOperationVariablesOptimum_df.index]
                            )
                            _chargeOperationVariablesOptimum_df = (
                                (_chargeOperationVariablesOptimum_df.set_index(idx))
                                .rename_axis(len(idx.names) * [None], axis=0)
                                .rename_axis(None, axis=1)
                            )

                        if opt_variable == "dischargeOperationVariablesOptimum":
                            _dischargeOperationVariablesOptimum_df = (
                                xr_opt.to_dataframe()
                                .unstack(level=0)
                                .droplevel(0, axis=1)
                            )
                            idx = pd.MultiIndex.from_product(
                                [
                                    [component],
                                    _dischargeOperationVariablesOptimum_df.index,
                                ]
                            )
                            _dischargeOperationVariablesOptimum_df = (
                                (_dischargeOperationVariablesOptimum_df.set_index(idx))
                                .rename_axis(len(idx.names) * [None], axis=0)
                                .rename_axis(None, axis=1)
                            )

                        if opt_variable == "stateOfChargeOperationVariablesOptimum":
                            _stateOfChargeOperationVariablesOptimum_df = (
                                xr_opt.to_dataframe()
                                .unstack(level=0)
                                .droplevel(0, axis=1)
                            )
                            idx = pd.MultiIndex.from_product(
                                [
                                    [component],
                                    _stateOfChargeOperationVariablesOptimum_df.index,
                                ]
                            )
                            _stateOfChargeOperationVariablesOptimum_df = (
                                (
                                    _stateOfChargeOperationVariablesOptimum_df.set_index(
                                        idx
                                    )
                                )
                                .rename_axis(len(idx.names) * [None], axis=0)
                                .rename_axis(None, axis=1)
                            )
                    if isinstance(_operationVariablesOptimum_df, pd.Series):
                        _operationVariablesOptimum_df = (
                            _operationVariablesOptimum_df.to_frame().T
                        )
                    operationVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            operationVariablesOptimum_dict[int(ip)],
                            _operationVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(_capacityVariablesOptimum_df, pd.Series):
                        _capacityVariablesOptimum_df = (
                            _capacityVariablesOptimum_df.to_frame().T
                        )
                    capacityVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            capacityVariablesOptimum_dict[int(ip)],
                            _capacityVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(_isBuiltVariablesOptimum_df, pd.Series):
                        _isBuiltVariablesOptimum_df = (
                            _isBuiltVariablesOptimum_df.to_frame().T
                        )
                    isBuiltVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            isBuiltVariablesOptimum_dict[int(ip)],
                            _isBuiltVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(_commissioningVariablesOptimum_df, pd.Series):
                        _commissioningVariablesOptimum_df = (
                            _commissioningVariablesOptimum_df.to_frame().T
                        )
                    commissioningVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            commissioningVariablesOptimum_dict[int(ip)],
                            _commissioningVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(_decommissioningVariablesOptimum_df, pd.Series):
                        _decommissioningVariablesOptimum_df = (
                            _decommissioningVariablesOptimum_df.to_frame().T
                        )
                    decommissioningVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            decommissioningVariablesOptimum_dict[int(ip)],
                            _decommissioningVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(_chargeOperationVariablesOptimum_df, pd.Series):
                        _chargeOperationVariablesOptimum_df = (
                            _chargeOperationVariablesOptimum_df.to_frame().T
                        )
                    chargeOperationVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            chargeOperationVariablesOptimum_dict[int(ip)],
                            _chargeOperationVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(_dischargeOperationVariablesOptimum_df, pd.Series):
                        _dischargeOperationVariablesOptimum_df = (
                            _dischargeOperationVariablesOptimum_df.to_frame().T
                        )
                    dischargeOperationVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            dischargeOperationVariablesOptimum_dict[int(ip)],
                            _dischargeOperationVariablesOptimum_df,
                        ],
                        axis=0,
                    )
                    if isinstance(
                        _stateOfChargeOperationVariablesOptimum_df, pd.Series
                    ):
                        _stateOfChargeOperationVariablesOptimum_df = (
                            _stateOfChargeOperationVariablesOptimum_df.to_frame().T
                        )
                    stateOfChargeOperationVariablesOptimum_dict[int(ip)] = pd.concat(
                        [
                            stateOfChargeOperationVariablesOptimum_dict[int(ip)],
                            _stateOfChargeOperationVariablesOptimum_df,
                        ],
                        axis=0,
                    )

                # check if empty, if yes convert to None
                if operationVariablesOptimum_dict[int(ip)].empty:
                    operationVariablesOptimum_dict[int(ip)] = None
                if capacityVariablesOptimum_dict[int(ip)].empty:
                    capacityVariablesOptimum_dict[int(ip)] = None
                if isBuiltVariablesOptimum_dict[int(ip)].empty:
                    isBuiltVariablesOptimum_dict[int(ip)] = None
                if commissioningVariablesOptimum_dict[int(ip)].empty:
                    commissioningVariablesOptimum_dict[int(ip)] = None
                if decommissioningVariablesOptimum_dict[int(ip)].empty:
                    decommissioningVariablesOptimum_dict[int(ip)] = None
                if chargeOperationVariablesOptimum_dict[int(ip)].empty:
                    chargeOperationVariablesOptimum_dict[int(ip)] = None
                if dischargeOperationVariablesOptimum_dict[int(ip)].empty:
                    dischargeOperationVariablesOptimum_dict[int(ip)] = None
                if stateOfChargeOperationVariablesOptimum_dict[int(ip)].empty:
                    stateOfChargeOperationVariablesOptimum_dict[int(ip)] = None

            componentModel._operationVariablesOptimum = operationVariablesOptimum_dict
            componentModel._capacityVariablesOptimum = capacityVariablesOptimum_dict
            componentModel._isBuiltVariablesOptimum = isBuiltVariablesOptimum_dict
            componentModel._commissioningVariablesOptimum = (
                commissioningVariablesOptimum_dict
            )
            componentModel._decommissioningVariablesOptimum = (
                decommissioningVariablesOptimum_dict
            )
            componentModel._chargeOperationVariablesOptimum = (
                chargeOperationVariablesOptimum_dict
            )
            componentModel._dischargeOperationVariablesOptimum = (
                dischargeOperationVariablesOptimum_dict
            )
            componentModel._stateOfChargeOperationVariablesOptimum = (
                stateOfChargeOperationVariablesOptimum_dict
            )

            # if only one investment period -> keep optimal values unchanged for end user
            def setFinalOptimalValues(esM, name):
                if len(esM.investmentPeriodNames) == 1:
                    data = getattr(componentModel, "_" + name)
                    setattr(componentModel, name, data[int(startyear)])
                else:
                    data = getattr(componentModel, "_" + name)
                    setattr(componentModel, name, data)
                return esM

            optimalParameters = [
                "optSummary",
                "operationVariablesOptimum",
                "capacityVariablesOptimum",
                "commissioningVariablesOptimum",
                "decommissioningVariablesOptimum",
                "isBuiltVariablesOptimum",
                "chargeOperationVariablesOptimum",
                "dischargeOperationVariablesOptimum",
                "stateOfChargeOperationVariablesOptimum",
            ]
            for name in optimalParameters:
                esM = setFinalOptimalValues(esM, name)

    return esM


def writeEnergySystemModelToNetCDF(
    esM,
    outputFilePath="my_esm.nc",
    overwriteExisting=False,
    optSumOutputLevel=0,
    groupPrefix=None,
    includeShadowPrices=False,
    shadowPriceConstraintStr="commodityBalanceConstraint",
):
    """Write energySystemModel (input and if exists, output) to netCDF file.

    :param esM: EnergySystemModel instance in which the model is held
    :type esM: EnergySystemModel instance

    :param outputFilePath: output file name (can include full path)
        |br| * the default value is "my_esm.nc"
    :type file_path: string

    :param overwriteExisting: Overwrite existing netCDF file
        |br| * the default value is False
    :type outputFileName: boolean

    :param optSumOutputLevel: Output level of the optimization summary (see
        EnergySystemModel). Either an integer (0,1,2) which holds for all model
        classes or a dictionary with model class names as keys and an integer
        (0,1,2) for each key (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2) or dict

    :param groupPrefix: if specified, multiple xarray datasets (with esM
        instance data) are saved to the same netcdf file. The dictionary
        structure is then {group_prefix}/{group}/{...} instead of {group}/{...}
        |br| * the default value is None
    :type group_prefix: string

    :param includeShadowPrices: Whether to include shadow prices in the output netCDF file.
        |br| * the default value is False
    :type includeShadowPrices: boolean

    :param shadowPriceConstraintStr: The string to identify the constraints for which shadow prices should be included.
        |br| * the default value is "commodityBalanceConstraint"
    :type shadowPriceConstraintStr: string

    :return: Nested dictionary containing xr.Dataset with all result values
        for each component.
    :rtype: Dict[str, Dict[str, xr.Dataset]]
    """
    if overwriteExisting:
        if Path(outputFilePath).is_file():
            Path(outputFilePath).unlink()

    utils.output("\nWriting output to netCDF... ", esM.verboseLogLevel, 0)
    _t = time.time()

    xr_dss_input = convertOptimizationInputToDatasets(esM)
    writeDatasetsToNetCDF(xr_dss_input, outputFilePath, groupPrefix=groupPrefix)
    if esM.objectiveValue is not None:  # model was optimized
        xr_dss_output = convertOptimizationOutputToDatasets(esM, optSumOutputLevel)
        if "performanceSummary" in vars(esM):
            xr_dss_performance = convertPerformanceSummaryToDatasets(esM)
            xr_dss_output["PerformanceSummary"] = xr_dss_performance[
                "PerformanceSummary"
            ]
        if includeShadowPrices:
            xr_dss_shadowPrices = utilsIO.getShadowPriceXarray(
                esM, constraint_str=shadowPriceConstraintStr
            )
            xr_dss_output["ShadowPrices"] = xr_dss_shadowPrices
        logger.debug("Output datasets keys: %s", list(xr_dss_output.keys()))
        writeDatasetsToNetCDF(xr_dss_output, outputFilePath, groupPrefix=groupPrefix)

    utils.output("Done. (%.4f" % (time.time() - _t) + " sec)", esM.verboseLogLevel, 0)


def convertEnergySystemModelToDatasets(
    esM,
    optSumOutputLevel=0,
    includeShadowPrices=False,
    shadowPriceConstraintStr="commodityBalanceConstraint",
):
    """Convert an esM instance (input and, if it was optimized, output) into datasets.

    This is the one export. Every writer starts here, and every storage format is
    built from the result, so a model has one canonical layout whatever it is
    written to.

    **Required arguments:**

    :param esM: EnergySystemModel instance in which the model is held
    :type esM: EnergySystemModel instance

    **Default arguments:**

    :param optSumOutputLevel: Output level of the optimization summary (see
        EnergySystemModel). Either an integer (0,1,2) which holds for all model
        classes or a dictionary with model class names as keys and an integer
        (0,1,2) for each key (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        |br| * the default value is 0
    :type optSumOutputLevel: int (0,1,2) or dict

    :param includeShadowPrices: states if the shadow prices are part of the result
        |br| * the default value is False
    :type includeShadowPrices: boolean

    :param shadowPriceConstraintStr: name of the constraint the shadow prices are
        taken from
        |br| * the default value is "commodityBalanceConstraint"
    :type shadowPriceConstraintStr: string

    :return: the esM data in xarray dataset format, keyed by group
    :rtype: dict
    """
    datasets = convertOptimizationInputToDatasets(esM)

    if esM.objectiveValue is not None:  # model was optimized
        datasets.update(convertOptimizationOutputToDatasets(esM, optSumOutputLevel))
        if "performanceSummary" in vars(esM):
            datasets.update(convertPerformanceSummaryToDatasets(esM))
        if includeShadowPrices:
            datasets["ShadowPrices"] = utilsIO.getShadowPriceXarray(
                esM, constraint_str=shadowPriceConstraintStr
            )

    return datasets


def writeEnergySystemModelToDatasets(
    esM,
    includeShadowPrices=False,
    shadowPriceConstraintStr="commodityBalanceConstraint",
):
    """Convert esM instance (input and output) into a xarray dataset.

    Deprecated. Call :func:`convertEnergySystemModelToDatasets`, which carries the
    optimization summary output level as well.

    :param esM: EnergySystemModel instance in which the optimized model is held
    :type esM: EnergySystemModel instance

    :param includeShadowPrices: Whether to include shadow prices in the output xarray dataset.
        |br| * the default value is False
    :type includeShadowPrices: boolean

    :param shadowPriceConstraintStr: The string to identify the constraints for which shadow prices should be included.
        |br| * the default value is "commodityBalanceConstraint"
    :type shadowPriceConstraintStr: string

    :return: xr_dss_results - esM instance (input and output) data in xarray
        dataset format
    :rtype: xr.DataSet
    """
    return convertEnergySystemModelToDatasets(
        esM,
        includeShadowPrices=includeShadowPrices,
        shadowPriceConstraintStr=shadowPriceConstraintStr,
    )


def readNetCDFToDatasets(filePath="my_esm.nc", groupPrefix=None, lazy_load=False):
    """Read optimization results from grouped netCDF file to dictionary of
    xr.Datasets.

    :param filePath: output file name of netCDF file (can include full path)
        |br| * the default value is "my_esm.nc"
    :type filePath: string

    :param groupPrefix: if specified, multiple xarray datasets (with esM
        instance data) are saved to the same netcdf file. The dictionary
        structure is then {group_prefix}/{group}/{...} instead of {group}/{...}
        |br| * the default value is None
    :type groupPrefix: string

    :param lazy_load: If True, the data is not loaded into memory until it is
        accessed. This can be useful for large datasets. Refer to xarray documentation for more information
        |br| * the default value is False
    :type lazy_load: boolean

    :return: Nested dictionary containing an xr.Dataset with all result values
        for each component.
    :rtype: Nested dict
    """
    with Dataset(filePath, "r", format="NETCDF4") as rootgrp:
        if groupPrefix:
            group_keys = rootgrp[groupPrefix].groups
        else:
            group_keys = rootgrp.groups

    if lazy_load:
        loader = xr.open_dataset
    else:
        loader = xr.load_dataset

    if not groupPrefix:
        xr_dss = {}
        # read input from netcdf
        xr_dss["Input"] = {
            model_key: {
                comp_key: loader(filePath, group=f"Input/{model_key}/{comp_key}")
                for comp_key in group_keys["Input"][model_key].groups
            }
            for model_key in group_keys["Input"].groups
        }
        # read results from netcdf
        if "Results" in group_keys:
            xr_dss["Results"] = {
                ip_key: {
                    model_key: {
                        comp_key: loader(
                            filePath, group=f"Results/{ip_key}/{model_key}/{comp_key}"
                        )
                        for comp_key in group_keys["Results"][ip_key][model_key].groups
                    }
                    for model_key in group_keys["Results"][ip_key].groups
                }
                for ip_key in group_keys["Results"].groups
            }
        # read parameters from netcdf
        xr_dss["Parameters"] = xr.load_dataset(filePath, group="Parameters")
        # read performance summary from netcdf (if exists)
        if "PerformanceSummary" in group_keys:
            xr_dss["PerformanceSummary"] = xr.load_dataset(
                filePath, group="PerformanceSummary"
            )
    else:
        xr_dss = {}
        # read input from netcdf
        xr_dss["Input"] = {
            model_key: {
                comp_key: loader(
                    filePath,
                    group=f"{groupPrefix}/Input/{model_key}/{comp_key}",
                )
                for comp_key in group_keys["Input"][model_key].groups
            }
            for model_key in group_keys["Input"].groups
        }
        # read results from netcdf
        if "Results" in group_keys:
            xr_dss["Results"] = {
                ip_key: {
                    model_key: {
                        comp_key: loader(
                            filePath,
                            group=f"{groupPrefix}/Results/{ip_key}/{model_key}/{comp_key}",
                        )
                        for comp_key in group_keys["Results"][ip_key][model_key].groups
                    }
                    for model_key in group_keys["Results"][ip_key].groups
                }
                for ip_key in group_keys["Results"].groups
            }
        # read parameters from netcdf
        xr_dss["Parameters"] = xr.load_dataset(
            filePath, group=f"{groupPrefix}/Parameters"
        )
        # read performance summary from netcdf (if exists)
        if "PerformanceSummary" in group_keys:
            xr_dss["PerformanceSummary"] = xr.load_dataset(
                filePath, group=f"{groupPrefix}/PerformanceSummary"
            )

    return xr_dss


def readNetCDFToEnergySystemModel(filePath, groupPrefix=None):
    """Convert netCDF file into an EnergySystemModel instance.

    :param filePath: file name of netCDF file (can include full path) in which
        the esM data is stored
        |br| * the default value is "my_esm.nc"
    :type filePath: string

    :return: EnergySystemModel instance
    :rtype: EnergySystemModel instance
    """
    # netcdf to xarray dataset
    xr_dss = readNetCDFToDatasets(filePath, groupPrefix)

    # xarray dataset to esm
    return convertDatasetsToEnergySystemModel(xr_dss)


# the older spelling, with the lower case "to". It is alone among all of them, so
# the name above is the one to use, and this stays for the callers that have it.
readNetCDFtoEnergySystemModel = readNetCDFToEnergySystemModel


def _make_datasets_lazy(data_dict, chunks="auto", _path=()):
    """Turn the in-memory arrays of a nested dataset dictionary into dask arrays.

    ``chunks`` must never be None. ``Dataset.chunk(None)`` does not pick a sensible
    size, it puts the whole array into a single dask chunk. For a reconstructed
    time series that chunk can pass 2 GB, which the Blosc codec refuses, and a
    single chunk also rules out a streamed write. "auto" bounds a chunk by dask's
    target size, 128 MiB by default.

    ``_path`` is the dictionary key path down to the current dataset, and it is
    passed to ``chunk`` as the ``token``. Without a token, xarray names each dask
    array by hashing the whole array content, sha1 over every byte, which dominates
    the save time for large result arrays that are already in memory. A dictionary
    path is unique, so it is as safe a name as a content hash, and it costs nothing
    to compute.

    :param data_dict: nested dictionary of xarray datasets
    :type data_dict: dict

    :param chunks: dask chunk sizes, or "auto"
    :type chunks: string or dict

    :param _path: key path to the current dataset, used to build the dask token
    :type _path: tuple

    :return: the same structure, with every dataset chunked
    :rtype: dict
    """
    if isinstance(data_dict, dict):
        return {
            key: _make_datasets_lazy(value, chunks, _path=(*_path, str(key)))
            for key, value in data_dict.items()
        }

    if isinstance(data_dict, xr.Dataset):
        try:
            return data_dict.chunk(chunks, token="/".join(_path) if _path else None)
        except (ValueError, TypeError, NotImplementedError) as error:
            logger.warning("Could not chunk the dataset at %s: %s", _path, error)
            return data_dict

    return data_dict


# Chunk sizes used when a dataset is written to Zarr.
#
# Zarr needs a uniform chunk size along every axis except in the last chunk, and
# a dataset that comes out of a concatenation carries the ragged chunks of that
# concatenation. Every dimension therefore has to be chunked, not only the ones
# named here: the two mask variables are indexed by "parameter", which appears in
# no scheme, and leaving them unchunked produced chunks like ((1,)*42, (49,1,1,19)).
#
# "auto" rather than -1 for the spatial and component axes: -1 keeps a whole
# dimension in one chunk, which is unbounded. For a large model a single chunk of
# (1000 time steps x every location x every location x every component) passes 2 GB,
# which the Blosc codec refuses ("Codec does not support buffers of > 2147483647
# bytes"). "auto" keeps the deliberate 1000-step time chunk and lets dask size the
# rest so a chunk stays near 128 MiB whatever the model size.
ZARR_CHUNK_SCHEME = {
    "time": 1000,
    "space": "auto",
    "space_2": "auto",
    utilsIO.COMPONENT_DIMENSION: "auto",
}

# fill value written for float variables when replaceFillValue is set
ZARR_FLOAT_FILL_VALUE = -9999.0

# dimension the components of one model class are concatenated along
ZARR_COMPONENT_DIMENSION = utilsIO.COMPONENT_DIMENSION

# name the ShadowPrices DataArray is stored under inside its own Zarr group
ZARR_SHADOW_PRICE_VARIABLE = "shadowPrices"

# version of the Zarr layout, recorded in structure.json
ZARR_FORMAT_VERSION = 1


def _zarrCompressorEncoding(compressionAlgorithm, compressionLevel):
    """Build the per variable Zarr encoding that selects the Blosc compressor.

    zarr 2 takes a numcodecs codec under the key "compressor". zarr 3 renamed the
    key to "compressors" and wants a codec from zarr.codecs, and it rejects a raw
    numcodecs object with "Expected a BytesBytesCodec". FINE supports both, so the
    key and the codec are chosen from the installed version rather than pinned.

    :param compressionAlgorithm: Blosc compressor name, e.g. "zstd"
    :type compressionAlgorithm: string

    :param compressionLevel: Blosc compression level, 1 to 9
    :type compressionLevel: int

    :return: the encoding entries to merge into each variable's encoding
    :rtype: dict
    """
    import zarr  # noqa: PLC0415 - optional dependency, imported where it is used

    if int(zarr.__version__.split(".")[0]) >= 3:
        from zarr.codecs import BloscCodec, BloscShuffle  # noqa: PLC0415

        return {
            "compressors": [
                BloscCodec(
                    cname=compressionAlgorithm,
                    clevel=compressionLevel,
                    shuffle=BloscShuffle.shuffle,
                )
            ]
        }

    from numcodecs import Blosc  # noqa: PLC0415

    return {
        "compressor": Blosc(
            cname=compressionAlgorithm,
            clevel=compressionLevel,
            shuffle=Blosc.SHUFFLE,
        )
    }


def _chunkForZarr(dataset, useScheme=True):
    """Chunk a dataset so Zarr can write it.

    :param dataset: dataset to chunk
    :type dataset: xr.Dataset

    :param useScheme: states if ZARR_CHUNK_SCHEME is applied. False falls back to
        letting dask choose every chunk, which is the retry after a chunking failure.
    :type useScheme: boolean

    :return: the chunked dataset
    :rtype: xr.Dataset
    """
    if not useScheme:
        return dataset.chunk("auto")
    return dataset.chunk(
        {dim: ZARR_CHUNK_SCHEME.get(dim, "auto") for dim in dataset.dims}
    )


def _writeZarr(dataset, path, compressorEncoding, replaceFillValue):
    """Write one dataset to one Zarr group, retrying with plain chunking on failure.

    :param dataset: dataset to write
    :type dataset: xr.Dataset

    :param path: path of the Zarr group
    :type path: string

    :param compressorEncoding: encoding entries from :func:`_zarrCompressorEncoding`
    :type compressorEncoding: dict

    :param replaceFillValue: states if float variables get an explicit fill value
    :type replaceFillValue: boolean
    """

    def _encoding(chunked):
        encoding = {}
        for name, variable in chunked.data_vars.items():
            encoding[name] = dict(compressorEncoding)
            if replaceFillValue and np.issubdtype(variable.dtype, np.floating):
                encoding[name]["_FillValue"] = ZARR_FLOAT_FILL_VALUE
        return encoding

    try:
        chunked = _chunkForZarr(dataset, useScheme=True)
        chunked.to_zarr(path, mode="w", encoding=_encoding(chunked))
    except (ValueError, TypeError) as chunkingError:
        logger.warning(
            "Writing %s with the standard chunk scheme failed (%s). Retrying with "
            "chunks chosen by dask.",
            path,
            chunkingError,
        )
        chunked = _chunkForZarr(dataset, useScheme=False)
        try:
            chunked.to_zarr(path, mode="w", encoding=_encoding(chunked))
        except (ValueError, TypeError) as retryError:
            raise ValueError(
                f"Could not write {path} to Zarr, with the standard chunk scheme "
                "or without it."
            ) from retryError


def writeDatasetsToZarr(
    datasets,
    output_zarr_path="my_esm.zarr",
    overwrite_existing=True,
    compression_level=5,
    compression_algorithm="zstd",
    replace_fill_value=False,
):
    """Write a nested dictionary of xarray datasets to a Zarr store.

    It takes the canonical datasets, that is the ones
    :func:`convertEnergySystemModelToDatasets` builds, and stacks them. The
    components of a model class are concatenated into one dataset along a
    "component" dimension, so the store holds a handful of large arrays instead of
    thousands of small ones. That is what makes it fast to read a single variable
    across all components. See :func:`~fine.IOManagement.utilsIO.stackComponents`.

    **Required arguments:**

    :param datasets: nested dictionary of xarray datasets holding the esM data
    :type datasets: dict

    **Default arguments:**

    :param output_zarr_path: path of the Zarr store directory
        |br| * the default value is "my_esm.zarr"
    :type output_zarr_path: string or pathlib.Path

    :param overwrite_existing: states if an existing store at that path is removed first
        |br| * the default value is True
    :type overwrite_existing: boolean

    :param compression_level: Blosc compression level, 1 to 9. Higher compresses more and
        writes more slowly.
        |br| * the default value is 5
    :type compression_level: int

    :param compression_algorithm: Blosc compressor name, e.g. "zstd" or "lz4"
        |br| * the default value is "zstd"
    :type compression_algorithm: string

    :param replace_fill_value: states if float variables are written with an explicit fill
        value instead of NaN
        |br| * the default value is False
    :type replace_fill_value: boolean
    """
    output_zarr_path = Path(output_zarr_path)
    if overwrite_existing and output_zarr_path.exists():
        shutil.rmtree(output_zarr_path)
    output_zarr_path.mkdir(parents=True, exist_ok=True)

    startTime = time.time()
    compressorEncoding = _zarrCompressorEncoding(
        compression_algorithm, compression_level
    )
    lazy_datasets = _make_datasets_lazy(datasets, chunks="auto")

    for model_class, components in lazy_datasets["Input"].items():
        stacked = utilsIO.stackComponents(components, prefixed=True)
        if stacked is not None:
            _writeZarr(
                stacked,
                f"{output_zarr_path}/Input/{model_class}",
                compressorEncoding,
                replace_fill_value,
            )

    for ip, models in lazy_datasets.get("Results", {}).items():
        for model_class, components in models.items():
            stacked = utilsIO.stackComponents(components, prefixed=False)
            if stacked is not None:
                _writeZarr(
                    stacked,
                    f"{output_zarr_path}/Results/{ip}/{model_class}",
                    compressorEncoding,
                    replace_fill_value,
                )

    parameters = serialiseDatasetsForWriting(
        {"Parameters": lazy_datasets["Parameters"]}
    )
    parameters["Parameters"].to_zarr(f"{output_zarr_path}/Parameters", mode="w")
    if "PerformanceSummary" in lazy_datasets:
        lazy_datasets["PerformanceSummary"].to_zarr(
            f"{output_zarr_path}/PerformanceSummary", mode="w"
        )
    if lazy_datasets.get("ShadowPrices") is not None:
        # a DataArray, not a dataset. Zarr writes a group, so give it a name.
        shadowPrices = lazy_datasets["ShadowPrices"]
        utilsIO._normaliseDtypes(
            shadowPrices.rename(ZARR_SHADOW_PRICE_VARIABLE).to_dataset()
        ).to_zarr(f"{output_zarr_path}/ShadowPrices", mode="w")

    # A model class is a directory in the store, and a directory listing has no
    # order. Record the order the classes were written in, so a model read back
    # holds its components in the order it had before, as the netCDF format does.
    structure = {
        "fine_zarr_format": ZARR_FORMAT_VERSION,
        "Input": list(lazy_datasets["Input"]),
        "Results": {
            str(ip): list(models)
            for ip, models in lazy_datasets.get("Results", {}).items()
        },
    }
    with (output_zarr_path / STRUCTURE_FILE_NAME).open("w") as structureFile:
        json.dump(structure, structureFile, indent=2)

    logger.debug("Wrote the Zarr store in %.4f sec", time.time() - startTime)


def readZarrToDatasets(zarr_path, lazy_load=True, chunks=None):
    """Read a Zarr store written by :func:`writeDatasetsToZarr` back into datasets.

    **Required arguments:**

    :param zarr_path: path of the Zarr store directory
    :type zarr_path: string or pathlib.Path

    **Default arguments:**

    :param lazy_load: states if the data is read on demand, as dask arrays, instead of at once
        |br| * the default value is True
    :type lazy_load: boolean

    :param chunks: dask chunk sizes used when reading lazily
        |br| * the default value is None
    :type chunks: None or dict

    The store is returned as it was written, that is stacked and, unless
    ``lazy_load`` is off, lazy. That is the point of the format. Use
    :func:`readZarrToEnergySystemModel` to get a model back, or
    :func:`~fine.IOManagement.utilsIO.unstackComponents` to get one dataset per
    component.

    :return: the nested dictionary of datasets, with one dataset per model class
    :rtype: dict
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found at: {zarr_path}")

    loader = xr.open_dataset if lazy_load else xr.load_dataset
    xr_dss = {}

    structure = {}
    structure_path = zarr_path / STRUCTURE_FILE_NAME
    if structure_path.exists():
        with structure_path.open() as structureFile:
            structure = json.load(structureFile)

    def _orderedGroups(path, order):
        """List the subdirectories of a store group in the order they were written."""
        present = {group.name: group for group in path.iterdir() if group.is_dir()}
        names = [name for name in order if name in present]
        names += sorted(set(present) - set(names))
        return [(name, present[name]) for name in names]

    input_path = zarr_path / "Input"
    if input_path.exists():
        xr_dss["Input"] = {
            name: loader(group, engine="zarr", chunks=chunks)
            for name, group in _orderedGroups(input_path, structure.get("Input", []))
        }

    results_path = zarr_path / "Results"
    if results_path.exists():
        results_order = structure.get("Results", {})
        xr_dss["Results"] = {
            ip_name: {
                name: loader(group, engine="zarr", chunks=chunks)
                for name, group in _orderedGroups(
                    ip_path, results_order.get(ip_name, [])
                )
            }
            for ip_name, ip_path in _orderedGroups(results_path, list(results_order))
        }

    for group in ("Parameters", "PerformanceSummary"):
        group_path = zarr_path / group
        if group_path.exists():
            xr_dss[group] = loader(group_path, engine="zarr")

    shadow_path = zarr_path / "ShadowPrices"
    if shadow_path.exists():
        # written as a one variable dataset, handed back as the DataArray it was
        xr_dss["ShadowPrices"] = loader(shadow_path, engine="zarr")[
            ZARR_SHADOW_PRICE_VARIABLE
        ]

    return xr_dss


def readZarrToEnergySystemModel(zarr_path):
    """Read a Zarr store written by :func:`writeDatasetsToZarr` back into an esM.

    The store holds one dataset per model class, with the components stacked along
    "component" and the shape of each parameter in the two masks. The unstack undoes
    both, which leaves exactly the layout
    :func:`convertDatasetsToEnergySystemModel` reads, so there is one reader and
    not two.

    :param zarr_path: path of the Zarr store directory
    :type zarr_path: string or pathlib.Path

    :return: esM - EnergySystemModel instance
    :rtype: EnergySystemModel instance
    """
    datasets = readZarrToDatasets(zarr_path, lazy_load=False)
    datasets["Input"] = {
        model_class: utilsIO.unstackComponents(stacked)
        for model_class, stacked in datasets["Input"].items()
    }
    if "Results" in datasets:
        datasets["Results"] = {
            ip: {
                model_class: utilsIO.unstackComponents(stacked)
                for model_class, stacked in models.items()
            }
            for ip, models in datasets["Results"].items()
        }
    return convertDatasetsToEnergySystemModel(datasets)


def writeEnergySystemModelToZarr(
    esM,
    output_zarr_path="my_esm.zarr",
    optSumOutputLevel=0,
    includeShadowPrices=False,
    shadowPriceConstraintStr="commodityBalanceConstraint",
    **kwargs,
):
    """Write an esM (input and, if it was optimized, output) to a Zarr store.

    :param esM: EnergySystemModel instance in which the model is held
    :type esM: EnergySystemModel instance

    :param output_zarr_path: path of the Zarr store directory
        |br| * the default value is "my_esm.zarr"
    :type output_zarr_path: string or pathlib.Path

    :param optSumOutputLevel: output level of the optimization summary
        |br| * the default value is 0
    :type optSumOutputLevel: int (0,1,2) or dict

    :param includeShadowPrices: states if the shadow prices are written as well
        |br| * the default value is False
    :type includeShadowPrices: boolean

    :param shadowPriceConstraintStr: name of the constraint the shadow prices are
        taken from
        |br| * the default value is "commodityBalanceConstraint"
    :type shadowPriceConstraintStr: string

    :param kwargs: passed on to :func:`writeDatasetsToZarr`, e.g. the compression
    """
    datasets = convertEnergySystemModelToDatasets(
        esM,
        optSumOutputLevel=optSumOutputLevel,
        includeShadowPrices=includeShadowPrices,
        shadowPriceConstraintStr=shadowPriceConstraintStr,
    )
    writeDatasetsToZarr(datasets, output_zarr_path=output_zarr_path, **kwargs)


def writeEnergySystemModelToNetCDFfolder(
    esM,
    base_path="my_esm",
    optSumOutputLevel=0,
    includeShadowPrices=False,
    shadowPriceConstraintStr="commodityBalanceConstraint",
    **kwargs,
):
    """Write an esM (input and, if it was optimized, output) to a netCDF folder.

    :param esM: EnergySystemModel instance in which the model is held
    :type esM: EnergySystemModel instance

    :param base_path: directory the tree is written into
        |br| * the default value is "my_esm"
    :type base_path: string or pathlib.Path

    :param optSumOutputLevel: output level of the optimization summary
        |br| * the default value is 0
    :type optSumOutputLevel: int (0,1,2) or dict

    :param includeShadowPrices: states if the shadow prices are written as well
        |br| * the default value is False
    :type includeShadowPrices: boolean

    :param shadowPriceConstraintStr: name of the constraint the shadow prices are
        taken from
        |br| * the default value is "commodityBalanceConstraint"
    :type shadowPriceConstraintStr: string

    :param kwargs: passed on to :func:`writeDatasetsToNetCDFfolder`, e.g. parallel

    :return: the structure that was written, see :func:`writeDatasetsToNetCDFfolder`
    :rtype: dict
    """
    datasets = convertEnergySystemModelToDatasets(
        esM,
        optSumOutputLevel=optSumOutputLevel,
        includeShadowPrices=includeShadowPrices,
        shadowPriceConstraintStr=shadowPriceConstraintStr,
    )
    return writeDatasetsToNetCDFfolder(datasets, base_path=base_path, **kwargs)


def readNetCDFfolderToEnergySystemModel(base_path, **kwargs):
    """Read a netCDF folder written by :func:`writeDatasetsToNetCDFfolder` into an esM.

    :param base_path: directory holding the tree and its structure.json
    :type base_path: string or pathlib.Path

    :param kwargs: passed on to :func:`readNetCDFfolderToDatasets`

    :return: esM - EnergySystemModel instance
    :rtype: EnergySystemModel instance
    """
    datasets = readNetCDFfolderToDatasets(base_path, **kwargs)
    return convertDatasetsToEnergySystemModel(datasets)
