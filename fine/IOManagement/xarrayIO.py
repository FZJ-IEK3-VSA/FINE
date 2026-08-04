import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

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


def writeEnergySystemModelToDatasets(
    esM,
    includeShadowPrices=False,
    shadowPriceConstraintStr="commodityBalanceConstraint",
):
    """Convert esM instance (input and output) into a xarray dataset.

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
    if esM.objectiveValue is not None:  # model was optimized
        xr_dss_output = convertOptimizationOutputToDatasets(esM)
        xr_dss_input = convertOptimizationInputToDatasets(esM)

        xr_dss_results = {
            "Results": xr_dss_output["Results"],
            "Input": xr_dss_input["Input"],
            "Parameters": xr_dss_input["Parameters"],
        }
        if "performanceSummary" in vars(esM):
            xr_dss_performance = convertPerformanceSummaryToDatasets(esM)
            xr_dss_results["PerformanceSummary"] = xr_dss_performance[
                "PerformanceSummary"
            ]

        if includeShadowPrices:
            xr_dss_shadowPrices = utilsIO.getShadowPriceXarray(
                esM, constraint_str=shadowPriceConstraintStr
            )
            xr_dss_results["ShadowPrices"] = xr_dss_shadowPrices
    else:
        xr_dss_input = convertOptimizationInputToDatasets(esM)
        xr_dss_results = {
            "Input": xr_dss_input["Input"],
            "Parameters": xr_dss_input["Parameters"],
        }

    return xr_dss_results


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


def readNetCDFtoEnergySystemModel(filePath, groupPrefix=None):
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
