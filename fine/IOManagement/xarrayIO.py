import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset

from fine import utils
from fine.IOManagement import dictIO, utilsIO


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

    # STEP 2. Get the iteration dicts
    ip = esM.investmentPeriods
    (
        df_iteration_dict,
        series_iteration_dict,
        constants_iteration_dict,
    ) = utilsIO.generateIterationDicts(component_dict, ip)

    # STEP 3. Initiate xarray dataset
    xr_dss = dict.fromkeys(component_dict.keys())
    for classname in component_dict:
        xr_dss[classname] = {
            component: xr.Dataset() for component in component_dict[classname]
        }

    # STEP 3.1 get _mapC for all transmission components
    _mapC_dict = {}
    for transmission_class in ["LinearOptimalPowerFlow", "Transmission"]:
        for tech in component_dict[transmission_class].keys():
            _mapC_dict[tech] = esM.getComponent(tech)._mapC

    # STEP 4. Add all df variables to xr_ds
    xr_dss = utilsIO.addDFVariablesToXarray(
        xr_dss, component_dict, df_iteration_dict, _mapC_dict, list(esM.locations)
    )

    # STEP 5. Add all series variables to xr_ds
    locations = sorted(esm_dict["locations"])
    xr_dss = utilsIO.addSeriesVariablesToXarray(
        xr_dss, component_dict, series_iteration_dict, locations
    )

    # STEP 6. Add all constant value variables to xr_ds
    xr_dss = utilsIO.addConstantsToXarray(
        xr_dss, component_dict, constants_iteration_dict, useProcessedValues
    )

    # STEP 7. Add the data present in esm_dict as xarray attributes
    # (These attributes contain esM init info).
    attributes_xr = xr.Dataset()
    attributes_xr.attrs = esm_dict

    return {"Input": xr_dss, "Parameters": attributes_xr}


def convertOptimizationInputToDatasetsZarr(esM, useProcessedValues=False):
    """
    Takes esM instance input and converts it into xarray datasets.

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
    
    # STEP 3.1 get _mapC for all transmission components
    _mapC_dict = {}
    for tech in component_dict["Transmission"].keys():
        _mapC_dict[tech] = esM.getComponent(tech)._mapC
    
    component_dict_mod = utilsIO.processComponentDict(component_dict, list(esM.locations), _mapC_dict)

    # STEP 1.1. Create comprehensive parameter mask for all parameters and their dimensions
    dimension_mask = utilsIO.createParameterDimensionDict(component_dict_mod)

    # STEP 1.2. Create was_none_mask for perfect reconstruction
    was_none_mask = utilsIO.createWasNoneMask(component_dict_mod)
    # STEP 1.3. Replace None values with appropriate defaults for xarray
    component_dict = utilsIO.replaceNoneValuesForXarray(component_dict_mod)

    # STEP 2. Convert esm_dict to xarray datasets
    xr_dss = utilsIO.convertToXarray(component_dict_mod)
    
    # STEP 7. Add comprehensive parameter information (dimensions). And None tracker. Instead of 0d,1d,2d,ts prefix
    xr_dss = utilsIO.addParameterDimensionsToXarray(xr_dss, dimension_mask, was_none_mask)

    attributes_xr = xr.Dataset()
    attributes_xr.attrs = esm_dict

    xr_dss = {"Input": xr_dss, "Parameters": attributes_xr}

    return xr_dss

def convertOptimizationInputToDatasetsZarr(esM, useProcessedValues=False):
    """
    Takes esM instance input and converts it into xarray datasets.

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
    
    # STEP 3.1 get _mapC for all transmission components
    _mapC_dict = {}
    for tech in component_dict["Transmission"].keys():
        _mapC_dict[tech] = esM.getComponent(tech)._mapC
    
    component_dict_mod = utilsIO.processComponentDict(component_dict, list(esM.locations), _mapC_dict)

    # STEP 1.1. Create comprehensive parameter mask for all parameters and their dimensions
    dimension_mask = utilsIO.createParameterDimensionDict(component_dict_mod)

    # STEP 1.2. Create was_none_mask for perfect reconstruction
    was_none_mask = utilsIO.createWasNoneMask(component_dict_mod)
    # STEP 1.3. Replace None values with appropriate defaults for xarray
    component_dict = utilsIO.replaceNoneValuesForXarray(component_dict_mod)

    # STEP 2. Convert esm_dict to xarray datasets
    xr_dss = utilsIO.convertToXarray(component_dict_mod)
    
    # STEP 7. Add comprehensive parameter information (dimensions). And None tracker. Instead of 0d,1d,2d,ts prefix
    xr_dss = utilsIO.addParameterDimensionsToXarray(xr_dss, dimension_mask, was_none_mask)

    attributes_xr = xr.Dataset()
    attributes_xr.attrs = esm_dict

    xr_dss = {"Input": xr_dss, "Parameters": attributes_xr}

    return xr_dss

def convertPerformanceSummaryToDatasets(esM):
    import pandas as pd
    df = esM.performanceSummary.squeeze()
    df = df.droplevel("Category")
    df = df.apply(lambda x: pd.to_numeric(x, errors="ignore"))
    # convert datetime to string
    for idx, value in df.items():
        if isinstance(value, pd.Timestamp):
            print(value)
            df.loc[idx] = value.strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(value, dict):
            df.loc[idx] = str(value)
    summary_dict = df.to_dict()
    summary_xr = xr.Dataset()
    summary_xr.attrs = summary_dict

    return {"PerformanceSummary": summary_xr}

def convertOptimizationInputToDatasetsZarr(esM, useProcessedValues=False):
    """
    Takes esM instance input and converts it into xarray datasets.

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
    
    # STEP 3.1 get _mapC for all transmission components
    _mapC_dict = {}
    for tech in component_dict["Transmission"].keys():
        _mapC_dict[tech] = esM.getComponent(tech)._mapC
    
    component_dict_mod = utilsIO.processComponentDict(component_dict, list(esM.locations), _mapC_dict)

    # STEP 1.1. Create comprehensive parameter mask for all parameters and their dimensions
    dimension_mask = utilsIO.createParameterDimensionDict(component_dict_mod)

    # STEP 1.2. Create was_none_mask for perfect reconstruction
    was_none_mask = utilsIO.createWasNoneMask(component_dict_mod)
    # STEP 1.3. Replace None values with appropriate defaults for xarray
    component_dict = utilsIO.replaceNoneValuesForXarray(component_dict_mod)

    # STEP 2. Convert esm_dict to xarray datasets
    xr_dss = utilsIO.convertToXarray(component_dict_mod)
    
    # STEP 7. Add comprehensive parameter information (dimensions). And None tracker. Instead of 0d,1d,2d,ts prefix
    xr_dss = utilsIO.addParameterDimensionsToXarray(xr_dss, dimension_mask, was_none_mask)

    attributes_xr = xr.Dataset()
    attributes_xr.attrs = esm_dict

    return {"Input": xr_dss, "Parameters": attributes_xr}

    return xr_dss


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
            utils.output("\tProcessing " + name + " ...", esM.verbose, 0)
            oL = optSumOutputLevel
            oL_ = oL[name] if isinstance(oL, dict) else oL
            optSum = esM.getOptimizationSummary(name, ip=ip, outputLevel=oL_)
            if esM.componentModelingDict[name].dimension == "1dim":
                for component in optSum.index.get_level_values(0).unique():
                    for variable in (
                        optSum.loc[component].index.get_level_values(0).unique()
                    ):
                        df_o = optSum.loc[(component, variable)]
                        # differentiate if two entries per variable, i.e. operation: annual and normal, because nc4 can not save two entries with the same name
                        if df_o.shape[0] == 2:
                            # first half of df_o, as it is annual and normal operation
                            df = df_o.iloc[0].copy()
                            df.name = variable
                            df.index.rename("space", inplace=True)
                            df = pd.to_numeric(df, errors="coerce")
                            xr_da = df.to_xarray()
                            unit = df_o.iloc[0].name
                            # unit = variables_unit[variable]
                            xr_da.attrs[variable] = unit
                            # merge to overall xr_dss
                            xr_dss[ip][name][component] = xr.merge(
                                [xr_dss[ip][name][component], xr_da],
                                combine_attrs="drop_conflicts",
                            )
                            # second half of df_o, as it is annual and normal operation
                            df = df_o.iloc[1].copy()
                            df.name = f"{variable}_{1}"
                            df.index.rename("space", inplace=True)
                            df = pd.to_numeric(df, errors="coerce")
                            xr_da = df.to_xarray()
                            # add variable [e.g. 'TAC'] and units to attributes of xarray
                            unit = df_o.iloc[1].name
                            xr_da.attrs[df.name] = unit
                        else:
                            df = df_o.iloc[-1]
                            df.name = variable
                            df.index.rename("space", inplace=True)
                            df = pd.to_numeric(df, errors="coerce")
                            xr_da = df.to_xarray()
                            # add variable [e.g. 'TAC'] and units to attributes of xarray
                            unit = df_o.iloc[-1].name
                            xr_da.attrs[variable] = unit

                        # merge to overall xr_ds
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            combine_attrs="drop_conflicts",
                        )
            elif esM.componentModelingDict[name].dimension == "2dim":
                for component in optSum.index.get_level_values(0).unique():
                    for variable in (
                        optSum.loc[component].index.get_level_values(0).unique()
                    ):
                        df_o = optSum.loc[(component, variable)]
                        # differentiate if two entries per variable, i.e. operation: annual and normal
                        if "operation" in variable or variable == "operation":
                            # first half of df_o, as it is annual and normal operation
                            len_df_o = int(len(df_o))
                            df = df_o.iloc[0 : int(len_df_o / 2), :].copy()
                            if len(df.index.get_level_values(0).unique()) > 1:
                                idx = df.index.get_level_values(0).unique()[-1]
                                df = df.xs(idx, level=0)
                            else:
                                df.index = df.index.droplevel(0)
                            df = df.stack()
                            df.name = variable
                            df.index.rename(["LocationIn", "LocationOut"], inplace=True)
                            df = pd.to_numeric(df, errors="coerce")
                            xr_da = df.to_xarray()
                            # add variable [e.g. 'TAC'] and units to attributes of xarray
                            unit = df_o.iloc[
                                0 : int(len_df_o / 2), :
                            ].index.get_level_values(0)[0]
                            xr_da.attrs[variable] = unit
                            # merge to overall xr_ds
                            xr_dss[ip][name][component] = xr.merge(
                                [xr_dss[ip][name][component], xr_da],
                                combine_attrs="drop_conflicts",
                            )

                            # second half of df_o, as it is annual and normal operation
                            df = df_o.iloc[int(len_df_o / 2) : len_df_o, :].copy()
                            if len(df.index.get_level_values(0).unique()) > 1:
                                idx = df.index.get_level_values(0).unique()[-1]
                                df = df.xs(idx, level=0)
                            else:
                                df.index = df.index.droplevel(0)
                            df = df.stack()
                            df.name = f"{variable}_{1}"
                            df.index.rename(["LocationIn", "LocationOut"], inplace=True)
                            df = pd.to_numeric(df, errors="coerce")
                            xr_da = df.to_xarray()
                            # add variable [e.g. 'TAC'] and units to attributes of xarray
                            unit = df_o.iloc[
                                int(len_df_o / 2) : len_df_o, :
                            ].index.get_level_values(0)[0]
                            xr_da.attrs[df.name] = unit

                        else:
                            df = df_o.copy()
                            if len(df.index.get_level_values(0).unique()) > 1:
                                idx = df.index.get_level_values(0).unique()[-1]
                                df = df.xs(idx, level=0)
                            else:
                                df.index = df.index.droplevel(0)
                            df = df.stack()
                            df.name = variable
                            df.index.rename(["LocationIn", "LocationOut"], inplace=True)
                            df = pd.to_numeric(df, errors="coerce")
                            xr_da = df.to_xarray()

                            # add variable [e.g. 'TAC'] and units to attributes of xarray
                            unit = df_o.index.get_level_values(0)[0]
                            xr_da.attrs[variable] = unit

                        # merge to overall xr_ds
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            combine_attrs="drop_conflicts",
                        )

            # Write output from esM.esM.componentModelingDict[name].getOptimalValues() to datasets
            data = esM.componentModelingDict[name].getOptimalValues(ip=ip)
            dataTD1dim, indexTD1dim, dataTD2dim, indexTD2dim = [], [], [], []
            dataTI, indexTI = [], []
            for key, d in data.items():
                if d["values"] is None:
                    continue
                if d["timeDependent"]:
                    if d["dimension"] == "1dim":
                        dataTD1dim.append(d["values"]), indexTD1dim.append(key)
                    elif d["dimension"] == "2dim":
                        dataTD2dim.append(d["values"]), indexTD2dim.append(key)
                else:
                    dataTI.append(d["values"]), indexTI.append(key)
            # One dimensional time dependent data
            if dataTD1dim:
                names = ["Variable", "Component", "Location"]
                dfTD1dim = pd.concat(dataTD1dim, keys=indexTD1dim, names=names)
                # dfTD1dim = dfTD1dim.loc[
                #    ((dfTD1dim != 0) & (~dfTD1dim.isnull())).any(axis=1)
                # ]
                for variable in dfTD1dim.index.get_level_values(0).unique():
                    # for component in dfTD1dim.index.get_level_values(1).unique():
                    for component in (
                        dfTD1dim.loc[variable].index.get_level_values(0).unique()
                    ):
                        df = dfTD1dim.loc[(variable, component)].T.stack()
                        # df.name = (name, component, variable)
                        df.name = variable
                        df.index.rename(["time", "space"], inplace=True)
                        xr_da = df.to_xarray()
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da]
                        )
            # Two dimensional time dependent data
            if dataTD2dim:
                names = ["Variable", "Component", "LocationIn", "LocationOut"]
                dfTD2dim = pd.concat(dataTD2dim, keys=indexTD2dim, names=names)
                # dfTD2dim = dfTD2dim.loc[
                #    ((dfTD2dim != 0) & (~dfTD2dim.isnull())).any(axis=1)
                # ]
                for variable in dfTD2dim.index.get_level_values(0).unique():
                    # for component in dfTD2dim.index.get_level_values(1).unique():
                    for component in (
                        dfTD2dim.loc[variable].index.get_level_values(0).unique()
                    ):
                        df = dfTD2dim.loc[(variable, component)].stack()
                        # df.name = (name, component, variable)
                        df.name = variable
                        df.index.rename(["space", "space_2", "time"], inplace=True)
                        df.index = df.index.reorder_levels([2, 0, 1])
                        xr_da = df.to_xarray()
                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da]
                        )
            # Time independent data
            if dataTI:
                # One dimensional
                if esM.componentModelingDict[name].dimension == "1dim":
                    names = ["Variable type", "Component"]
                    dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                    # dfTI = dfTI.loc[((dfTI != 0) & (~dfTI.isnull())).any(axis=1)]
                    for variable in dfTI.index.get_level_values(0).unique():
                        # for component in dfTI.index.get_level_values(1).unique():
                        for component in (
                            dfTI.loc[variable].index.get_level_values(0).unique()
                        ):
                            df = dfTI.loc[(variable, component)].T
                            # df.name = (name, component, variable)
                            df.name = variable
                            df.index.rename("space", inplace=True)
                            xr_da = df.to_xarray()
                            xr_dss[ip][name][component] = xr.merge(
                                [xr_dss[ip][name][component], xr_da]
                            )
                # Two dimensional
                elif esM.componentModelingDict[name].dimension == "2dim":
                    names = ["Variable type", "Component", "Location"]
                    dfTI = pd.concat(dataTI, keys=indexTI, names=names)
                    # dfTI = dfTI.loc[((dfTI != 0) & (~dfTI.isnull())).any(axis=1)]
                    for variable in dfTI.index.get_level_values(0).unique():
                        # for component in dfTI.index.get_level_values(1).unique():
                        for component in (
                            dfTI.loc[variable].index.get_level_values(0).unique()
                        ):
                            df = dfTI.loc[(variable, component)].T.stack()
                            # df.name = (name, component, variable)
                            df.name = variable
                            df.index.rename(["space", "space_2"], inplace=True)
                            xr_da = df.to_xarray()
                            xr_dss[ip][name][component] = xr.merge(
                                [xr_dss[ip][name][component], xr_da]
                            )

        for name in esM.componentModelingDict.keys():
            for component in esM.componentModelingDict[name].componentsDict.keys():
                if list(xr_dss[ip][name][component].data_vars) == []:
                    # Delete components that have not been built.
                    del xr_dss[ip][name][component]
                else:
                    # Cast space coordinats to str. If this is not done then dtype will be object.
                    xr_dss[ip][name][component].coords["space"] = (
                        xr_dss[ip][name][component].coords["space"].astype(str)
                    )
                    if esM.componentModelingDict[name].dimension == "2dim":
                        xr_dss[ip][name][component].coords["space_2"] = (
                            xr_dss[ip][name][component].coords["space_2"].astype(str)
                        )

    return {"Results": xr_dss}

def processDataset(datasets):
    """
    Process a dataset to make it saveable.
    """

    for group in datasets.keys():
        if group == "Parameters" or group == "PerformanceSummary":
            xarray_dataset = datasets[group]
            _xarray_dataset = (
                xarray_dataset.copy()
            )  # Copying to avoid errors due to change of size during iteration

            for attr_name, attr_value in _xarray_dataset.attrs.items():
                # if the attribute is set, convert into sorted list
                if isinstance(attr_value, set):
                    xarray_dataset.attrs[attr_name] = sorted(
                        xarray_dataset.attrs[attr_name]
                    )

                # if the attribute is dict, convert into a "flattened" list
                elif isinstance(attr_value, dict):
                    xarray_dataset.attrs[attr_name] = list(
                        f"{k} : {v}"
                        for (k, v) in xarray_dataset.attrs[attr_name].items()
                    )

                # if the attribute is pandas series, add a new attribute corresponding
                # to each row.
                elif isinstance(attr_value, pd.Series):
                    for idx, value in attr_value.items():
                        xarray_dataset.attrs.update({f"{attr_name}.{idx}": value})

                    # Delete the original attribute
                    del xarray_dataset.attrs[attr_name]

                # if the attribute is pandas df, add a new attribute corresponding
                # to each row by converting the column into a numpy array.
                elif isinstance(attr_value, pd.DataFrame):
                    _df = attr_value
                    _df = _df.reindex(sorted(_df.columns), axis=1)
                    for idx, row in _df.iterrows():
                        xarray_dataset.attrs.update(
                            {f"{attr_name}.{idx}": row.to_list()}
                        )

                    # Delete the original attribute
                    del xarray_dataset.attrs[attr_name]

                # if the attribute is bool, add a corresponding string
                elif isinstance(attr_value, bool):
                    xarray_dataset.attrs[attr_name] = (
                        "True" if attr_value is True else "False"
                    )

                # if the attribute is None, add a corresponding string
                elif attr_value is None:
                    xarray_dataset.attrs[attr_name] = "None"
    
    return datasets
    
def writeDatasetsToNetCDFfolder(
    data_dict,
    base_path="my_esm",
    compression=True,
    parallel=False,
    chunks=None,
    mode="w",
):
    """
    Save a nested dictionary of xarray datasets to disk with optimized performance.
    
    Parameters:
    -----------
    data_dict : dict
        Nested dictionary containing xarray datasets
    base_path : str
        Base directory to save the files
    compression : bool, optional
        Whether to enable compression (default: True)
    parallel : bool, optional
        Whether to use parallel processing (default: False)
    chunks : dict, optional
        Chunk sizes for dask arrays (e.g., {'time': 100, 'lat': 50})
    mode : str, optional
        Write mode ('w' or 'a') (default: 'w')
    
    Returns:
    --------
    dict
        Dictionary with the same structure but containing paths to saved files
    """
    
    import xarray as xr
    import os
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    import dask
    import json

    def save_dataset(args):
        """Helper function for parallel saving"""
        filepath, dataset, compression_settings = args
        dataset.to_netcdf(filepath, encoding={
            var: compression_settings for var in dataset.data_vars
        }, mode=mode)
        return filepath

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    compression_settings = {
        'zlib': True,
        'complevel': 5,
        'shuffle': True
    } if compression else {}
    
    save_tasks = []
    paths_dict = {}
    
    def collect_save_tasks(item, current_path, current_dict):
        if isinstance(item, dict):
            current_dict_level = {}
            for key, value in item.items():
                new_path = current_path / str(key)
                new_path.mkdir(exist_ok=True)
                current_dict_level[key] = collect_save_tasks(value, new_path, {})
            return current_dict_level
        
        elif isinstance(item, xr.Dataset):
            filename = current_path / "data.nc"
            
            if chunks is not None:
                item = item.chunk(chunks)
            
            # Store absolute path in save_tasks but relative path in paths_dict
            save_tasks.append((str(filename), item, compression_settings))
            return str(filename.relative_to(base_path))
        
        else:
            raise ValueError(f"Unsupported type: {type(item)}")
    
    data_dict = processDataset(data_dict)
    paths_dict = collect_save_tasks(data_dict, base_path, paths_dict)
    
    with open(base_path / "structure.json", 'w') as f:
        json.dump(paths_dict, f, indent=2)
    
    if parallel and save_tasks:
        with ProcessPoolExecutor() as executor:
            list(executor.map(save_dataset, save_tasks))
    else:
        for task in save_tasks:
            save_dataset(task)
    
    return paths_dict

def _load_single_dataset(path, chunks=None, lazy_load=False):
    """Helper function to load a single dataset."""
    import xarray as xr
    if lazy_load:
        return xr.open_dataset(path, chunks=chunks)
    else:
        return xr.load_dataset(path, chunks=chunks)

def _collect_paths(item, path=()):
    """Helper function to collect all paths from nested dictionary."""
    if isinstance(item, dict):
        return {k: _collect_paths(v, path + (k,)) for k, v in item.items()}
    elif isinstance(item, str):
        return item
    else:
        raise ValueError(f"Unsupported type: {type(item)}")

def _rebuild_structure(item, loaded_datasets):
    """Helper function to rebuild dictionary structure with loaded datasets."""
    if isinstance(item, dict):
        return {k: _rebuild_structure(v, loaded_datasets) for k, v in item.items()}
    elif isinstance(item, str):
        return loaded_datasets[item]
    else:
        raise ValueError(f"Unsupported type: {type(item)}")

def readNetCDFfolderToDatasets(base_path, parallel=True, chunks=None, lazy_load=False):
    """
    Load nested xarray datasets with optimized performance.
    
    Parameters:
    -----------
    base_path : str
        Base directory containing the dataset files
    parallel : bool, optional
        Whether to use parallel processing (default: True)
    chunks : dict, optional
        Chunk sizes for lazy loading (e.g., {'time': 100, 'lat': 50})
    
    Returns:
    --------
    dict
        Dictionary with the same structure containing loaded datasets
    """
    base_path = Path(base_path)
    import json
    import os
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    # Load structure metadata
    with open(base_path / "structure.json", 'r') as f:
        paths_dict = json.load(f)
    
    # Collect all paths that need to be loaded
    paths_to_load = []
    def collect_all_paths(d):
        if isinstance(d, dict):
            for v in d.values():
                collect_all_paths(v)
        elif isinstance(d, str):
            # Convert relative path from structure.json to absolute path
            paths_to_load.append(str(base_path / d))
    
    collect_all_paths(paths_dict)
    
    # Load datasets (in parallel if requested)
    loaded_datasets = {}
    load_fn = partial(_load_single_dataset, chunks=chunks, lazy_load=lazy_load)
    
    if parallel and paths_to_load:
        with ProcessPoolExecutor() as executor:
            results = executor.map(load_fn, paths_to_load)
            # Use relative paths as keys in loaded_datasets
            loaded_datasets = dict(zip(
                (str(Path(p).relative_to(base_path)) for p in paths_to_load),
                results
            ))
    else:
        for path in paths_to_load:
            loaded_datasets[str(Path(path).relative_to(base_path))] = load_fn(path)
    
    # Rebuild the original structure
    result = _rebuild_structure(paths_dict, loaded_datasets)
    
    return result

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
            _xarray_dataset = (
                xarray_dataset.copy()
            )  # Copying to avoid errors due to change of size during iteration

            for attr_name, attr_value in _xarray_dataset.attrs.items():
                # if the attribute is set, convert into sorted list
                if isinstance(attr_value, set):
                    xarray_dataset.attrs[attr_name] = sorted(
                        xarray_dataset.attrs[attr_name]
                    )

                # if the attribute is dict, convert into a "flattened" list
                elif isinstance(attr_value, dict):
                    xarray_dataset.attrs[attr_name] = list(
                        f"{k} : {v}"
                        for (k, v) in xarray_dataset.attrs[attr_name].items()
                    )

                # if the attribute is pandas series, add a new attribute corresponding
                # to each row.
                elif isinstance(attr_value, pd.Series):
                    for idx, value in attr_value.items():
                        xarray_dataset.attrs.update({f"{attr_name}.{idx}": value})

                    # Delete the original attribute
                    del xarray_dataset.attrs[attr_name]

                # if the attribute is pandas df, add a new attribute corresponding
                # to each row by converting the column into a numpy array.
                elif isinstance(attr_value, pd.DataFrame):
                    _df = attr_value
                    _df = _df.reindex(sorted(_df.columns), axis=1)
                    for idx, row in _df.iterrows():
                        xarray_dataset.attrs.update(
                            {f"{attr_name}.{idx}": row.to_numpy().astype(str)}
                        )
                        if attr_name in ("balanceLimit", "componentLimit"):
                            xarray_dataset.attrs.update(
                                {f"{attr_name}_columns": _df.columns.tolist()}
                            )
                            xarray_dataset.attrs.update(
                                {f"{attr_name}_dtypes": _df.dtypes.astype(str).tolist()}
                            )

                    # Delete the original attribute
                    del xarray_dataset.attrs[attr_name]

                # if the attribute is bool, add a corresponding string
                elif isinstance(attr_value, bool):
                    xarray_dataset.attrs[attr_name] = (
                        "True" if attr_value is True else "False"
                    )

                # if the attribute is None, add a corresponding string
                elif attr_value is None:
                    xarray_dataset.attrs[attr_name] = "None"

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
                    
                    blacklist = ["aggregated"]
                    if any(blacklisted in variable for blacklisted in blacklist):
                        continue

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
            optSum = {}
            operationVariablesOptimum_dict = {}
            capacityVariablesOptimum_dict = {}
            isBuiltVariablesOptimum_dict = {}
            commissioningVariablesOptimum_dict = {}
            decommissioningVariablesOptimum_dict = {}
            chargeOperationVariablesOptimum_dict = {}
            dischargeOperationVariablesOptimum_dict = {}
            stateOfChargeOperationVariablesOptimum_dict = {}

            for ip in datasets["Results"].keys():
                # read opt Summary
                optSum_df = pd.DataFrame([])
                for component in datasets["Results"][ip][model]:
                    optSum_df_comp = pd.DataFrame([])
                    for variable in datasets["Results"][ip][model][component]:
                        if "Optimum" in variable:
                            continue
                        if "space_2" in list(
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
                                ][variable]["LocationIn"].values
                            ]
                            idx = pd.MultiIndex.from_tuples(tuple(iterables2))
                            _optSum_df.index = idx
                            _optSum_df.index.set_names(
                                names=[
                                    "Component",
                                    "Property",
                                    "Unit",
                                    "LocationIn",
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

                        if (
                            "operation" in variable and "_1" in variable
                        ):  # operation needed to be renamed in conversion
                            optSum_df_comp = optSum_df_comp.rename(
                                index={variable: variable.replace("_1", "")}
                            )  # to dataset and xarray and now is renamed to operation again

                    if isinstance(optSum_df_comp, pd.Series):
                        optSum_df_comp = optSum_df_comp.to_frame().T
                    optSum_df = pd.concat(
                        [optSum_df, optSum_df_comp],
                        axis=0,
                    )
                optSum[int(ip)] = optSum_df

                setattr(esM.componentModelingDict[model], "_optSummary", optSum)

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

                    for variable in datasets["Results"][ip][model][component]:
                        if "Optimum" not in variable:
                            continue
                        opt_variable = variable
                        xr_opt = None
                        if opt_variable in datasets["Results"][ip][model][component]:
                            xr_opt = datasets["Results"][ip][model][component][
                                opt_variable
                            ]
                        else:
                            continue

                        if opt_variable == "operationVariablesOptimum":
                            if "space_2" in list(xr_opt.coords):
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
                            if "space_2" in list(xr_opt.coords):
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
                            if "space_2" in list(xr_opt.coords):
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
                            if "space_2" in list(xr_opt.coords):
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

            setattr(
                esM.componentModelingDict[model],
                "_operationVariablesOptimum",
                operationVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_capacityVariablesOptimum",
                capacityVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_isBuiltVariablesOptimum",
                isBuiltVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_commissioningVariablesOptimum",
                commissioningVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_decommissioningVariablesOptimum",
                decommissioningVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_chargeOperationVariablesOptimum",
                chargeOperationVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_dischargeOperationVariablesOptimum",
                dischargeOperationVariablesOptimum_dict,
            )
            setattr(
                esM.componentModelingDict[model],
                "_stateOfChargeOperationVariablesOptimum",
                stateOfChargeOperationVariablesOptimum_dict,
            )

            # if only one investment period -> keep optimal values unchanged for end user
            def setFinalOptimalValues(esM, name):
                if len(esM.investmentPeriodNames) == 1:
                    data = getattr(esM.componentModelingDict[model], "_" + name)
                    setattr(
                        esM.componentModelingDict[model], name, data[int(startyear)]
                    )
                else:
                    data = getattr(esM.componentModelingDict[model], "_" + name)
                    setattr(esM.componentModelingDict[model], name, data)
                return esM

            optimalParameters = [
                "optSummary",
                "operationVariablesOptimum",
                "capacityVariablesOptimum",
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

    :return: Nested dictionary containing xr.Dataset with all result values
        for each component.
    :rtype: Dict[str, Dict[str, xr.Dataset]]
    """
    if overwriteExisting:
        if Path(outputFilePath).is_file():
            Path(outputFilePath).unlink()

    utils.output("\nWriting output to netCDF... ", esM.verbose, 0)
    _t = time.time()

    xr_dss_input = convertOptimizationInputToDatasets(esM)
    writeDatasetsToNetCDF(xr_dss_input, outputFilePath, groupPrefix=groupPrefix)
    if esM.objectiveValue is not None:  # model was optimized
        xr_dss_output = convertOptimizationOutputToDatasets(esM, optSumOutputLevel)
        if hasattr(esM, "performanceSummary"):
            xr_dss_performance = convertPerformanceSummaryToDatasets(esM)
            xr_dss_output["PerformanceSummary"] = xr_dss_performance[
                "PerformanceSummary"
            ]
            print(xr_dss_output.keys())
        writeDatasetsToNetCDF(xr_dss_output, outputFilePath, groupPrefix=groupPrefix)

    utils.output("Done. (%.4f" % (time.time() - _t) + " sec)", esM.verbose, 0)


def writeEnergySystemModelToDatasets(esM, zarrFormat=False):
    """Converts esM instance (input and output) into a xarray dataset.

    :param esM: EnergySystemModel instance in which the optimized model is held
    :type esM: EnergySystemModel instance

    :return: xr_dss_results - esM instance (input and output) data in xarray
        dataset format
    :rtype: xr.DataSet
    """
    if esM.objectiveValue is not None:  # model was optimized
        xr_dss_output = convertOptimizationOutputToDatasets(esM)
        if zarrFormat:
            xr_dss_input = convertOptimizationInputToDatasetsZarr(esM)
        else:
            xr_dss_input = convertOptimizationInputToDatasets(esM)
        if hasattr(esM, "performanceSummary"):
            xr_dss_performance = convertPerformanceSummaryToDatasets(esM)
            
            xr_dss_results = {
                "Results": xr_dss_output["Results"],
                "Input": xr_dss_input["Input"],
                "Parameters": xr_dss_input["Parameters"],
                "PerformanceSummary": xr_dss_performance["PerformanceSummary"],
            }
        else:
            xr_dss_results = {
                "Results": xr_dss_output["Results"],
                "Input": xr_dss_input["Input"],
                "Parameters": xr_dss_input["Parameters"],
            }
    else:
        if zarrFormat:
            xr_dss_input = convertOptimizationInputToDatasetsZarr(esM)
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
        xr_dss["Parameters"] = loader(filePath, group=f"Parameters")
        # read performance summary from netcdf (if exists)
        if "PerformanceSummary" in group_keys:
            xr_dss["PerformanceSummary"] = loader(
                filePath, group=f"PerformanceSummary"
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
        xr_dss["Parameters"] = loader(
            filePath, group=f"{groupPrefix}/Parameters"
        )
        # read performance summary from netcdf (if exists)
        if "PerformanceSummary" in group_keys:
            xr_dss["PerformanceSummary"] = loader(
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
    esM = convertDatasetsToEnergySystemModel(xr_dss)

    return esM

def _make_datasets_lazy(data_dict, chunks=None):
    """
    Recursively traverses a dictionary of datasets and converts their
    in-memory NumPy arrays to lazy Dask arrays by chunking them.
    Skips chunking for datasets with object dtypes to avoid NotImplementedError.
    """
    if isinstance(data_dict, dict):
        return {key: _make_datasets_lazy(value, chunks) for key, value in data_dict.items()}
    
    elif isinstance(data_dict, xr.Dataset):
        try:
            return data_dict.chunk(chunks)
        except Exception as e:
            print(f"Error chunking dataset: {e}")
            return data_dict

    else:
        return data_dict

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Optional, Any, List, Union


def writeDatasetsToZarr(
    data_dict,
    output_zarr_path="my_esm.zarr",
    overwrite_existing=True,
    compression_level=5,
    compression_algorithm='zstd',
    replace_fill_value=False,
):
    """
    Writes an entire EnergySystemModel instance to a consolidated Zarr store.
    data_dict : dict
        Nested dictionary containing xarray datasets
    output_zarr_path : str
        Path to the output Zarr store directory.
    overwrite_existing : bool
        If True, existing Zarr store at the output path will be overwritten.
        If False, an error will be raised if the path already exists.
    compression_level : int
        Compression level (1-9, higher = better compression but slower).
    """
    from numcodecs import Blosc  # Zarr's preferred compressor
    if overwrite_existing and Path(output_zarr_path).exists():
        import shutil
        shutil.rmtree(output_zarr_path)
    
    Path(output_zarr_path).mkdir(parents=True, exist_ok=True)
    
    print(f"\nPreparing data for Zarr storage...")
    
    # try:
    #     import pickle
    #     import datetime
    #     my_path = r"/fast/central/projects/2021-p-dunkel-phd/02_Research/06_Post/05_Runs/19_zarr_testing/02_Results"
    #     # save as pickle
    #     current_date_time = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    #     with open(f"{my_path}/data_dict_{current_date_time}.pickle", "wb") as f:
    #         pickle.dump(data_dict, f)
    # except Exception as e:
    #     print("Could not save data_dict as pickle, skipping...")

    
    _t = time.time()
    
    # Use higher compression and better algorithm for smaller files
    compressor = Blosc(cname=compression_algorithm, clevel=compression_level, shuffle=Blosc.SHUFFLE)
    
    lazy_data_dict = _make_datasets_lazy(data_dict, chunks=None)
    # lazy_data_dict = data_dict  # Assuming data_dict is already lazy-loaded or chunked


    def robust_concat(components_dict, tech_dim_name='technology'):
        if not components_dict:
            return None

        datasets_to_concat = list(components_dict.values())
        component_names = list(components_dict.keys())

        # If only one dataset, return it with the technology dimension added
        if len(datasets_to_concat) == 1:
            ds = datasets_to_concat[0].copy()
            # Add the technology dimension
            ds = ds.expand_dims({tech_dim_name: [component_names[0]]})
            
            # Fix string and numeric dtypes for data variables
            for var_name, var_data in ds.data_vars.items():
                if var_data.dtype == 'object':
                    # Check if it should be string type
                    sample_values = var_data.values.flatten()
                    non_null_values = [v for v in sample_values if pd.notna(v) and v is not None]
                    if non_null_values and all(isinstance(v, str) for v in non_null_values):
                        ds[var_name] = var_data.astype('U')  # Convert to string
                    elif non_null_values and all(isinstance(v, (int, float, np.integer, np.floating)) for v in non_null_values):
                        # Convert to numeric
                        try:
                            ds[var_name] = var_data.astype('float64')
                        except (ValueError, TypeError):
                            # If conversion fails, try to convert individual elements
                            data = var_data.values
                            numeric_data = pd.to_numeric(data.ravel(), errors='coerce').reshape(data.shape)
                            ds[var_name] = xr.DataArray(
                                numeric_data, 
                                dims=var_data.dims, 
                                coords=var_data.coords,
                                name=var_name
                            )
            
            # Fix coordinate dtypes
            for coord_name, coord_data in ds.coords.items():
                if coord_data.dtype == 'object':
                    # Check if coordinate contains string data
                    coord_values = coord_data.values.flatten()
                    non_null_values = [v for v in coord_values if pd.notna(v) and v is not None]
                    
                    if non_null_values and all(isinstance(v, str) for v in non_null_values):
                        # Convert string coordinates to Unicode string dtype
                        ds = ds.assign_coords({
                            coord_name: coord_data.astype('U')
                        })
                    elif non_null_values and all(isinstance(v, (int, float, np.integer, np.floating)) for v in non_null_values):
                        # Convert numeric coordinates
                        try:
                            ds = ds.assign_coords({
                                coord_name: coord_data.astype('float64')
                            })
                        except (ValueError, TypeError):
                            # If conversion fails, try to convert individual elements
                            numeric_coord = pd.to_numeric(coord_values, errors='coerce')
                            ds = ds.assign_coords({
                                coord_name: (coord_data.dims, numeric_coord)
                            })
            return ds
        
        # First, collect all variables across all datasets
        all_variables = set()
        for ds in datasets_to_concat:
            all_variables.update(ds.data_vars.keys())
        
        # collect dtypes for each variable and determine if variables should be strings
        dtypes_per_var = dict()

        for ds in datasets_to_concat:
            for var in ds.data_vars:
                dtypes_per_var.setdefault(var, []).append(ds.data_vars[var].dtype)
        
        
        should_be_string = dict()
        should_be_numeric = dict()
        # based on dtypes_per_var check if variable should be string or numeric. 
        # as soon as there is dtype object we convert everything to string

        # determine whether each variable should be string or numeric
        for var, dtypes in dtypes_per_var.items():
            if any(np.issubdtype(dtype, np.object_) or np.issubdtype(dtype, np.str_) for dtype in dtypes):
                should_be_string[var] = True
                should_be_numeric[var] = False
            else:
                should_be_string[var] = False
                should_be_numeric[var] = True
        
        # Convert all variables to correct dtype
        for i, ds in enumerate(datasets_to_concat):
            for var in ds.data_vars:
                if should_be_string.get(var, False):
                    ds[var] = ds[var].astype('U')
                elif should_be_numeric.get(var, False):
                    ds[var] = ds[var].astype('float64')

        # Ensure all datasets have all variables (filled with appropriate values where missing)
        standardized_datasets = []
        for i, ds in enumerate(datasets_to_concat):
            ds_copy = ds.copy()
            for var in all_variables:
                if var not in ds_copy.data_vars:
                    # Create a variable filled with appropriate missing values
                    if len(ds_copy.data_vars) > 0:
                        # Use the first available variable as template for coordinates
                        template_var = list(ds_copy.data_vars.values())[0]
                        if should_be_string.get(var, False):
                            # For string variables, use empty string instead of NaN
                            nan_var = xr.full_like(template_var, '', dtype='U')
                        else:
                            nan_var = xr.full_like(template_var, np.nan)
                        nan_var.name = var
                        ds_copy[var] = nan_var
                    else:
                        # If dataset has no variables, skip this component
                        continue
            standardized_datasets.append(ds_copy)
        
        if not standardized_datasets:
            return None
            
        # Update component names to match standardized datasets
        component_names = component_names[:len(standardized_datasets)]

        # Concatenate using join='outer'.
        # This tells xarray to:
        # 1. Create a superset of all data variables from all datasets.
        # 2. For any component that is missing a variable, automatically create
        #    it and fill it with appropriate values.
        # 3. `coords='minimal'` robustly handles different coordinates (like time).
        tracemalloc.start()

        consolidated_ds = xr.concat(
            standardized_datasets,
            dim=pd.Index(component_names, name=tech_dim_name),
            join='outer',
            coords='minimal',
            fill_value=np.nan  # Default fill value for non-string variables
        )
        
        # Post-process to fix string and numeric dtypes for both data variables and coordinates
        for var_name, should_str in should_be_string.items():
            if should_str and var_name in consolidated_ds.data_vars:
                if consolidated_ds[var_name].dtype == 'object':
                    # Convert object dtype to string dtype
                    consolidated_ds[var_name] = consolidated_ds[var_name].astype('U')
        
        for var_name, should_num in should_be_numeric.items():
            if should_num and var_name in consolidated_ds.data_vars:
                if consolidated_ds[var_name].dtype == 'object':
                    # Convert object dtype to float64 (handles both int and float)
                    try:
                        consolidated_ds[var_name] = consolidated_ds[var_name].astype('float64')
                    except (ValueError, TypeError):
                        # If conversion fails, try to convert individual elements
                        data = consolidated_ds[var_name].values
                        # Convert object array to numeric, keeping NaN for non-numeric values
                        numeric_data = pd.to_numeric(data.ravel(), errors='coerce').reshape(data.shape)
                        consolidated_ds[var_name] = xr.DataArray(
                            numeric_data, 
                            dims=consolidated_ds[var_name].dims, 
                            coords=consolidated_ds[var_name].coords,
                            name=var_name
                        )
        
        # Fix coordinate dtypes
        for coord_name, coord_data in consolidated_ds.coords.items():
            if coord_data.dtype == 'object':
                # Check if coordinate contains string data
                coord_values = coord_data.values.flatten()
                non_null_values = [v for v in coord_values if pd.notna(v) and v is not None]
                
                if non_null_values and all(isinstance(v, str) for v in non_null_values):
                    # Convert string coordinates to Unicode string dtype
                    consolidated_ds = consolidated_ds.assign_coords({
                        coord_name: coord_data.astype('U')
                    })
                elif non_null_values and all(isinstance(v, (int, float, np.integer, np.floating)) for v in non_null_values):
                    # Convert numeric coordinates
                    try:
                        consolidated_ds = consolidated_ds.assign_coords({
                            coord_name: coord_data.astype('float64')
                        })
                    except (ValueError, TypeError):
                        # If conversion fails, try to convert individual elements
                        numeric_coord = pd.to_numeric(coord_values, errors='coerce')
                        consolidated_ds = consolidated_ds.assign_coords({
                            coord_name: (coord_data.dims, numeric_coord)
                        })
        
        del standardized_datasets  # Free memory
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1e6:.2f} MB")
        print(f"Peak memory usage: {peak / 1e6:.2f} MB")
        return consolidated_ds

    # --- The rest of the function remains the same, using the new helper ---

    # 1. Prepare and Consolidate Input Data
    consolidated_input = {}
    for model_class, components in lazy_data_dict["Input"].items():
        consolidated_ds = robust_concat(components)
        if consolidated_ds:
            consolidated_input[model_class] = consolidated_ds

    # 2. Prepare and Consolidate Results Data
    consolidated_results = {}
    if "Results" in lazy_data_dict:
        for ip in lazy_data_dict['Results'].keys():
            if ip not in consolidated_results:
                consolidated_results[ip] = {}
            for model_class, components in lazy_data_dict['Results'][ip].items():
                consolidated_ds = robust_concat(components)
                if consolidated_ds:
                    consolidated_results[ip][model_class] = consolidated_ds

    print("Writing consolidated data to Zarr store...")
    
    master_chunk_scheme  = {
        'time': 1000, # One chunk for the whole time series
        'space': -1,
        'space_2': -1,
        'technology': -1
    }
    
    def _chunk(ds, chunk=False):
        if not chunk:
            return ds
        actual_chunks_for_this_ds = {
            dim: size for dim, size in master_chunk_scheme.items() if dim in ds.dims
        }

        # Apply the valid, filtered chunking scheme
        if actual_chunks_for_this_ds:
            chunked_ds = ds.chunk(actual_chunks_for_this_ds)
        else:
            # If the dataset has no dimensions that we want to chunk (e.g., only scalars)
            chunked_ds = ds
        return chunked_ds
    
                 
    for model_class, ds in consolidated_input.items():

        try:
            chunked_ds = _chunk(ds, chunk=True)

            encoding_var = {}
            for var, da in chunked_ds.data_vars.items():
                if replace_fill_value and np.issubdtype(da.dtype, np.floating):
                    encoding_var[var] = {'compressor': compressor, '_FillValue': -9999.0}
                else:
                    encoding_var[var] = {'compressor': compressor}


            chunked_ds.to_zarr(
                f"{output_zarr_path}/Input/{model_class}", mode='w',
                encoding=encoding_var
            )
        except Exception as e:
            print(e)
            chunked_ds = _chunk(ds, chunk=False)
            encoding_var = {}
            for var, da in chunked_ds.data_vars.items():
                if replace_fill_value and np.issubdtype(da.dtype, np.floating):
                    encoding_var[var] = {'compressor': compressor, '_FillValue': -9999.0}
                else:
                    encoding_var[var] = {'compressor': compressor}

            chunked_ds.to_zarr(
                f"{output_zarr_path}/Input/{model_class}", mode='w',
                encoding=encoding_var
            )

    # Write RESULTS
    for ip, ip_dict in consolidated_results.items():
        for model_class, ds in ip_dict.items():
            # Chunk the dataset if needed
            try:
                chunked_ds = _chunk(ds, chunk=True)

                encoding_var = {}
                for var, da in chunked_ds.data_vars.items():
                    if replace_fill_value and np.issubdtype(da.dtype, np.floating):
                        encoding_var[var] = {'compressor': compressor, '_FillValue': -9999.0}
                    else:
                        encoding_var[var] = {'compressor': compressor}

                chunked_ds.to_zarr(
                    f"{output_zarr_path}/Results/{ip}/{model_class}", mode='w',
                    encoding=encoding_var
                )
                
            except Exception as e:
                chunked_ds = _chunk(ds, chunk=False)

                encoding_var = {}
                for var, da in chunked_ds.data_vars.items():
                    if replace_fill_value and np.issubdtype(da.dtype, np.floating):
                        encoding_var[var] = {'compressor': compressor, '_FillValue': -9999.0}
                    else:
                        encoding_var[var] = {'compressor': compressor}

                chunked_ds.to_zarr(
                    f"{output_zarr_path}/Results/{ip}/{model_class}", mode='w',
                    encoding=encoding_var
                )
            
    # Write Parameters and Performance Summary
    params_processed = processDataset({'Parameters': lazy_data_dict['Parameters']})
    params_processed['Parameters'].to_zarr(f"{output_zarr_path}/Parameters", mode='w')
    if 'PerformanceSummary' in lazy_data_dict:
        lazy_data_dict['PerformanceSummary'].to_zarr(f"{output_zarr_path}/PerformanceSummary", mode='w')

    print("Done. (%.4f" % (time.time() - _t) + " sec)")
    
    
def readZarrToDatasets(
    zarr_path,
    lazy_load=True,
    chunks=None,
):
    """
    Reads an esM data structure from a consolidated Zarr store.

    This function reads the entire model structure, loading data lazily
    by default for high-performance analysis.

    Args:
        zarr_path (str): Path to the Zarr store directory.
        lazy_load (bool): If True, loads data as Dask arrays. If False, loads into memory.
        chunks (dict or str): Chunking scheme to use if lazy_load is True.

    Returns:
        dict: The nested dictionary of xarray.Datasets representing the esM.
    """
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found at: {zarr_path}")

    # Use the appropriate loader based on the lazy_load flag
    loader = xr.open_dataset if lazy_load else xr.load_dataset
    
    # This dictionary will hold the final reconstructed data
    xr_dss = {}

    # --- Read Input Data ---
    input_path = zarr_path / 'Input'
    if input_path.exists():
        xr_dss['Input'] = {
            model_class.name: loader(model_class, engine='zarr', chunks=chunks)
            for model_class in input_path.iterdir() if model_class.is_dir()
        }

    # --- Read Results Data ---
    results_path = zarr_path / 'Results'
    if results_path.exists():
        xr_dss['Results'] = {}
        
        for ip_path in results_path.iterdir():
            if not ip_path.is_dir():
                continue
            xr_dss['Results'][ip_path.name] = {
                model_class.name: loader(model_class, engine='zarr', chunks=chunks)
                for model_class in ip_path.iterdir() if model_class.is_dir()
            }

    # --- Read Parameters and Performance Summary ---
    params_path = zarr_path / 'Parameters'
    if params_path.exists():
        xr_dss['Parameters'] = loader(params_path, engine='zarr')

    perf_path = zarr_path / 'PerformanceSummary'
    if perf_path.exists():
        xr_dss['PerformanceSummary'] = loader(perf_path, engine='zarr')
        
    return xr_dss