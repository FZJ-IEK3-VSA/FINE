import time
from pathlib import Path
from typing import Dict

import pandas as pd
import xarray as xr
from netCDF4 import Dataset

import FINE.utils as utils
from FINE.IOManagement import dictIO, utilsIO


def convertOptimizationInputToDatasets(esM, useProcessedValues=False):
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

    # STEP 4. Add all df variables to xr_ds
    xr_dss = utilsIO.addDFVariablesToXarray(xr_dss, component_dict, df_iteration_dict)

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

    xr_dss = {"Input": xr_dss, "Parameters": attributes_xr}

    return xr_dss


def convertOptimizationOutputToDatasets(esM, optSumOutputLevel=0, optValOutputLevel=1):
    """
    Takes esM instance output and converts it into an xarray dataset.

    :param esM: EnergySystemModel instance in which the optimized model is held
    :type esM: EnergySystemModel instance

    :param optSumOutputLevel: Output level of the optimization summary (see
        EnergySystemModel). Either an integer (0,1,2) which holds for all model
        classes or a dictionary with model class names as keys and an integer
        (0,1,2) for each key (e.g. {'StorageModel':1,'SourceSinkModel':1,...}
        |br| * the default value is 2
    :type optSumOutputLevel: int (0,1,2) or dict

    :param optValOutputLevel: Output level of the optimal values. Either an
        integer (0,1) which holds for all model classes or a dictionary with
        model class names as keys and an integer (0,1) for each key (e.g.
        {'StorageModel':1,'SourceSinkModel':1,...}

        - 0: all values are kept.
        - 1: Lines containing only zeroes are dropped.

        |br| * the default value is 1
    :type optValOutputLevel: int (0,1) or dict

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
            oL_ = oL[name] if type(oL) == dict else oL
            optSum = esM.getOptimizationSummary(name, ip=ip, outputLevel=oL_)
            # if isinstance(optSum, pd.DataFrame):
            #     if esM.numberOfInvestmentPeriods != 1:
            #         raise ValueError()
            #     optSum = {}
            #     optSum[esM.startYear] = optSum
            if esM.componentModelingDict[name].dimension == "1dim":
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
                        df.index.rename("space", inplace=True)
                        df = pd.to_numeric(df)
                        xr_da = df.to_xarray()

                        # add variable [e.g. 'TAC'] and units to attributes of xarray
                        unit = variables_unit[variable]
                        xr_da.attrs[variable] = unit

                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            combine_attrs="drop_conflicts",
                        )
            elif esM.componentModelingDict[name].dimension == "2dim":
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
                        # df.name = (name, component, variable)
                        df.name = variable
                        df.index.rename(["space", "space_2"], inplace=True)
                        df = pd.to_numeric(df)
                        xr_da = df.to_xarray()

                        # add variable [e.g. 'TAC'] and units to attributes of xarray
                        unit = variables_unit[variable]
                        xr_da.attrs[variable] = unit

                        xr_dss[ip][name][component] = xr.merge(
                            [xr_dss[ip][name][component], xr_da],
                            combine_attrs="drop_conflicts",
                        )

            # Write output from esM.esM.componentModelingDict[name].getOptimalValues() to datasets
            data = esM.componentModelingDict[name].getOptimalValues(ip=ip)
            oL = optValOutputLevel
            oL_ = oL[name] if type(oL) == dict else oL
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

    xr_dss = {"Results": xr_dss}

    return xr_dss


def writeDatasetsToNetCDF(
    datasets,
    outputFilePath="my_esm.nc",
    removeExisting=False,
    mode="a",
    groupPrefix=None,
):
    """
    Saves dictionary of xarray datasets (with esM instance data) to a netCDF
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
        with Dataset(outputFilePath, "w", format="NETCDF4") as rootgrp:
            pass

    for group in datasets.keys():
        if group == "Parameters":
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
                            {f"{attr_name}.{idx}": row.to_numpy()}
                        )

                    # Delete the original attribute
                    del xarray_dataset.attrs[attr_name]

                # if the attribute is bool, add a corresponding string
                elif isinstance(attr_value, bool):
                    xarray_dataset.attrs[attr_name] = (
                        "True" if attr_value == True else "False"
                    )

                # if the attribute is None, add a corresponding string
                elif attr_value == None:
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
            continue

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
                                    var: {"zlib": True, "complevel": 9}
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
                                var: {"zlib": True, "complevel": 9}
                                for var in list(
                                    datasets[group][model][component].data_vars
                                )
                            },
                        )


def convertDatasetsToEnergySystemModel(datasets):
    """
    Takes dictionary of xarray datasets (with esM instance data) and converts
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
            optSum = {}
            operationVariablesOptimum_dict = {}
            capacityVariablesOptimum_dict = {}
            isBuiltVariablesOptimum_dict = {}
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
                                ][variable]["space"].values
                            ]
                            idx = pd.MultiIndex.from_tuples(tuple(iterables2))
                            _optSum_df.index = idx
                            _optSum_df.index.names = [
                                "Component",
                                "Property",
                                "Unit",
                                "LocationIn",
                            ]
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
                            _optSum_df.index = pd.MultiIndex.from_tuples(iterables)
                            _optSum_df.index.names = ["Component", "Property", "Unit"]
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

                setattr(esM.componentModelingDict[model], "_optSummary", optSum)

                # read optimal Values (3 types exist)
                operationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                capacityVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                isBuiltVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                chargeOperationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                dischargeOperationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])
                stateOfChargeOperationVariablesOptimum_dict[int(ip)] = pd.DataFrame([])

                for component in datasets["Results"][ip][model]:
                    _operationVariablesOptimum_df = pd.DataFrame([])
                    _capacityVariablesOptimum_df = pd.DataFrame([])
                    _isBuiltVariablesOptimum_df = pd.DataFrame([])
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

                        if opt_variable == "_operationVariablesOptimum":
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
                                    )

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
                                )

                        if opt_variable == "_capacityVariablesOptimum":
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
                                _capacityVariablesOptimum_df = _df
                            else:
                                _capacityVariablesOptimum_df = xr_opt.to_dataframe().T
                                _capacityVariablesOptimum_df = (
                                    _capacityVariablesOptimum_df.set_axis([component])
                                )

                        if opt_variable == "_isBuiltVariablesOptimum":
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
                            )

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
                                _chargeOperationVariablesOptimum_df.set_index(idx)
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
                                _dischargeOperationVariablesOptimum_df.set_index(idx)
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
                                _stateOfChargeOperationVariablesOptimum_df.set_index(
                                    idx
                                )
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
    optValOutputLevel=1,
    groupPrefix=None,
):
    """
    Write energySystemModel (input and if exists, output) to netCDF file.

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

    :param optValOutputLevel: Output level of the optimal values. Either an
        integer (0,1) which holds for all model classes or a dictionary with
        model class names as keys and an integer (0,1) for each key (e.g.
        {'StorageModel':1,'SourceSinkModel':1,...}

        * 0: all values are kept.
        * 1: Lines containing only zeroes are dropped.

        |br| * the default value is 1
    :type optValOutputLevel: int (0,1) or dict

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
    if esM.objectiveValue != None:  # model was optimized
        xr_dss_output = convertOptimizationOutputToDatasets(
            esM, optSumOutputLevel, optValOutputLevel
        )
        writeDatasetsToNetCDF(xr_dss_output, outputFilePath, groupPrefix=groupPrefix)

    utils.output("Done. (%.4f" % (time.time() - _t) + " sec)", esM.verbose, 0)


def writeEnergySystemModelToDatasets(esM):
    """Converts esM instance (input and output) into a xarray dataset.

    :param esM: EnergySystemModel instance in which the optimized model is held
    :type esM: EnergySystemModel instance

    :return: xr_dss_results - esM instance (input and output) data in xarray
        dataset format
    :rtype: xr.DataSet
    """
    if esM.objectiveValue != None:  # model was optimized
        xr_dss_output = convertOptimizationOutputToDatasets(esM)
        xr_dss_input = convertOptimizationInputToDatasets(esM)
        xr_dss_results = {
            "Results": xr_dss_output["Results"],
            "Input": xr_dss_input["Input"],
            "Parameters": xr_dss_input["Parameters"],
        }
    else:
        xr_dss_input = convertOptimizationInputToDatasets(esM)
        xr_dss_results = {
            "Input": xr_dss_input["Input"],
            "Parameters": xr_dss_input["Parameters"],
        }

    return xr_dss_results


def readNetCDFToDatasets(filePath="my_esm.nc", groupPrefix=None):
    """
    Read optimization results from grouped netCDF file to dictionary of
    xr.Datasets.

    :param filePath: output file name of netCDF file (can include full path)
        |br| * the default value is "my_esm.nc"
    :type filePath: string

    :param groupPrefix: if specified, multiple xarray datasets (with esM
        instance data) are saved to the same netcdf file. The dictionary
        structure is then {group_prefix}/{group}/{...} instead of {group}/{...}
        |br| * the default value is None
    :type groupPrefix: string

    :return: Nested dictionary containing an xr.Dataset with all result values
        for each component.
    :rtype: Nested dict
    """

    with Dataset(filePath, "r", format="NETCDF4") as rootgrp:
        if groupPrefix:
            group_keys = rootgrp[groupPrefix].groups
        else:
            group_keys = rootgrp.groups

    if not groupPrefix:
        xr_dss = {}
        # read input from netcdf
        xr_dss["Input"] = {
            model_key: {
                comp_key: xr.load_dataset(
                    filePath, group=f"Input/{model_key}/{comp_key}"
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
                        comp_key: xr.load_dataset(
                            filePath, group=f"Results/{ip_key}/{model_key}/{comp_key}"
                        )
                        for comp_key in group_keys["Results"][ip_key][model_key].groups
                    }
                    for model_key in group_keys["Results"][ip_key].groups
                }
                for ip_key in group_keys["Results"].groups
            }
        # read parameters from netcdf
        xr_dss["Parameters"] = xr.load_dataset(filePath, group=f"Parameters")
    else:
        xr_dss = {}
        # read input from netcdf
        xr_dss["Input"] = {
            model_key: {
                comp_key: xr.load_dataset(
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
                        comp_key: xr.load_dataset(
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

    return xr_dss


def readNetCDFtoEnergySystemModel(filePath, groupPrefix=None):
    """
    Converts netCDF file into an EnergySystemModel instance.

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
