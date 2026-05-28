import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce  # forward compatibility for Python 3
import operator


def getFromDict(dataDict, mapList):
    """Get value from a dict by a list, which contains the dict keys.

    e.g. for dict={'a': {'b': 1}} with mapList ['a','b'] the function returns 1

    :param dataDict: nested dict, e.g. {'a': {'b'}
    :type dataDict: dict

    :param mapList: list with dictionary keys
    :type mapList: list
    """
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    """Set a value in a nested dict, where mapList contains the dict keys.

    e.g. for dict={'a': {'b': 1}} with mapList ['a','b'] and value 2, the function sets dict={'a': {'b': 2}}

    :param dataDict: nested dict, e.g. {'a': {'b'}
    :type dataDict: dict

    :param mapList: list with dictionary keys
    :type mapList: list
    """
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def getKeyHierarchyOfNestedDict(
    variable_description,
):
    """Get a list of dictionary keys for a nested dict from the variable description.

    e.g. 'processedCapacityMax.0.1' leads to ['processedCapacityMax', 0, 1]

    :param variable_description: variable description
    :type variable_description: str
    """
    if variable_description.count(".") == 0:
        key_list = [variable_description]
    elif variable_description.count(".") >= 1:
        key_list = variable_description.split(".")

    # for (commis, ip) dependency: string to tuple
    key_list = [eval(x) if x.startswith("(") else x for x in key_list]

    # for ip: string of digits to ints
    key_list = [
        int(x) if (not isinstance(x, tuple) and x.isdigit()) else x for x in key_list
    ]

    return key_list  # noqa: RET504


def _get_base_name_and_ip(variable_description):
    key_list = getKeyHierarchyOfNestedDict(variable_description)

    if len(key_list) > 1 and isinstance(key_list[1], int):
        ip = key_list[1]
        base_key_list = [key_list[0]] + key_list[2:]  # strip ip from middle
        base_name = ".".join(str(k) for k in base_key_list)  # rejoin remainder
        return base_name, ip

    return variable_description, None


def getListsOfKeyPathsInNestedDict(data_dict, variable_name):
    """Get a list of all paths in a nested dict, starting after the variable_name,
    until the next value is not a dict anymore.

    e.g. variable_name='a' and data_dict ={
        'a': {
            'b':{'c':1},
            'f':{'g':1}
        }
        returns: [['b','c'],['f','g']]

    :param data_dict: dict with data
    :type data_dict: dict

    :param variable_name: name of variable, as key in the dict
    :type variable_name: string or int
    """
    if isinstance(data_dict[variable_name], dict):
        key_lists_in_nested_dict = []
        # either for ip dependency or for commodity conversion factors
        for key1, data1 in data_dict[variable_name].items():
            if isinstance(data1, dict):
                # for commodity conversion factors which are ip depending -> 3 levels
                # {"commodityConversionFactors":{ip:{"electricity":1,"hydrogen":1}}}}}
                for key2, data2 in data1.items():
                    if isinstance(data2, dict):
                        # for commodity conversion factors of flexible conversion components
                        # which are ip depending -> 4 levels
                        # {"commodityConversionFactors":{ip:{"group1":{"electricity":1,"hydrogen":1}}}}}}
                        for key3, data3 in data2.items():
                            key_lists_in_nested_dict.append(
                                [variable_name, key1, key2, key3]
                            )
                    else:
                        key_lists_in_nested_dict.append([variable_name, key1, key2])
            else:
                key_lists_in_nested_dict.append([variable_name, key1])
        return key_lists_in_nested_dict
    return [[variable_name]]


def transform1dSeriesto2dDataFrame(series, locations):
    """Expand pandas Series into a pandas DataFrame.

    :param series: the series that need to be converted
    :type series: pd.Series

    :param locations: sorted esM locations
    :type locations: list

    :return: df - converted pandas DataFrame

    """
    values = np.zeros((len(locations), len(locations)))

    df = pd.DataFrame(values, columns=locations, index=locations)

    for row in series.items():
        # Seperate loc1_loc2
        loc = ""

        for n in range(len(row[0])):
            loc += row[0][n]
            if (loc in locations) & (row[0][n + 1] == "_"):
                id_1, id_2 = row[0][: n + 1], row[0][n + 2 :]
                break
        df.loc[id_1, id_2] = row[1]

    return df


class PowerDict(dict):
    """Dictionary with additional functions.
    Helps in creating nested dictionaries on the fly.
    """

    def __init__(self, parent=None, key=None):
        self.parent = parent
        self.key = key

    def __missing__(self, key):
        """Creation of subdictionaries on fly."""
        self[key] = PowerDict(self, key)
        return self[key]

    def append(self, item):
        """Additional append function for lists in dict."""
        self.parent[self.key] = [item]

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        if isinstance(val, PowerDict):
            val.parent = self
            val.key = key


def generateIterationDicts(component_dict, investmentPeriods):
    """Create iteration dictionaries that contain descriptions of all
    dataframes, series, and constants present in component_dict.

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param investmentPeriods: investment periods
    :type investmentPeriods: list

    :return: df_iteration_dict, series_iteration_dict, constants_iteration_dict
    """
    df_iteration_dict, series_iteration_dict, constants_iteration_dict = {}, {}, {}

    # Loop through every class-component-variable combination
    for classname in component_dict:
        for component in component_dict[classname]:
            for variable_description in component_dict[classname][component].keys():
                # 1. iterate through nested dict levels until constant, series or df, add
                # 1. find list of keys in nested dict level
                key_lists = getListsOfKeyPathsInNestedDict(
                    component_dict[classname][component],
                    variable_name=variable_description,
                )

                # iterate over all key-"paths" in nested dict
                for key_list in key_lists:
                    _variable_description = ".".join(map(str, key_list))

                    description_tuple = (classname, component)

                    # add to the corresponding dicts
                    data = getFromDict(component_dict[classname][component], key_list)

                    # 1 add dataframes
                    if isinstance(data, pd.DataFrame):
                        if _variable_description not in df_iteration_dict.keys():
                            df_iteration_dict[_variable_description] = [
                                description_tuple
                            ]
                        else:
                            df_iteration_dict[_variable_description].append(
                                description_tuple
                            )
                    # 2 add series
                    elif isinstance(data, pd.Series):
                        if _variable_description not in series_iteration_dict.keys():
                            series_iteration_dict[_variable_description] = [
                                description_tuple
                            ]
                        else:
                            series_iteration_dict[_variable_description].append(
                                description_tuple
                            )
                    # 3 add constant
                    elif _variable_description not in constants_iteration_dict.keys():
                        constants_iteration_dict[_variable_description] = [
                            description_tuple
                        ]
                    else:
                        constants_iteration_dict[_variable_description].append(
                            description_tuple
                        )

    return df_iteration_dict, series_iteration_dict, constants_iteration_dict


def addDFVariablesToXarray(
    xr_ds, component_dict, df_iteration_dict, _mapC_dict, locations
):
    """Add all variables whose data is contained in a pd.DataFrame to xarray dataset.

    These variables are normally regional time series (dimensions - space, time)

    :param xr_ds: xarray dataset or a dict of xarray datasets to which the DF variables should be added
    :type xr_ds: xr.Dataset/dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param df_iteration_dict: dictionary with:
        keys - DF variable names
        values - list of tuple of component class and component name
    :type df_iteration_dict: dict

    :param locations: esM locations
    :type locations: list

    :return: xr_ds
    """
    # Group all variable descriptions by base name (stripping ip suffix if present)
    grouped = {}
    for variable_description in df_iteration_dict:
        base_name, ip_value = _get_base_name_and_ip(variable_description)
        if base_name not in grouped:
            grouped[base_name] = {}
        grouped[base_name][ip_value] = variable_description

    def _add_ip_independent_to_xarray(xr_ds, df_dict, base_name):
        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True)
        ds_component = xr.Dataset()
        ds_component[f"ts_{base_name}"] = df_variable.sort_index().to_xarray()
        for comp in df_variable.index.get_level_values(0).unique():
            this_class = comp.split("; ")[0]
            this_comp = comp.split("; ")[1]
            this_ds_component = ds_component.sel(component=comp).drop_vars("component")
            try:
                xr_ds[this_class][this_comp] = xr.merge(
                    [xr_ds[this_class][this_comp], this_ds_component]
                )
            except Exception:
                pass
        return xr_ds

    for base_name, description_by_ip in grouped.items():
        ip_independent_description = description_by_ip.get(None)
        ip_dependent_descriptions = {
            k: v for k, v in description_by_ip.items() if k is not None
        }

        # ── ip-independent path ──────────────────────────────────────────────
        if ip_independent_description is not None:
            df_dict = {}
            df_dict_3dim = {}

            for classname, component in df_iteration_dict[ip_independent_description]:
                df_description = f"{classname}; {component}"
                key_list = getKeyHierarchyOfNestedDict(ip_independent_description)
                data = component_dict[classname][component]
                for key in key_list:
                    data = data[key]

                multi_index_dataframe = data.stack()
                if "Period" in multi_index_dataframe.index.names:
                    multi_index_dataframe = multi_index_dataframe.droplevel(0)
                multi_index_dataframe.index.set_names("time", level=0, inplace=True)
                multi_index_dataframe.index.set_names("space", level=1, inplace=True)

                if classname in ["Transmission", "LinearOptimalPowerFlow"]:
                    # use _mapC to split via location names
                    space_index = multi_index_dataframe.index.get_level_values("space")
                    time_index = multi_index_dataframe.index.get_level_values("time")
                    # reconstruct multiindex
                    space_index_split = []
                    for idx in space_index:
                        loc1, loc2 = _mapC_dict[component][idx]
                        space_index_split.append((loc1, loc2))
                    multi_index_dataframe.index = pd.MultiIndex.from_tuples(
                        [
                            (
                                time_index[i],
                                space_index_split[i][0],
                                space_index_split[i][1],
                            )
                            for i in range(len(space_index_split))
                        ],
                        names=["time", "space", "space_2"],
                    )
                    df_dict_3dim[df_description] = multi_index_dataframe
                else:
                    df_dict[df_description] = multi_index_dataframe

            if df_dict:
                xr_ds = _add_ip_independent_to_xarray(xr_ds, df_dict, base_name)
            if df_dict_3dim:
                xr_ds = _add_ip_independent_to_xarray(xr_ds, df_dict_3dim, base_name)

        # ── ip-dependent path ────────────────────────────────────────────────
        if ip_dependent_descriptions:
            comp_ip_dict = {}  # {comp_desc: {ip_str: Series}} for (time, space)
            comp_ip_dict_3dim = {}  # {comp_desc: {ip_str: Series}} for (time, space, space_2)

            for ip_value, variable_description in ip_dependent_descriptions.items():
                ip_str = str(ip_value)
                for classname, component in df_iteration_dict[variable_description]:
                    df_description = f"{classname}; {component}"
                    key_list = getKeyHierarchyOfNestedDict(variable_description)
                    data = component_dict[classname][component]
                    for key in key_list:
                        data = data[key]

                    multi_index_dataframe = data.stack()
                    if "Period" in multi_index_dataframe.index.names:
                        multi_index_dataframe = multi_index_dataframe.droplevel(0)
                    multi_index_dataframe.index.set_names("time", level=0, inplace=True)
                    multi_index_dataframe.index.set_names(
                        "space", level=1, inplace=True
                    )

                    if classname in ["Transmission", "LinearOptimalPowerFlow"]:
                        space_index = multi_index_dataframe.index.get_level_values(
                            "space"
                        )
                        time_index = multi_index_dataframe.index.get_level_values(
                            "time"
                        )
                        space_index_split = []
                        for idx in space_index:
                            loc1, loc2 = _mapC_dict[component][idx]
                            space_index_split.append((loc1, loc2))
                        multi_index_dataframe.index = pd.MultiIndex.from_tuples(
                            [
                                (
                                    time_index[i],
                                    space_index_split[i][0],
                                    space_index_split[i][1],
                                )
                                for i in range(len(space_index_split))
                            ],
                            names=["time", "space", "space_2"],
                        )
                        comp_ip_dict_3dim.setdefault(df_description, {})[ip_str] = (
                            multi_index_dataframe
                        )
                    else:
                        comp_ip_dict.setdefault(df_description, {})[ip_str] = (
                            multi_index_dataframe
                        )

            if comp_ip_dict:
                frames = {
                    comp_desc: pd.DataFrame(ip_dict).rename_axis(columns="ip")
                    for comp_desc, ip_dict in comp_ip_dict.items()
                }
                combined = pd.concat(frames)
                combined.index.set_names(["component", "time", "space"], inplace=True)
                stacked = combined.stack().rename_axis(
                    ["component", "time", "space", "ip"]
                )
                ds_component = xr.Dataset()
                ds_component[f"ts_{base_name}"] = stacked.to_xarray()
                for comp in ds_component[f"ts_{base_name}"].coords["component"].values:
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

            if comp_ip_dict_3dim:
                frames = {
                    comp_desc: pd.DataFrame(ip_dict).rename_axis(columns="ip")
                    for comp_desc, ip_dict in comp_ip_dict_3dim.items()
                }
                combined = pd.concat(frames)
                combined.index.set_names(
                    ["component", "time", "space", "space_2"], inplace=True
                )
                stacked = combined.stack().rename_axis(
                    ["component", "time", "space", "space_2", "ip"]
                )
                ds_component = xr.Dataset()
                ds_component[f"ts_{base_name}"] = stacked.to_xarray()
                for comp in ds_component[f"ts_{base_name}"].coords["component"].values:
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

    return xr_ds


def addSeriesVariablesToXarray(xr_ds, component_dict, series_iteration_dict, locations):
    """Add all variables whose data is contained in a pd.Series to xarray dataset.

    These variables can be either:
        - 2d (dimensions - space, space). Series indices in this case are packed like loc1_loc2
        or
        - 1d (dimension - space)
        or
        - time series (dimension - time). This situation is unique to single node esM model

    :param xr_ds: xarray dataset or a dict of xarray datasets to which the series variables should be added
    :type xr_ds: xr.Dataset/dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param series_iteration_dict: dictionary with:
        keys - series variable names
        values - list of tuple of component class and component name
    :type series_iteration_dict: dict

    :param locations: sorted esM locations
    :type locations: list

    :return: xr_ds
    """
    # Group all variable descriptions by base name (stripping ip suffix if present)
    grouped = {}
    for variable_description in series_iteration_dict:
        base_name, ip_value = _get_base_name_and_ip(variable_description)
        if base_name not in grouped:
            grouped[base_name] = {}
        grouped[base_name][ip_value] = variable_description

    for base_name, description_by_ip in grouped.items():
        ip_independent_description = description_by_ip.get(None)
        ip_dependent_descriptions = {
            k: v for k, v in description_by_ip.items() if k is not None
        }

        # ── ip-independent path ──────────────────────────────────────────────
        if ip_independent_description is not None:
            space_space_dict = {}
            space_dict = {}
            time_dict = {}

            for classname, component in series_iteration_dict[
                ip_independent_description
            ]:
                df_description = f"{classname}; {component}"
                key_list = getKeyHierarchyOfNestedDict(ip_independent_description)
                data = component_dict[classname][component]
                for item in key_list:
                    data = data[item]

                if classname in ["Transmission", "LinearOptimalPowerFlow"]:
                    df = transform1dSeriesto2dDataFrame(data, locations)
                    multi_index_dataframe = df.stack()
                    multi_index_dataframe.index.set_names(
                        ["space", "space_2"], inplace=True
                    )
                    space_space_dict[df_description] = multi_index_dataframe
                elif set(data.index.values).issubset(set(locations)):
                    space_dict[df_description] = data.rename_axis("space")
                else:
                    time_dict[df_description] = data.rename_axis("time")
                    time_dict[df_description] = pd.concat(
                        {locations[0]: time_dict[df_description]}, names=["space"]
                    )
                    time_dict[df_description] = time_dict[
                        df_description
                    ].reorder_levels(["time", "space"])

            if len(space_space_dict) > 0:
                df_variable = pd.concat(space_space_dict)
                df_variable.index.set_names("component", level=0, inplace=True)
                ds_component = xr.Dataset()
                ds_component[f"2d_{base_name}"] = df_variable.sort_index().to_xarray()
                for comp in df_variable.index.get_level_values(0).unique():
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

            if len(space_dict) > 0:
                df_variable = pd.concat(space_dict)
                df_variable.index.set_names("component", level=0, inplace=True)
                ds_component = xr.Dataset()
                ds_component[f"1d_{base_name}"] = df_variable.sort_index().to_xarray()
                for comp in df_variable.index.get_level_values(0).unique():
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

            if len(time_dict) > 0:
                df_variable = pd.concat(time_dict)
                df_variable.index.set_names("component", level=0, inplace=True)
                ds_component = xr.Dataset()
                ds_component[f"ts_{base_name}"] = df_variable.sort_index().to_xarray()
                for comp in df_variable.index.get_level_values(0).unique():
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

        # ── ip-dependent path ────────────────────────────────────────────────
        if ip_dependent_descriptions:
            # Collect {comp_desc: {ip_str: Series}} for each data type
            space_space_comp_ip = {}
            space_comp_ip = {}
            time_comp_ip = {}

            for ip_value, variable_description in ip_dependent_descriptions.items():
                ip_str = str(ip_value)
                for classname, component in series_iteration_dict[variable_description]:
                    df_description = f"{classname}; {component}"
                    key_list = getKeyHierarchyOfNestedDict(variable_description)
                    data = component_dict[classname][component]
                    for item in key_list:
                        data = data[item]

                    if classname in ["Transmission", "LinearOptimalPowerFlow"]:
                        df = transform1dSeriesto2dDataFrame(data, locations)
                        mi_s = df.stack()
                        mi_s.index.set_names(["space", "space_2"], inplace=True)
                        space_space_comp_ip.setdefault(df_description, {})[ip_str] = (
                            mi_s
                        )
                    elif set(data.index.values).issubset(set(locations)):
                        space_comp_ip.setdefault(df_description, {})[ip_str] = (
                            data.rename_axis("space")
                        )
                    else:
                        ts = data.rename_axis("time")
                        ts = pd.concat({locations[0]: ts}, names=["space"])
                        ts = ts.reorder_levels(["time", "space"])
                        time_comp_ip.setdefault(df_description, {})[ip_str] = ts

            if space_comp_ip:
                # Build DataFrame per component: index=space, columns=ip strings
                # concat into MultiIndex(component, space) x ip, then stack ip into index
                frames = {
                    comp_desc: pd.DataFrame(ip_dict).rename_axis(columns="ip")
                    for comp_desc, ip_dict in space_comp_ip.items()
                }
                combined = pd.concat(frames)
                combined.index.set_names(["component", "space"], inplace=True)
                stacked = combined.stack().rename_axis(["component", "space", "ip"])
                ds_component = xr.Dataset()
                ds_component[f"1d_{base_name}"] = stacked.to_xarray()
                for comp in ds_component[f"1d_{base_name}"].coords["component"].values:
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

            if space_space_comp_ip:
                frames = {
                    comp_desc: pd.DataFrame(ip_dict).rename_axis(columns="ip")
                    for comp_desc, ip_dict in space_space_comp_ip.items()
                }
                combined = pd.concat(frames)
                combined.index.set_names(
                    ["component", "space", "space_2"], inplace=True
                )
                stacked = combined.stack().rename_axis(
                    ["component", "space", "space_2", "ip"]
                )
                ds_component = xr.Dataset()
                ds_component[f"2d_{base_name}"] = stacked.to_xarray()
                for comp in ds_component[f"2d_{base_name}"].coords["component"].values:
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

            if time_comp_ip:
                frames = {
                    comp_desc: pd.DataFrame(ip_dict).rename_axis(columns="ip")
                    for comp_desc, ip_dict in time_comp_ip.items()
                }
                combined = pd.concat(frames)
                combined.index.set_names(["component", "time", "space"], inplace=True)
                stacked = combined.stack().rename_axis(
                    ["component", "time", "space", "ip"]
                )
                ds_component = xr.Dataset()
                ds_component[f"ts_{base_name}"] = stacked.to_xarray()
                for comp in ds_component[f"ts_{base_name}"].coords["component"].values:
                    this_class = comp.split("; ")[0]
                    this_comp = comp.split("; ")[1]
                    this_ds_component = ds_component.sel(component=comp).drop_vars(
                        "component"
                    )
                    try:
                        xr_ds[this_class][this_comp] = xr.merge(
                            [xr_ds[this_class][this_comp], this_ds_component]
                        )
                    except Exception:
                        pass

    return xr_ds


def addConstantsToXarray(
    xr_ds, component_dict, constants_iteration_dict, useProcessedValues
):
    """Add all variables whose data is just a constant value, to xarray dataset.

    :param xr_ds: A dict of xarray datasets to which the constant value variables should be added
    :type xr_ds: dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param constants_iteration_dict: dictionary with:
        keys - constant value variable names
        values - list of tuple of component class and component name
    :type constants_iteration_dict: dict

    :return: xr_ds
    """
    # Group all variable descriptions by base name (stripping ip suffix if present)
    grouped = {}
    for variable_description in constants_iteration_dict:
        base_name, ip_value = _get_base_name_and_ip(variable_description)
        if base_name not in grouped:
            grouped[base_name] = {}
        grouped[base_name][ip_value] = variable_description

    for base_name, description_by_ip in grouped.items():
        ip_independent_description = description_by_ip.get(None)
        ip_dependent_descriptions = {
            k: v for k, v in description_by_ip.items() if k is not None
        }

        datasets_to_merge = []

        if ip_independent_description is not None:
            # ip-independent: gather data into a Series, write as scalar DataArray
            df_dict = {}
            for classname, component in constants_iteration_dict[
                ip_independent_description
            ]:
                df_description = f"{classname}; {component}"
                key_list = getKeyHierarchyOfNestedDict(ip_independent_description)
                data = component_dict[classname][component]
                for item in key_list:
                    data = data[item]
                df_dict[df_description] = data

            df_variable = pd.Series(df_dict)
            df_variable.index.set_names("component", inplace=True)

            ds_none = xr.Dataset()
            ds_none[f"0d_{base_name}"] = xr.DataArray.from_series(df_variable)
            datasets_to_merge.append(ds_none)

        if ip_dependent_descriptions:
            # ip-dependent: gather one Series per ip, combine into DataFrame, write with ip dim
            ip_series = {}
            for ip_value, variable_description in ip_dependent_descriptions.items():
                df_dict = {}
                for classname, component in constants_iteration_dict[
                    variable_description
                ]:
                    df_description = f"{classname}; {component}"
                    key_list = getKeyHierarchyOfNestedDict(variable_description)
                    data = component_dict[classname][component]
                    for item in key_list:
                        data = data[item]
                    df_dict[df_description] = data
                ip_series[str(ip_value)] = pd.Series(df_dict)

            df_combined = pd.DataFrame(ip_series)
            df_combined.index.set_names("component", inplace=True)
            df_combined.columns.set_names("ip", inplace=True)

            ds_ip = xr.Dataset()
            ds_ip[f"0d_{base_name}"] = xr.DataArray(
                df_combined.values,
                coords={"component": df_combined.index, "ip": df_combined.columns},
                dims=["component", "ip"],
            )
            datasets_to_merge.append(ds_ip)

        # Merge each component's slice into xr_ds (covers both scalar and ip-dim datasets)
        for ds_component in datasets_to_merge:
            for comp in ds_component[f"0d_{base_name}"].coords["component"].values:
                this_class = comp.split("; ")[0]
                this_comp = comp.split("; ")[1]
                this_ds_component = ds_component.sel(component=comp).drop_vars(
                    "component"
                )

                try:
                    xr_ds[this_class][this_comp] = xr.merge(
                        [xr_ds[this_class][this_comp], this_ds_component]
                    )
                except Exception:
                    pass

    return xr_ds


def processXarrayAttributes(xarray_dataset):
    """Convert non-serializable data types such as sets, dicts, bools, pandas DataFrames/Series, and NoneType to lists
    or strings when saving, and convert them back to their original formats when setting up the esM instance.

    :param xarray_dataset: The xarray datasets holding all data required to set up an esM instance.
    :type xarray_dataset: Dict[xr.Dataset]

    :return: xarray_dataset
    """
    _xarray_dataset = (
        xarray_dataset.copy()
    )  # Copying to avoid errors due to change of size during iteration

    dot_attrs_dict = PowerDict()
    keys_to_delete = []

    # STEP 1. Loop through each attribute, convert datatypes
    # or append to dot_attrs_dict for conversion in a later step
    balanceLimit_dict = {}
    balanceLimit_columns = None
    balanceLimit_dtypes = {}
    hasBalanceLimit = False
    for attr_name, attr_value in _xarray_dataset.attrs.items():
        if "balanceLimit" in attr_name:
            if attr_name == "balanceLimit_index":
                keys_to_delete.append("balanceLimit_index")
                continue
            if attr_name == "balanceLimit_columns":
                balanceLimit_columns = attr_value
                keys_to_delete.append("balanceLimit_columns")
            elif attr_name == "balanceLimit_dtypes":
                balanceLimit_dtypes = attr_value
                keys_to_delete.append("balanceLimit_dtypes")
            else:
                balanceLimit_dict[attr_name.replace("balanceLimit.", "")] = attr_value
                keys_to_delete.append(attr_name)
                hasBalanceLimit = True

    if hasBalanceLimit:
        balanceLimit_df = None
    else:
        balanceLimit_df = pd.DataFrame(
            data=balanceLimit_dict, index=balanceLimit_columns
        ).T
        for column, dtype in zip(balanceLimit_df.columns, balanceLimit_dtypes):
            balanceLimit_df[column] = balanceLimit_df[column].astype(dtype)

    for attr_name, attr_value in _xarray_dataset.attrs.items():
        if "balanceLimit" in attr_name:
            continue
        if attr_name in ["locations", "commodities"] and isinstance(attr_value, str):
            xarray_dataset.attrs[attr_name] = set([attr_value])
        if attr_name in ["commodityUnitsDict"] and isinstance(attr_value, str):
            [k, v] = attr_value.split(" : ")
            _dict = {k: v}
            xarray_dataset.attrs[attr_name] = _dict

        elif isinstance(attr_value, list):
            # If its a "flattened" list, convert it to dict
            if all(":" in v for v in attr_value):
                _dict = {}
                for item in attr_value:
                    [k, v] = item.split(" : ")
                    _dict.update({k: v})

                xarray_dataset.attrs[attr_name] = _dict

            # Otherwise, convert it to set
            else:
                xarray_dataset.attrs[attr_name] = set(attr_value)

        # sometimes ints are converted to numpy numbers while saving, but these should strictly be ints
        elif isinstance(attr_value, np.number):
            xarray_dataset.attrs[attr_name] = int(attr_value)

        # convert string values
        elif isinstance(attr_value, str):
            if attr_value == "None":
                xarray_dataset.attrs[attr_name] = None

            elif attr_value == "True":
                xarray_dataset.attrs[attr_name] = True

            elif attr_value == "False":
                xarray_dataset.attrs[attr_name] = False

        # if there is a . in attr_name, collect the values in dot_attrs_dict
        # to reconstruct pandas series or df later
        if "." in attr_name:
            [new_attr_name, sub_attr_name] = attr_name.split(".")
            dot_attrs_dict[new_attr_name][sub_attr_name] = attr_value

            keys_to_delete.append(attr_name)

    # STEP 2. Reconstruct pandas series or df for each item in dot_attrs_dict
    if len(dot_attrs_dict) > 0:
        for new_attr_name, new_attr_dict in dot_attrs_dict.items():
            if all(
                [
                    isinstance(value, np.ndarray)
                    for value in list(new_attr_dict.values())
                ]
            ):
                data = np.stack(new_attr_dict.values())
                columns = sorted(xarray_dataset.attrs["locations"])
                index = new_attr_dict.keys()

                df = pd.DataFrame(data, columns=columns, index=index)

                xarray_dataset.attrs.update({new_attr_name: df})

            else:
                series = pd.Series(new_attr_dict)
                xarray_dataset.attrs.update({new_attr_name: series})

        # cleaning up the many keys
    for key in keys_to_delete:
        xarray_dataset.attrs.pop(key)

    xarray_dataset.attrs["balanceLimit"] = balanceLimit_df

    return xarray_dataset


def addTimeSeriesVariableToDict(
    component_dict, comp_var_xr, component, variable, drop_component=True
):
    """Convert the time series variable data to required format and add it to component_dict.

    :param component_dict: The dict to which the variable data needs to be added
    :type component_dict: dict

    :param comp_var_xr: The xarray DataArray that holds the data
    :type comp_var_xr: xr.DataArray

    :param component: The component name corresponding to the variable
    :type component: string

    :param variable: The variable name
    :type variable: string

    :return: component_dict
    """
    # ip-dependent: one DataFrame per ip stored along the "ip" dimension
    if "ip" in comp_var_xr.dims:
        class_name = component.split("; ")[0]
        comp_name = component.split("; ")[1]
        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]
        for ip_str in comp_var_xr.coords["ip"].values:
            da_ip = comp_var_xr.sel(ip=ip_str).drop_vars("ip")
            if "space_2" in da_ip.dims:
                ip_df = da_ip.to_dataframe().squeeze()
                space_index = ip_df.index.get_level_values("space")
                space_2_index = ip_df.index.get_level_values("space_2")
                new_space_index = [
                    f"{space_index[i]}_{space_2_index[i]}"
                    for i in range(len(space_index))
                ]
                ip_df.index = pd.MultiIndex.from_tuples(
                    [
                        (ip_df.index.get_level_values("time")[i], new_space_index[i])
                        for i in range(len(new_space_index))
                    ],
                    names=["time", "space"],
                )
                ip_df = ip_df.unstack()
                ip_df = ip_df.dropna(axis=1, how="all")
            elif len(da_ip.space.dims) == 0:
                ip_df = da_ip.to_series()
            else:
                ip_df = da_ip.to_dataframe().unstack(level=1)
                if isinstance(ip_df, pd.DataFrame):
                    if len(ip_df.columns) > 1:
                        ip_df.columns = ip_df.columns.droplevel(0)
            ip_key_list = [key_list[0], int(ip_str)] + key_list[1:]
            setInDict(
                component_dict[class_name][comp_name], ip_key_list, ip_df.sort_index()
            )
        return component_dict

    if len(comp_var_xr.space.dims) == 0:
        df = comp_var_xr.to_series()
    elif drop_component:
        df = comp_var_xr.drop("component").to_dataframe().unstack(level=1)
    elif "space_2" in comp_var_xr.dims:
        df = comp_var_xr.to_dataframe().squeeze()
        # merge space and space_2 levels
        space_index = df.index.get_level_values("space")
        space_2_index = df.index.get_level_values("space_2")
        new_space_index = [
            f"{space_index[i]}_{space_2_index[i]}" for i in range(len(space_index))
        ]
        df.index = pd.MultiIndex.from_tuples(
            [
                (df.index.get_level_values("time")[i], new_space_index[i])
                for i in range(len(new_space_index))
            ],
            names=["time", "space"],
        )
        df = df.unstack()
        df = df.dropna(axis=1, how="all")
    else:
        df = comp_var_xr.to_dataframe().unstack(level=1)

    if isinstance(df, pd.DataFrame) and "space_2" not in comp_var_xr.dims:
        if len(df.columns) > 1:
            df.columns = df.columns.droplevel(0)

    class_name = component.split("; ")[0]
    comp_name = component.split("; ")[1]

    key_list = getKeyHierarchyOfNestedDict(variable)

    key_list[0] = key_list[0][3:]

    # update the dict value
    setInDict(component_dict[class_name][comp_name], key_list, df.sort_index())

    # NOTE: Thanks to PowerDict(), the nested dictionaries need not be created before adding the data.

    return component_dict


def add2dVariableToDict(
    component_dict, comp_var_xr, component, variable, drop_component=True
):
    """Convert the 2d variable data to required format and add it to component_dict.

    :param component_dict: The dict to which the variable data needs to be added
    :type component_dict: dict

    :param comp_var_xr: The xarray DataArray that holds the data
    :type comp_var_xr: xr.DataArray

    :param component: The component name corresponding to the variable
    :type component: string

    :param variable: The variable name
    :type variable: string

    :return: component_dict
    """
    # ip-dependent: one Series per ip stored along the "ip" dimension
    if "ip" in comp_var_xr.dims:
        class_name = component.split("; ")[0]
        comp_name = component.split("; ")[1]
        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]
        for ip_str in comp_var_xr.coords["ip"].values:
            da_ip = comp_var_xr.sel(ip=ip_str).drop_vars("ip")
            series = da_ip.to_dataframe().stack(level=0)
            series.index = series.index.droplevel(level=2).map("_".join)
            series = series[series > 0]
            if not len(series.index) == 0:
                ip_key_list = [key_list[0], int(ip_str)] + key_list[1:]
                setInDict(
                    component_dict[class_name][comp_name],
                    ip_key_list,
                    series.sort_index(),
                )
        return component_dict

    # ip-independent: original logic unchanged
    if drop_component:
        series = comp_var_xr.drop("component").to_dataframe().stack(level=0)
    else:
        series = comp_var_xr.to_dataframe().stack(level=0)
    series.index = series.index.droplevel(level=2).map("_".join)

    # NOTE: In FINE, a check is made to make sure that locationalEligibility indices matches indices of other
    # attributes. Removing 0 values ensures the match. If all are 0s, empty series is fed in, leading to error.
    # Therefore, if series is empty, the variable is not added.
    series = series[series > 0]

    if not len(series.index) == 0:
        class_name = component.split("; ")[0]
        comp_name = component.split("; ")[1]

        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]

        setInDict(component_dict[class_name][comp_name], key_list, series.sort_index())

    return component_dict


def add1dVariableToDict(
    component_dict, comp_var_xr, component, variable, drop_component=True
):
    """Convert the 1d variable data to required format and add it to component_dict.

    :param component_dict: The dict to which the variable data needs to be added
    :type component_dict: dict

    :param comp_var_xr: The xarray DataArray that holds the data
    :type comp_var_xr: xr.DataArray

    :param component: The component name corresponding to the variable
    :type component: string

    :param variable: The variable name
    :type variable: string

    :return: component_dict
    """
    # ip-dependent: one Series per ip stored along the "ip" dimension
    if "ip" in comp_var_xr.dims:
        class_name = component.split("; ")[0]
        comp_name = component.split("; ")[1]
        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]
        for ip_str in comp_var_xr.coords["ip"].values:
            series = comp_var_xr.sel(ip=ip_str).to_series()
            ip_key_list = [key_list[0], int(ip_str)] + key_list[1:]
            setInDict(
                component_dict[class_name][comp_name], ip_key_list, series.sort_index()
            )
        return component_dict

    # ip-independent: original logic unchanged
    if len(comp_var_xr.dims) == 0:
        # We check for the dimensionality again because single node models will have scalars here.
        series = pd.Series([comp_var_xr.item()], index=[comp_var_xr.space.item()])
    elif drop_component:
        series = comp_var_xr.drop("component").to_dataframe().unstack(level=0)
        series.index = series.index.droplevel(level=0)
    else:
        series = comp_var_xr.to_dataframe().unstack(level=0)
        series.index = series.index.droplevel(level=0)

    class_name = component.split("; ")[0]
    comp_name = component.split("; ")[1]

    key_list = getKeyHierarchyOfNestedDict(variable)
    key_list[0] = key_list[0][3:]

    setInDict(component_dict[class_name][comp_name], key_list, series.sort_index())

    return component_dict


def add0dVariableToDict(component_dict, comp_var_xr, component, variable):
    """Convert the dimensionless variable data to required format and add it to component_dict.

    :param component_dict: The dict to which the variable data needs to be added
    :type component_dict: dict

    :param comp_var_xr: The xarray DataArray that holds the data
    :type comp_var_xr: xr.DataArray

    :param component: The component name corresponding to the variable
    :type component: string

    :param variable: The variable name
    :type variable: string

    :return: component_dict
    """
    # ip-dependent: one scalar per ip stored along the "ip" dimension
    if "ip" in comp_var_xr.dims:
        class_name = component.split("; ")[0]
        comp_name = component.split("; ")[1]
        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]
        for ip_str in comp_var_xr.coords["ip"].values:
            da_ip = comp_var_xr.sel(ip=ip_str)
            var_value = da_ip.values
            if var_value.dtype == "int8":
                var_value = var_value.astype("bool")
            ip_key_list = [key_list[0], int(ip_str)] + key_list[1:]
            setInDict(
                component_dict[class_name][comp_name], ip_key_list, var_value.item()
            )
        return component_dict

    # ip-independent: original logic unchanged
    var_value = comp_var_xr.values

    if (
        var_value.dtype == "int8"
    ):  # NOTE: when saving to netcdf, the bool values are changed to int8 sometimes
        var_value = var_value.astype("bool")

    if (
        not var_value == ""
    ):  # NOTE: when saving to netcdf, the nans in string arrays are converted
        # to empty string (''). These need to be skipped.

        class_name = component.split("; ")[0]
        comp_name = component.split("; ")[1]

        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]

        setInDict(component_dict[class_name][comp_name], key_list, var_value.item())

    return component_dict
