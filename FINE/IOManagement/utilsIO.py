import math
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce  # forward compatibility for Python 3
import operator


def getFromDict(dataDict, mapList):
    """
    Get value from a dict by a list, which contains the dict keys.
    e.g. for dict={'a': {'b': 1}} with mapList ['a','b'] the function returns 1

    :param dataDict: nested dict, e.g. {'a': {'b'}
    :type dataDict: dict

    :param mapList: list with dictionary keys
    :type mapList: list
    """
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    """
    Set a value in a nested dict, where mapList contains the dict keys.
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
    """
    Get a list of dictionary keys for a nested dict from the variable description.
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

    return key_list


def getListsOfKeyPathsInNestedDict(data_dict, variable_name):
    """
    Get a list of all paths in a nested dict, starting after the variable_name,
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
                    key_lists_in_nested_dict.append([variable_name, key1, key2])
            else:
                key_lists_in_nested_dict.append([variable_name, key1])
        return key_lists_in_nested_dict
    else:
        return [[variable_name]]


def transform1dSeriesto2dDataFrame(series, locations):
    """
    Expands pandas Series into a pandas DataFrame.

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
        """
        Creation of subdictionaries on fly
        """
        self[key] = PowerDict(self, key)
        return self[key]

    def append(self, item):
        """
        Additional append function for lists in dict
        """
        self.parent[self.key] = [item]

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)
        if isinstance(val, PowerDict):
            val.parent = self
            val.key = key


def generateIterationDicts(component_dict, investmentPeriods):
    """Creates iteration dictionaries that contain descriptions of all
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
            for variable_description, data in component_dict[classname][
                component
            ].items():
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
                    else:
                        if _variable_description not in constants_iteration_dict.keys():
                            constants_iteration_dict[_variable_description] = [
                                description_tuple
                            ]
                        else:
                            constants_iteration_dict[_variable_description].append(
                                description_tuple
                            )

    return df_iteration_dict, series_iteration_dict, constants_iteration_dict


def addDFVariablesToXarray(xr_ds, component_dict, df_iteration_dict):
    """Adds all variables whose data is contained in a pd.DataFrame to xarray dataset.
    These variables are normally regional time series (dimensions - space, time)

    :param xr_ds: xarray dataset or a dict of xarray datasets to which the DF variables should be added
    :type xr_ds: xr.Dataset/dict

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param df_iteration_dict: dictionary with:
        keys - DF variable names
        values - list of tuple of component class and component name
    :type df_iteration_dict: dict

    :return: xr_ds
    """

    for variable_description, description_tuple_list in df_iteration_dict.items():
        df_dict = {}

        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}; {component}"

            # If a . is present in variable name, then the data would be
            # another level further in the component_dict
            if "." in variable_description:
                [var_name, subvar_name] = variable_description.split(".")
                if subvar_name.isdigit():
                    subvar_name = int(subvar_name)
                data = component_dict[classname][component][var_name][subvar_name]
            else:
                data = component_dict[classname][component][variable_description]

            multi_index_dataframe = data.stack()
            if "Period" in multi_index_dataframe.index.names:
                multi_index_dataframe.index.set_names("time", level=1, inplace=True)
                multi_index_dataframe.index.set_names("space", level=2, inplace=True)
            else:
                multi_index_dataframe.index.set_names("time", level=0, inplace=True)
                multi_index_dataframe.index.set_names("space", level=1, inplace=True)

            df_dict[df_description] = multi_index_dataframe

        df_variable = pd.concat(df_dict)
        df_variable.index.set_names("component", level=0, inplace=True)

        ds_component = xr.Dataset()
        ds_component[
            f"ts_{variable_description}"
        ] = df_variable.sort_index().to_xarray()

        for comp in df_variable.index.get_level_values(0).unique():
            this_class = comp.split("; ")[0]
            this_comp = comp.split("; ")[1]

            this_ds_component = (
                ds_component.sel(component=comp)
                .squeeze()
                .reset_coords(names=["component"], drop=True)
            )

            try:
                xr_ds[this_class][this_comp] = xr.merge(
                    [xr_ds[this_class][this_comp], this_ds_component]
                )
            except Exception:
                pass

    return xr_ds


def addSeriesVariablesToXarray(xr_ds, component_dict, series_iteration_dict, locations):
    """Adds all variables whose data is contained in a pd.Series to xarray dataset.
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

    for variable_description, description_tuple_list in series_iteration_dict.items():
        space_space_dict = {}
        space_dict = {}
        time_dict = {}

        for description_tuple in description_tuple_list:
            classname, component = description_tuple

            df_description = f"{classname}; {component}"

            # If a . is present in variable name, then the data would be
            # another level further in the component_dict
            key_list = getKeyHierarchyOfNestedDict(variable_description)

            # get the data in the dict with all keys within the key_list
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

            else:
                # If the data indices correspond to esM locations, then the
                # data is appended to space_dict, else time_dict
                if set(data.index.values).issubset(set(locations)):
                    space_dict[df_description] = data.rename_axis("space")
                else:
                    time_dict[df_description] = data.rename_axis("time")
                    time_dict[df_description] = pd.concat(
                        {locations[0]: time_dict[df_description]}, names=["space"]
                    )
                    time_dict[df_description] = time_dict[
                        df_description
                    ].reorder_levels(["time", "space"])

        # If the dicts are populated with at least one item,
        # process them further and merge with xr_ds
        if len(space_space_dict) > 0:
            df_variable = pd.concat(space_space_dict)
            df_variable.index.set_names("component", level=0, inplace=True)
            ds_component = xr.Dataset()
            ds_component[
                f"2d_{variable_description}"
            ] = df_variable.sort_index().to_xarray()

            for comp in df_variable.index.get_level_values(0).unique():
                this_class = comp.split("; ")[0]
                this_comp = comp.split("; ")[1]
                this_ds_component = (
                    ds_component.sel(component=comp)
                    .squeeze()
                    .reset_coords(names=["component"], drop=True)
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
            ds_component[
                f"1d_{variable_description}"
            ] = df_variable.sort_index().to_xarray()

            for comp in df_variable.index.get_level_values(0).unique():
                this_class = comp.split("; ")[0]
                this_comp = comp.split("; ")[1]
                this_ds_component = (
                    ds_component.sel(component=comp)
                    .squeeze()
                    .reset_coords(names=["component"], drop=True)
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
            ds_component[
                f"ts_{variable_description}"
            ] = df_variable.sort_index().to_xarray()

            for comp in df_variable.index.get_level_values(0).unique():
                this_class = comp.split("; ")[0]
                this_comp = comp.split("; ")[1]
                this_ds_component = (
                    ds_component.sel(component=comp)
                    .squeeze()
                    .reset_coords(names=["component"], drop=True)
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
    """Adds all variables whose data is just a constant value, to xarray dataset.

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

    for (
        variable_description,
        description_tuple_list,
    ) in constants_iteration_dict.items():
        df_dict = {}
        for description_tuple in description_tuple_list:
            classname, component = description_tuple
            df_description = f"{classname}; {component}"

            key_list = getKeyHierarchyOfNestedDict(variable_description)

            # get the data in the dict with all keys within the key_list
            data = component_dict[classname][component]
            for item in key_list:
                data = data[item]

            df_dict[df_description] = data

        df_variable = pd.Series(df_dict)
        df_variable.index.set_names("component", inplace=True)

        ds_component = xr.Dataset()
        ds_component[f"0d_{variable_description}"] = xr.DataArray.from_series(
            df_variable
        )

        for comp in df_variable.index.get_level_values(0).unique():
            this_class = comp.split("; ")[0]
            this_comp = comp.split("; ")[1]
            this_ds_component = (
                ds_component.sel(component=comp)
                .squeeze()
                .reset_coords(names=["component"], drop=True)
            )

            try:
                xr_ds[this_class][this_comp] = xr.merge(
                    [xr_ds[this_class][this_comp], this_ds_component]
                )
            except Exception:
                pass

    return xr_ds


def processXarrayAttributes(xarray_dataset):
    """Data types such as sets, dicts, bool, pandas df/series and Nonetype
    are not serializable. Therefore, they are converted to lists/strings while saving.
    They are converted back to right formats while setting up the esM instance.

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
    for attr_name, attr_value in _xarray_dataset.attrs.items():
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

    return xarray_dataset


def addTimeSeriesVariableToDict(
    component_dict, comp_var_xr, component, variable, drop_component=True
):
    """Converts the time series variable data to required format and adds it to
    component_dict

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

    if len(comp_var_xr.space.dims) == 0:
        df = comp_var_xr.to_series()
    elif drop_component:
        df = comp_var_xr.drop("component").to_dataframe().unstack(level=1)
    else:
        df = comp_var_xr.to_dataframe().unstack(level=1)

    if isinstance(df, pd.DataFrame):
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
    """Converts the 2d variable data to required format and adds it to
    component_dict

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
    """Converts the 1d variable data to required format and adds it to
    component_dict

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
    """Converts the dimensionless variable data to required format and adds it to
    component_dict

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
