import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce  # forward compatibility for Python 3
import operator
from fine.IOManagement.standardIO import getShadowPrices


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
        # Separate loc1_loc2
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


def _leafToIndexedData(data, classname, component, _mapC_dict, locations):
    """Convert a single component_dict leaf into a prefix and a pandas object
    (or constant) indexed with named dimensions, ready for ``.to_xarray()``.

    The prefix encodes the target xarray layout and mirrors the dispatch on the
    read side (see ``convertDatasetsToEnergySystemModel``):

        - "0d_": dimensionless constant
        - "1d_": one spatial dimension (space)
        - "2d_": two spatial dimensions (space, space_2) - Transmission
        - "ts_": regional time series (time, space[, space_2])

    :param data: the leaf value (pd.DataFrame, pd.Series or a constant)

    :param classname: component class name
    :type classname: str

    :param component: component name
    :type component: str

    :param _mapC_dict: mapping of Transmission component -> location tuple lookup
    :type _mapC_dict: dict

    :param locations: sorted esM locations
    :type locations: list

    :return: (prefix, indexed_data), where indexed_data is a pd.Series for the
        1d/2d/ts cases or the raw constant for the 0d case
    """
    isTransmission = classname in ["Transmission", "LinearOptimalPowerFlow"]

    # regional time series (time, space) - or (time, space, space_2) for Transmission
    if isinstance(data, pd.DataFrame):
        multi_index_dataframe = data.stack(future_stack=True).dropna()
        if "Period" in multi_index_dataframe.index.names:
            multi_index_dataframe = multi_index_dataframe.droplevel(0)

        multi_index_dataframe.index = multi_index_dataframe.index.set_names(
            "time", level=0
        )
        multi_index_dataframe.index = multi_index_dataframe.index.set_names(
            "space", level=1
        )

        if isTransmission:
            # use _mapC to split the packed loc1_loc2 index into two dimensions
            space_index = multi_index_dataframe.index.get_level_values("space")
            time_index = multi_index_dataframe.index.get_level_values("time")
            space_index_split = [_mapC_dict[component][idx] for idx in space_index]
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
        return "ts_", multi_index_dataframe

    if isinstance(data, pd.Series):
        if isTransmission:
            # 2d (space, space_2): series indices are packed like loc1_loc2
            df = transform1dSeriesto2dDataFrame(data, locations)
            multi_index_dataframe = df.stack(future_stack=True).dropna()
            multi_index_dataframe.index = multi_index_dataframe.index.set_names(
                ["space", "space_2"]
            )
            return "2d_", multi_index_dataframe

        if set(data.index.values).issubset(set(locations)):
            # 1d (space)
            return "1d_", data.rename_axis("space")

        # time series (time) - unique to single node esM models, expand to (time, space)
        series = data.rename_axis("time")
        series = pd.concat({locations[0]: series}, names=["space"])
        series = series.reorder_levels(["time", "space"])
        return "ts_", series

    # dimensionless constant
    return "0d_", data


def convertComponentDictToXarrayDict(component_dict, _mapC_dict, locations):
    """Convert component_dict into a nested dict of per-component xarray datasets.

    For every (class, component) the leaves of the nested component_dict are
    converted - each into a single named xarray variable - and merged into one
    xr.Dataset. This is the inverse of ``convertDatasetsToEnergySystemModel``.

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :param _mapC_dict: mapping of Transmission component -> location tuple lookup
    :type _mapC_dict: dict

    :param locations: sorted esM locations
    :type locations: list

    :return: {classname: {component: xr.Dataset}}
    :rtype: dict
    """
    xr_dss = {}
    for classname in component_dict:
        xr_dss[classname] = {}
        for component in component_dict[classname]:
            data_arrays = []
            for variable_description in component_dict[classname][component].keys():
                # a single variable may expand into several leaves (e.g. ip- or
                # commodity-dependent commodityConversionFactors)
                key_lists = getListsOfKeyPathsInNestedDict(
                    component_dict[classname][component],
                    variable_name=variable_description,
                )
                for key_list in key_lists:
                    variable_name = ".".join(map(str, key_list))
                    data = getFromDict(component_dict[classname][component], key_list)

                    prefix, indexed_data = _leafToIndexedData(
                        data, classname, component, _mapC_dict, locations
                    )

                    if isinstance(indexed_data, pd.Series):
                        data_array = indexed_data.sort_index().to_xarray()
                    else:
                        data_array = xr.DataArray(indexed_data)
                    # collapse length-1 dimensions (e.g. space for single-location
                    # components) into scalar coordinates, as the read side expects
                    data_array = data_array.squeeze()
                    data_array.name = f"{prefix}{variable_name}"
                    data_arrays.append(data_array)

            # outer-join all per-variable DataArrays into the component dataset
            xr_dss[classname][component] = xr.merge(data_arrays, compat="no_conflicts")

    return xr_dss


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
    if drop_component:
        series = (
            comp_var_xr.drop("component")
            .to_dataframe()
            .stack(level=0, future_stack=True)
        )
    else:
        series = comp_var_xr.to_dataframe().stack(level=0, future_stack=True)
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


def getShadowPriceXarray(esM, constraint_str="commodityBalanceConstraint"):
    """Retrieve shadow prices (dual values) for a specified constraint from the energy system model
    and return them as an xarray DataArray.

    The function handles fetching dual values for each investment period, processing them
    (including expanding time series if aggregated), and combining them into a single
    DataArray with dimensions for component, space, time, and investment period.

    :param esM: considered energy system model
    :type esM: EnergySystemModel class instance

    :param constraint_str: The name of the constraint in the Pyomo model (esM.pyM) to retrieve.
                           Defaults to "commodityBalanceConstraint".
    :type constraint_str: str, optional

    :return: An xarray DataArray containing the shadow prices, or None if retrieval fails.
             Typical dimensions: (component, space, time, ip).

    :rtype: xarray.DataArray or None
    :raises ValueError: If the constraint_str does not exist in the model.
    """

    def get_sp_xr(esM, ip=0, constraint_str="commodityBalanceConstraint"):
        # Helper function to process a single investment period

        # Verify constraint existence in the Pyomo model
        if not hasattr(esM.pyM, constraint_str):
            raise ValueError(f"Constraint '{constraint_str}' not found in model.")

        sp = getShadowPrices(
            esM,
            getattr(esM.pyM, constraint_str),
            ip=ip,
            dualValues=None,
            hasTimeSeries=True,
            periodOccurrences=esM.periodOccurrences,
            periodsOrder=esM.periodsOrder,
        )
        sp_xr = sp.to_xarray()

        # Rename dimensions from pandas default (level_0, ...) to meaningful names.
        # This mapping assumes the constraint index structure is (Component, Location, Time).
        rename_dict = {"level_0": "component", "level_1": "space", "level_2": "time"}
        # Only rename dimensions that actually exist in the result to avoid KeyErrors
        rename_dict = {k: v for k, v in rename_dict.items() if k in sp_xr.dims}

        sp_xr = sp_xr.rename(rename_dict)

        # Expand with investment period dimension. Here we use the investmentPeriodNames instead of the internal ip index.
        sp_xr = sp_xr.expand_dims(ip=[esM.investmentPeriodNames[ip]])
        return sp_xr

    sp_xr = None
    # Loop over investment periods to gather data for all periods
    for ip in range(len(esM.investmentPeriods)):
        sp_xr_ip = get_sp_xr(esM, ip=ip, constraint_str=constraint_str)

        if sp_xr_ip is not None:
            if sp_xr is None:
                # Initialize result with the first period found
                sp_xr = sp_xr_ip
            else:
                # Concatenate subsequent periods
                sp_xr = xr.concat([sp_xr, sp_xr_ip], dim="ip")
    # add constraint_str as attribute
    if sp_xr is not None:
        sp_xr.attrs["constraint"] = constraint_str

    return sp_xr
