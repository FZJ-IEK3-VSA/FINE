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
        multi_index_dataframe = data.stack()
        if "Period" in multi_index_dataframe.index.names:
            multi_index_dataframe = multi_index_dataframe.droplevel(0)

        multi_index_dataframe.index.set_names("time", level=0, inplace=True)
        multi_index_dataframe.index.set_names("space", level=1, inplace=True)

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
            multi_index_dataframe = df.stack()
            multi_index_dataframe.index.set_names(["space", "space_2"], inplace=True)
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
        _comp_var_xr = comp_var_xr.copy()
        _comp_var_xr = _comp_var_xr.stack(
            space_tmp=("space", "space_2")
        )

        # 2. Build merged labels from the MultiIndex
        space_idx = _comp_var_xr.coords["space_tmp"].to_index()

        merged_space = space_idx.map(lambda x: f"{x[0]}_{x[1]}")

        # 3. Assign merged labels back as a coordinate
        _comp_var_xr = _comp_var_xr.assign_coords(
            space_tmp=("space_tmp", merged_space)
        )

        # 4. Rename the stacked dimension
        _comp_var_xr = _comp_var_xr.rename(
            {"space_tmp": "space"}
        )

        # 5. Convert to pandas
        df = _comp_var_xr.to_dataframe().squeeze()

        # 6. Reshape
        df = df.dropna(how="all")
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


def createWasNoneMask(component_dict):
    """
    Creates a was_none_mask that tracks which parameters were originally None
    for each component. This allows perfect reconstruction of None values from xarray.
    Uses the comprehensive parameter_mask to ensure all parameters are considered.
    
    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
    
    :param parameter_mask: comprehensive parameter mask containing all parameters
    :type parameter_mask: dict
    
    :return: was_none_mask dictionary with structure {classname: {component: {parameter: bool}}}
    :rtype: dict
    """
    was_none_mask = {}
    
    for classname in component_dict:
        if classname not in was_none_mask:
            was_none_mask[classname] = {}
        for component in component_dict[classname]:
            if component not in was_none_mask[classname]:
                was_none_mask[classname][component] = {}
            for parameter_name, parameter_value in component_dict[classname][component].items():
                # Track if this parameter was originally None
                if parameter_value is None:
                    was_none_mask[classname][component][parameter_name] = True
                else:
                    was_none_mask[classname][component][parameter_name] = False
    
    return was_none_mask





def replaceNoneValuesForXarray(component_dict):
    """
    Replaces None values in component_dict with np.nan to ensure xarray compatibility.
    All None values are converted to np.nan regardless of their original type, and all 
    variables from the parameter_mask are ensured to exist in the component dict.
    
    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    
    :return: modified component_dict with None values replaced by np.nan and all parameters included
    :rtype: dict
    """
    import copy
    import numpy as np
    
    # Create a deep copy to avoid modifying the original
    modified_component_dict = copy.deepcopy(component_dict)
    
    # Process each component class
    for classname in component_dict:
        if classname not in modified_component_dict:
            modified_component_dict[classname] = {}
            
        for component in component_dict[classname]:
            if component not in modified_component_dict[classname]:
                modified_component_dict[classname][component] = {}
            
            for parameter_name in component_dict[classname][component].keys():
                # Convert None values to np.nan
                parameter_value = component_dict[classname][component][parameter_name]
                if parameter_value is None:
                    modified_component_dict[classname][component][parameter_name] = np.nan
    
    
    return modified_component_dict

def convertToXarray(component_dict):
    def _convert(data_dict):
        data_vars = {}
        # sort dict by keys to ensure consistent order
        data_dict = dict(sorted(data_dict.items()))

        for key, value in data_dict.items():
            if isinstance(value, pd.Series):
                # remove any named index
                #TODO: fix somewhere else!!!
                if "space" not in value.index.names:
                    value.index.names = ["space"] * value.index.nlevels
                # Handle pandas Series using xarray's built-in method
                data_vars[key] = xr.DataArray.from_series(value)
            elif isinstance(value, pd.DataFrame):
                # TODO: check why we have dataframes all of a sudden here!
                data_vars[key] = xr.DataArray.from_series(value.rename_axis("space",axis=1).rename_axis("time",axis=0).stack())
                
            elif isinstance(value, (str, int, float, bool)):
                # Handle all scalar values
                data_vars[key] = xr.DataArray(value)
            elif pd.isna(value):
                # Handle NaN values explicitly
                data_vars[key] = xr.DataArray(np.nan)
            else:
                print(f"Warning: Unexpected type for {key}: {type(value)}")
                # Try to convert anyway
                try:
                    data_vars[key] = xr.DataArray(value)
                except Exception as e:
                    print(f"Could not convert {key}: {e}")

        # Create and return the Dataset
        return xr.Dataset(data_vars)

    xr_dss = {}
    # convert dicts to xarray datasets
    for classname in component_dict:
        if classname not in xr_dss:
            xr_dss[classname] = {}
        for component in component_dict[classname]:
            if component not in xr_dss[classname]:
                xr_dss[classname][component] = _convert(component_dict[classname][component])
    return xr_dss
            
                
                    
    
    

def reconstructComponentDictFromXarray(xr_dss, was_none_mask):
    """
    Reconstructs the original component dictionary from xarray datasets,
    restoring None values based on the was_none_mask.
    
    :param xr_dss: A dict of xarray datasets containing the component data
    :type xr_dss: dict
    
    :param was_none_mask: dictionary tracking which parameters were originally None
    :type was_none_mask: dict
    
    :return: reconstructed component dictionary with original None values
    :rtype: dict
    """
    
    # First, extract the basic component structure from xarray
    # This would need to be implemented based on your specific xarray structure
    # For now, this is a placeholder showing the concept
    
    reconstructed_dict = {}
    
    for classname in xr_dss:
        reconstructed_dict[classname] = {}
        for component in xr_dss[classname]:
            reconstructed_dict[classname][component] = {}
            
            # Extract data from xarray datasets
            # This would involve converting xarray DataArrays back to pandas objects
            # and extracting scalar values
            
            # Apply the was_none_mask to restore None values
            if (classname in was_none_mask and 
                component in was_none_mask[classname]):
                
                for parameter, is_none in was_none_mask[classname][component].items():
                    if is_none:
                        # Restore the None value
                        reconstructed_dict[classname][component][parameter] = None
                    # else: keep the reconstructed value from xarray
    
    return reconstructed_dict


def extractWasNoneMaskFromXarray(xr_dss):
    """
    Extracts the was_none_mask from xarray datasets.
    
    :param xr_dss: A dict of xarray datasets containing the was_none_mask
    :type xr_dss: dict
    
    :return: was_none_mask dictionary
    :rtype: dict
    """
    was_none_mask = {}
    
    for classname in xr_dss:
        was_none_mask[classname] = {}
        for component in xr_dss[classname]:
            if 'was_none_mask' in xr_dss[classname][component]:
                # Convert xarray DataArray back to dictionary
                mask_da = xr_dss[classname][component]['was_none_mask']
                mask_dict = mask_da.to_series().to_dict()
                was_none_mask[classname][component] = mask_dict
            else:
                was_none_mask[classname][component] = {}
    
    return was_none_mask


def createDimensionMask(xr_dss):
    """
    Creates a dimension_mask that tracks the dimensionality of each parameter
    for each component using integer codes. This replaces the prefix system (0d_, 1d_, ts_, 2d_).
    
    Dimension mapping:
    - 0: Scalar (0D) - no dimensions, e.g., name, hasCapacityVariable
    - 1: Spatial (1D) - (space,) e.g., capacityMax per location
    - 2: Spatial Matrix (2D) - (space, space_2) e.g., transmission distances
    - 3: Time-Space Series (2D) - (time, space) e.g., operationRateMax time series
    - 4: Time-Space Matrix (3D) - (time, space, space_2) e.g., transmission time series
    - -1: Unknown/unsupported dimension combination
    
    :param xr_dss: A dict of xarray datasets containing the component data
    :type xr_dss: dict
    
    :return: dimension_mask dictionary with structure {classname: {component: {parameter: int}}}
    :rtype: dict
    """
    dimension_mask = {}
    
    for classname in xr_dss:
        dimension_mask[classname] = {}
        for component in xr_dss[classname]:
            dimension_mask[classname][component] = {}
            
            # Analyze each variable in the component's dataset
            for var_name, data_array in xr_dss[classname][component].data_vars.items():
                # Skip the was_none_mask itself
                if var_name == 'was_none_mask':
                    continue
                    
                # Extract the actual parameter name by removing the current prefix
                if var_name.startswith('0d_'):
                    param_name = var_name[3:]
                    dimension_type = 0  # scalar/constant
                elif var_name.startswith('1d_'):
                    param_name = var_name[3:]
                    dimension_type = 1  # 1-dimensional (space only)
                elif var_name.startswith('2d_'):
                    param_name = var_name[3:]
                    dimension_type = 2  # 2-dimensional (space x space_2)
                elif var_name.startswith('ts_'):
                    param_name = var_name[3:]
                    # Need to check actual dimensions to distinguish between type 3 and 4
                    dims = data_array.dims
                    if len(dims) == 2 and 'time' in dims and 'space' in dims:
                        dimension_type = 3  # time-space series
                    elif len(dims) == 3 and 'time' in dims and 'space' in dims and 'space_2' in dims:
                        dimension_type = 4  # time-space matrix (transmission time series)
                    else:
                        dimension_type = 3  # default to time-space series
                else:
                    # Fallback: infer from actual dimensions
                    param_name = var_name
                    dims = data_array.dims
                    if len(dims) == 0:
                        dimension_type = 0  # scalar
                    elif len(dims) == 1:
                        if 'space' in dims:
                            dimension_type = 1  # spatial
                    elif len(dims) == 2:
                        if 'space' in dims and 'space_2' in dims:
                            dimension_type = 2  # spatial matrix
                        elif 'time' in dims and 'space' in dims:
                            dimension_type = 3  # time-space series
                    elif len(dims) == 3:
                        if 'time' in dims and 'space' in dims and 'space_2' in dims:
                            dimension_type = 4  # time-space matrix
                    else:
                        dimension_type = -1  # unknown
                
                dimension_mask[classname][component][param_name] = dimension_type
    
    return dimension_mask


def addDimensionMaskToXarray(xr_dss, dimension_mask):
    """
    Adds the dimension_mask to each component's xarray dataset.
    
    :param xr_dss: A dict of xarray datasets to which the dimension_mask should be added
    :type xr_dss: dict
    
    :param dimension_mask: dictionary tracking parameter dimensions
    :type dimension_mask: dict
    
    :return: xr_dss with dimension_mask added
    :rtype: dict
    """
    for classname in xr_dss:
        if classname in dimension_mask:
            for component in xr_dss[classname]:
                if component in dimension_mask[classname]:
                    # Create dimension_mask for this component
                    component_dims = dimension_mask[classname][component]
                    if component_dims:  # Only add if there are parameters to track
                        # Convert to pandas Series and then to xarray DataArray
                        dims_series = pd.Series(component_dims, name='dimension_type')
                        dims_series.index.name = 'parameter'
                        dims_da = xr.DataArray.from_series(dims_series)
                        
                        # Add to the component's dataset
                        xr_dss[classname][component]['dimension_mask'] = dims_da
    
    return xr_dss



def extractDimensionMaskFromXarray(xr_dss):
    """
    Extracts the dimension_mask from xarray datasets.
    
    :param xr_dss: A dict of xarray datasets containing the dimension_mask
    :type xr_dss: dict
    
    :return: dimension_mask dictionary
    :rtype: dict
    """
    dimension_mask = {}
    
    for classname in xr_dss:
        dimension_mask[classname] = {}
        for component in xr_dss[classname]:
            if 'dimension_mask' in xr_dss[classname][component]:
                # Convert xarray DataArray back to dictionary
                dims_da = xr_dss[classname][component]['dimension_mask']
                dims_dict = dims_da.to_series().to_dict()
                dimension_mask[classname][component] = dims_dict
            else:
                dimension_mask[classname][component] = {}
    
    return dimension_mask


def getDimensionTypeMapping():
    """
    Returns the mapping between integer dimension types and their descriptions.
    
    :return: dictionary mapping integer dimension types to descriptions
    :rtype: dict
    """
    return {
        0: 'scalar',      # 0-dimensional (constant values)
        1: 'spatial',     # 1-dimensional (space only)  
        2: 'spatial_2d',  # 2-dimensional (space x space_2)
        3: 'temporal',    # time series (time x space or time only)
        -1: 'unknown'     # unknown dimension type
    }


def getDimensionTypeDescription(dimension_type):
    """
    Returns a human-readable description for the integer dimension type.
    
    :param dimension_type: Integer dimension type code
    :type dimension_type: int
    
    :return: Human-readable description
    :rtype: str
    """
    dimension_descriptions = {
        0: "Scalar (0D)",
        1: "Spatial (1D)",
        2: "Spatial Matrix (2D)",
        3: "Time-Space Series (2D)",
        4: "Time-Space Matrix (3D)",
        -1: "Unknown/Unsupported"
    }
    return dimension_descriptions.get(dimension_type, "Invalid dimension type")


def getDimensionTypeFromOldPrefix(old_var_name):
    """
    Maps old prefix-based variable names to new integer dimension types.
    
    :param old_var_name: Variable name with old prefix (e.g., '0d_name', 'ts_operationRateMax')
    :type old_var_name: str
    
    :return: Integer dimension type
    :rtype: int
    """
    if old_var_name.startswith('0d_'):
        return 0
    elif old_var_name.startswith('1d_'):
        return 1
    elif old_var_name.startswith('2d_'):
        return 2
    elif old_var_name.startswith('ts_'):
        return 3  # Note: 3D transmission time series (type 4) needs dimension analysis
    else:
        return -1


def createParameterDimensionDict(component_dict):
    """
    Creates a comprehensive parameter_mask that tracks ALL parameters and their dimensions
    for each component. This ensures that all variables in the component dict are included
    as parameters in the xarray output, regardless of whether they are None or not.
    
    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict
    
    :return: parameter_mask dictionary with structure {classname: {component: {parameter: dimension_type}}}
    :rtype: dict
    """
    import copy
    parameter_mask = {}
    
    for classname in copy.deepcopy(component_dict):
        parameter_mask[classname] = {}
        for component in component_dict[classname]:
            parameter_mask[classname][component] = {}
            
            # Analyze each parameter in the component
            for parameter_name, parameter_value in component_dict[classname][component].items():
                # Determine dimension type based on the parameter value
                dimension_type = _inferParameterDimensionType(parameter_value, classname)
                parameter_mask[classname][component][parameter_name] = dimension_type
    
    return parameter_mask

def processComponentDict(component_dict, locations, _mapC_dict):
    # add variable dimension names to pandas dataframes and series in component_dict
    # for constants do not do anything
    import copy
    component_dict_mod = copy.deepcopy(component_dict)
    for classname in component_dict:
        for component in component_dict[classname]:
            for parameter_name, parameter_value in component_dict[classname][component].items():
                if parameter_value is None or np.isscalar(parameter_value) or isinstance(parameter_value, (str, bool)) or isinstance(parameter_value, (int, float)):
                    # For scalar values, do not add dimension names
                    component_dict_mod[classname][component][parameter_name] = parameter_value
                elif isinstance(parameter_value, list):
                    # For lists, convert to string with ":" as separator
                    component_dict_mod[classname][component][parameter_name] = ":".join(map(str, parameter_value))
                elif isinstance(parameter_value, dict):
                        key_lists = getListsOfKeyPathsInNestedDict(
                            {parameter_name:parameter_value},
                            variable_name=parameter_name,
                        )

                        # iterate over all key-"paths" in nested dict
                        for key_list in key_lists:
                            _variable_description = ".".join(map(str, key_list))
                            
                            component_dict_mod[classname][component][_variable_description] = getFromDict(component_dict[classname][component], key_list)
                        # remove original key from dict
                        component_dict_mod[classname][component].pop(parameter_name, None)
                        
                    
                elif isinstance(parameter_value, pd.DataFrame):
                    
                    if classname in ["Transmission", "LinearOptimalPowerFlow"]:
                        mi_df = parameter_value.stack()
                        if set(parameter_value.index.to_list()).issubset(set(locations)):            
                            mi_df.index.set_names("space", level=0, inplace=True)
                            mi_df.index.set_names("space_2", level=1, inplace=True)
                        else:
                            # split X_X into multiindex
                            mi_df.index.set_names("time", level=0, inplace=True)
                            mi_df.index.set_names("space", level=1, inplace=True)
                            # use _mapC to split via location names 
                            space_index = mi_df.index.get_level_values("space")
                            time_index = mi_df.index.get_level_values("time")
                            # reconstruct multiindex
                            space_index_split = []
                            for idx in space_index:
                                loc1, loc2 = _mapC_dict[component][idx]
                                space_index_split.append((loc1, loc2))
                            mi_df.index = pd.MultiIndex.from_tuples(
                                [(time_index[i], space_index_split[i][0], space_index_split[i][1]) for i in range(len(space_index_split))],
                                names=["time", "space", "space_2"]
                            )
                    else:
                        mi_df = parameter_value.stack()
                        if "Period" in mi_df.index.names:
                            mi_df.index.set_names("time", level=1, inplace=True)
                            mi_df.index.set_names("space", level=2, inplace=True)
                        else:
                            mi_df.index.set_names("time", level=0, inplace=True)
                            mi_df.index.set_names("space", level=1, inplace=True)
                    
                    component_dict_mod[classname][component][parameter_name] = mi_df
                    
                elif isinstance(parameter_value, pd.Series):
                    if classname in ["Transmission", "LinearOptimalPowerFlow"]:
                        df = transform1dSeriesto2dDataFrame(parameter_value, locations)
                        mi_df = df.stack()
                        mi_df.index.set_names(
                            ["space", "space_2"], inplace=True
                        )
                        component_dict_mod[classname][component][parameter_name] = mi_df
                    else:
                        if set(parameter_value.index.values).issubset(set(locations)):
                            component_dict_mod[classname][component][parameter_name] = parameter_value.rename_axis("space")
                        else: # only time seires (unique to single node esM model)
                            component_dict_mod[classname][component][parameter_name] = parameter_value.rename_axis("time")
                    
                else:
                    raise TypeError(
                        f"Parameter '{parameter_name}' in component '{component}' has unsupported type: {type(parameter_value)}."
                    )
    return component_dict_mod


def _inferParameterDimensionType(parameter_value, classname):
    """
    Infers the dimension type of a parameter based on its value.
    
    Dimension mapping:
    - 0: Scalar (0D) - no dimensions, e.g., name, hasCapacityVariable, None values
    - 1: Temporal (1D) - pandas Series with time index (special for single node model)
    - 2: Spatial (1D) - pandas Series with spatial index
    - 3: Spatial Matrix (2D) - pandas DataFrame/Series with space x space_2 structure  
    - 4: Time-Space Series (2D) - pandas DataFrame with time x space structure
    - 5: Time-Space Matrix (3D) - pandas DataFrame with time x space x space_2 structure
    - -1: Unknown/unsupported dimension combination
    
    :param parameter_value: The value of the parameter to analyze
    :return: Integer dimension type code
    :rtype: int
    """
    import pandas as pd
    import numpy as np
    mapper = {
        frozenset({"time", "space", "space_2"}): 5,  # Time-Space Matrix (3D)
        frozenset({"time", "space"}): 4,            # Time-Space Series (2D)
        frozenset({"space", "space_2"}): 3,         # Spatial Matrix (2D)
        frozenset({"space"}): 2,                    # Spatial (1D)
        frozenset({"time"}): 1,                     # Temporal (1D)
    }

    def _get_category_from_index(index):
        """
        Helper function to get the category from index names.
        """
        if isinstance(index, pd.MultiIndex):
            dims = frozenset(index.names)
        else:
            dims = frozenset(index.names) if hasattr(index, 'names') else frozenset([index.name])
        
        return mapper.get(dims, -1)



    if parameter_value is None or np.isscalar(parameter_value) or isinstance(parameter_value, (str, bool)) or isinstance(parameter_value, (int, float)) or isinstance(parameter_value, dict):
        return 0
    elif isinstance(parameter_value, pd.DataFrame):
        return _get_category_from_index(parameter_value.index)
    elif isinstance(parameter_value, pd.Series):
        return _get_category_from_index(parameter_value.index)

    # Default for unknown types
    return -1


def addParameterDimensionsToXarray(xr_dss, dimension_mask, was_none_mask):
    """
    Adds comprehensive parameter information to each component's xarray dataset.
    This includes both the parameter names (as dimension) and their dimension types.
    
    :param xr_dss: A dict of xarray datasets to which the parameter info should be added
    :type xr_dss: dict
    
    :param parameter_mask: dictionary tracking all parameters and their dimensions
    :type parameter_mask: dict
    
    :return: xr_dss with comprehensive parameter information added
    :rtype: dict
    """
    import pandas as pd
    import xarray as xr
    import copy
    
    for classname in xr_dss:
        if classname in dimension_mask:
            for component in xr_dss[classname]:
                if component in dimension_mask[classname]:
                    # Get all parameters for this component
                    dim_params = copy.deepcopy(dimension_mask[classname][component])
                    none_params = copy.deepcopy(was_none_mask[classname][component])
                    
                    if dim_params:  # Only add if there are parameters
                        # Create parameter coordinate. sort alphabetically, ignoring upper/lower case
                        param_names = sorted(dim_params.keys())
                        
                        # Add parameter coordinate to the dataset if it doesn't exist
                        if 'parameter' not in xr_dss[classname][component].coords:
                            xr_dss[classname][component] = xr_dss[classname][component].assign_coords(
                                parameter=param_names
                            )                
                        
                        all_param_names = list(xr_dss[classname][component].coords['parameter'].values)
                        dimension_values = [dim_params.get(param, -1) for param in all_param_names]
                        none_values = [none_params.get(param, True) for param in all_param_names]  # Default True if missing
                        

                        dims_series = pd.Series(dimension_values, index=all_param_names, name='dimension_type')
                        dims_series.index.name = 'parameter'
                        dims_da = xr.DataArray.from_series(dims_series)
                        
                        mask_series = pd.Series(none_values, index=all_param_names, name='was_none')
                        mask_series.index.name = 'parameter'
                        mask_da = xr.DataArray.from_series(mask_series)
                        
                        # Add to the component's dataset
                        xr_dss[classname][component]['dimension_mask'] = dims_da
                        xr_dss[classname][component]['was_none_mask'] = mask_da

    
    return xr_dss

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
