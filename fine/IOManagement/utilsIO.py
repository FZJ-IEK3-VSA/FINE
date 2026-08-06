import json
import logging
import operator
from functools import reduce  # forward compatibility for Python 3

import numpy as np
import pandas as pd
import xarray as xr

from fine.IOManagement.standardIO import getShadowPrices

logger = logging.getLogger(__name__)


# A netCDF attribute cannot hold a DataFrame, so an EnergySystemModel argument
# that is one is written as a string attribute per index entry plus the column
# names, the column dtypes and the index. On read it is rebuilt from exactly
# those parts. The names of the attributes written that way are listed in this
# attribute, so the read side does not have to know them in advance.
DATAFRAME_ATTRIBUTE_REGISTRY = "_dataframeAttributes"

# fallback registry for files written before DATAFRAME_ATTRIBUTE_REGISTRY existed
DATAFRAME_ESM_ATTRIBUTES = ("balanceLimit",)

# separator between the levels of a MultiIndex entry in an attribute name.
# componentLimitEligibility2dim is indexed by (locFrom, locTo) connections.
_INDEX_LEVEL_SEPARATOR = " -> "

# values a None or a NaN turns into once the row is stringified for netCDF
_MISSING_VALUE_STRINGS = ("None", "nan", "NaN", "NaT", "<NA>", "")


def _asList(value):
    """Read a netCDF attribute back as a list.

    netCDF collapses a one-element array attribute to a scalar on read, so a
    one-column DataFrame would come back as a bare string. Wrap such a value
    instead of letting ``list()`` split it into characters.

    :param value: attribute value as read from the file
    :type value: string, numpy array or list

    :return: the value as a list
    :rtype: list
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [] if value == "" else [value]
    return list(value)


def encodeDataFrameAttributeIndex(index):
    """Encode the index of a DataFrame esM attribute as a list of strings.

    :param index: index of the DataFrame
    :type index: pd.Index or pd.MultiIndex

    :return: one string per index entry, in the order of the index
    :rtype: list of strings
    """
    if isinstance(index, pd.MultiIndex):
        return [
            _INDEX_LEVEL_SEPARATOR.join(str(level) for level in entry)
            for entry in index
        ]
    return [str(entry) for entry in index]


def decodeDataFrameAttributeIndex(encodedIndex, nlevels):
    """Rebuild the index of a DataFrame esM attribute from its encoded form.

    :param encodedIndex: index entries as written by encodeDataFrameAttributeIndex
    :type encodedIndex: list of strings

    :param nlevels: number of index levels, 1 for a flat index
    :type nlevels: int

    :return: the rebuilt index
    :rtype: pd.Index or pd.MultiIndex
    """
    if nlevels > 1:
        return pd.MultiIndex.from_tuples(
            [tuple(entry.split(_INDEX_LEVEL_SEPARATOR)) for entry in encodedIndex]
        )
    return pd.Index(list(encodedIndex))


def addDataFrameAttributeToXarray(xarray_dataset, attr_name, dataframe):
    """Write one DataFrame esM attribute into the attributes of an xarray dataset.

    The DataFrame is split into one attribute per index entry, holding that row as
    an array of strings, plus the column names, the column dtypes and the encoded
    index. The original attribute is removed.

    :param xarray_dataset: dataset whose attributes are written to
    :type xarray_dataset: xr.Dataset

    :param attr_name: name of the esM attribute, e.g. "balanceLimit"
    :type attr_name: string

    :param dataframe: value of the attribute
    :type dataframe: pd.DataFrame
    """
    # keep the column order: it is written out explicitly, and a DataFrame that
    # comes back with its columns reordered does not compare equal to the original
    _df = dataframe
    encodedIndex = encodeDataFrameAttributeIndex(_df.index)
    for encodedEntry, (_, row) in zip(encodedIndex, _df.iterrows()):
        xarray_dataset.attrs[f"{attr_name}.{encodedEntry}"] = row.to_numpy().astype(str)
    xarray_dataset.attrs[f"{attr_name}_columns"] = _df.columns.tolist()
    xarray_dataset.attrs[f"{attr_name}_dtypes"] = _df.dtypes.astype(str).tolist()
    # an empty list is not a valid netCDF attribute, so an empty DataFrame is
    # written as an empty string and read back as an empty DataFrame
    xarray_dataset.attrs[f"{attr_name}_index"] = encodedIndex or ""
    xarray_dataset.attrs[f"{attr_name}_index_nlevels"] = _df.index.nlevels
    registry = list(xarray_dataset.attrs.get(DATAFRAME_ATTRIBUTE_REGISTRY, []))
    xarray_dataset.attrs[DATAFRAME_ATTRIBUTE_REGISTRY] = sorted(
        set(registry) | {attr_name}
    )
    del xarray_dataset.attrs[attr_name]


def extractDataFrameAttributesFromXarray(attrs):
    """Rebuild every DataFrame esM attribute from the attributes of an xarray dataset.

    This is the inverse of :func:`addDataFrameAttributeToXarray`. Which attributes
    to rebuild is read from the registry attribute the write side leaves behind, so
    that a new DataFrame argument on EnergySystemModel needs no change here.

    :param attrs: attributes of the read dataset
    :type attrs: dict

    :return: the rebuilt DataFrames per attribute name, and the attribute keys
        that were consumed and have to be dropped
    :rtype: tuple of a dict and a list of strings
    """
    dataframes = {}
    consumedKeys = [DATAFRAME_ATTRIBUTE_REGISTRY]
    if DATAFRAME_ATTRIBUTE_REGISTRY in attrs:
        attributeNames = _asList(attrs[DATAFRAME_ATTRIBUTE_REGISTRY])
    else:
        attributeNames = DATAFRAME_ESM_ATTRIBUTES
    for attr_name in attributeNames:
        columnsKey = f"{attr_name}_columns"
        if columnsKey not in attrs:
            continue

        columns = _asList(attrs[columnsKey])
        dtypes = _asList(attrs.get(f"{attr_name}_dtypes"))
        nlevels = int(attrs.get(f"{attr_name}_index_nlevels", 1))
        rowPrefix = f"{attr_name}."

        # older files carry no _index attribute; fall back to the row keys
        if f"{attr_name}_index" in attrs:
            encodedIndex = _asList(attrs[f"{attr_name}_index"])
        else:
            encodedIndex = sorted(
                key[len(rowPrefix) :] for key in attrs if key.startswith(rowPrefix)
            )

        consumedKeys.extend(
            [
                columnsKey,
                f"{attr_name}_dtypes",
                f"{attr_name}_index",
                f"{attr_name}_index_nlevels",
            ]
        )
        consumedKeys.extend(rowPrefix + entry for entry in encodedIndex)

        data = [_asList(attrs[rowPrefix + entry]) for entry in encodedIndex]
        df = pd.DataFrame(
            data=data,
            index=decodeDataFrameAttributeIndex(encodedIndex, nlevels),
            columns=columns,
        )
        for column, dtype in zip(df.columns, dtypes):
            if dtype == "bool":
                # astype(bool) on a string is True for every non-empty string,
                # so "False" would come back as True
                df[column] = df[column] == "True"
            elif df[column].dtype == object and dtype == "object":
                # a None or a NaN became a string on the way out; turn it back,
                # otherwise "None" would read as a set value
                df[column] = df[column].where(
                    ~df[column].isin(_MISSING_VALUE_STRINGS), None
                )
            else:
                df[column] = df[column].astype(dtype)
        dataframes[attr_name] = df
    return dataframes, consumedKeys


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


def serialiseDatasetAttributes(xarray_dataset):
    """Convert the attributes of a dataset into types a netCDF file can hold.

    Sets become sorted lists, dicts become flattened lists, pandas objects are
    split into one attribute per row, and booleans and None become strings. The
    dataset is changed in place. :func:`utilsIO.processXarrayAttributes` is the
    inverse.

    :param xarray_dataset: dataset whose attributes are converted
    :type xarray_dataset: xr.Dataset

    :return: the same dataset, for convenience
    :rtype: xr.Dataset
    """
    _xarray_dataset = (
        xarray_dataset.copy()
    )  # Copying to avoid errors due to change of size during iteration

    for attr_name, attr_value in _xarray_dataset.attrs.items():
        # if the attribute is set, convert into sorted list
        if isinstance(attr_value, set):
            xarray_dataset.attrs[attr_name] = sorted(xarray_dataset.attrs[attr_name])

        # if the attribute is dict, convert into a "flattened" list
        elif isinstance(attr_value, dict):
            xarray_dataset.attrs[attr_name] = list(
                f"{k} : {v}" for (k, v) in xarray_dataset.attrs[attr_name].items()
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
            addDataFrameAttributeToXarray(xarray_dataset, attr_name, attr_value)

        # if the attribute is bool, add a corresponding string
        elif isinstance(attr_value, bool):
            xarray_dataset.attrs[attr_name] = "True" if attr_value is True else "False"

        # if the attribute is None, add a corresponding string
        elif attr_value is None:
            xarray_dataset.attrs[attr_name] = "None"

    return xarray_dataset


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

    # STEP 1. Rebuild the DataFrame attributes, which were written as one string
    # attribute per index entry, and take their parts out of the way
    dataframe_attrs, dataframe_keys = extractDataFrameAttributesFromXarray(
        _xarray_dataset.attrs
    )
    keys_to_delete.extend(dataframe_keys)
    consumed_keys = set(dataframe_keys)

    # STEP 2. Loop through the remaining attributes and convert datatypes,
    # or append to dot_attrs_dict for conversion in a later step
    for attr_name, attr_value in _xarray_dataset.attrs.items():
        if attr_name in consumed_keys:
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

    # STEP 3. Reconstruct pandas series or df for each item in dot_attrs_dict
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
        xarray_dataset.attrs.pop(key, None)

    xarray_dataset.attrs.update(dataframe_attrs)

    return xarray_dataset


# --------------------------------------------------------------------------- #
# The stacked layout
#
# The netCDF layout holds one dataset per component and encodes the shape of a
# parameter in a prefix on the variable name ("0d_", "1d_", "2d_", "ts_"). The
# Zarr layout holds one dataset per model class, with the components
# concatenated along a "component" dimension, so a handful of large arrays take
# the place of thousands of small ones. A name has to mean the same thing for
# every component of the class, so the prefix cannot survive that. Two mask
# variables over (component, parameter) take its place:
#
#   variable_present  False means this component did not hold this parameter
#   variable_dims     the index names, comma joined and in order, "" for a scalar
#
# variable_dims holds the index names, that is the dimensions plus the scalar
# coordinates, not the dimensions alone. The netCDF builder calls squeeze(), so a
# component that uses a single location holds "space" as a scalar coordinate. The
# dimensions alone would report a scalar and rebuild the wrong name.
#
# :func:`stackComponents` and :func:`unstackComponents` are the inverse pair.
# Together they replace the was-none mask the format used to carry: a parameter
# whose value is None is left out, and variable_present marks it absent, so the
# reader leaves it at its default, None. The value is never used to decide this.
# A string parameter that is None for one component and set for another shares one
# dtype with it, and writing NaN into a string array yields the literal "nan",
# which reads back as an ID that was never set.
#
# A parameter whose value is a list, such as componentLimitID, is held as a JSON
# string in one cell rather than as an array. Its length is a property of the
# component, so two components of one class disagree on it, and an array cannot
# hold two entries in one row and three in the next. The index names in
# variable_dims are what says a cell holds a list: a name that is not one of
# INDEX_NAMES belongs to the component, not to the model.
#
# Known limitation: the masks restore a missing variable, they do not restore a
# missing coordinate. Concatenating along "component" widens every variable to
# the union of the components' coordinates, so a component that uses two of five
# locations comes back padded to five. The netCDF layout has the same limitation.
# --------------------------------------------------------------------------- #

# dimension the components of one model class are concatenated along
COMPONENT_DIMENSION = "component"

# dimension the two mask variables are indexed by
PARAMETER_DIMENSION = "parameter"

# the two mask variables themselves are not component parameters
PRESENCE_MASK = "variable_present"
DIMENSION_MASK = "variable_dims"
MASK_VARIABLES = (PRESENCE_MASK, DIMENSION_MASK)

# group attribute that says whether the variable names carry a netCDF prefix. It
# is true for Input and false for Results, so the reader needs no rule of its own
# about which group is which.
PREFIXED_ATTRIBUTE = "fine_variables_prefixed"

# group attribute holding the per component, per variable attributes as JSON. The
# unit of a result variable lives there, and it differs between the components of
# one class, so a concatenation would drop all but the first.
VARIABLE_ATTRIBUTES = "fine_variable_attributes"

# separator between the index names in the variable_dims mask
_DIMENSION_SEPARATOR = ","

# index names in the order the netCDF layout puts them in
INDEX_NAMES = ("time", "space", "space_2")

# the index names each netCDF prefix allows. squeeze() turns a length-1 dimension
# into a scalar coordinate, and the merge that follows attaches that coordinate to
# every variable of the component, including the ones it never belonged to. In a
# single location model every variable therefore carries a scalar "space". The
# prefix is what says which ones it belongs to.
_PREFIX_INDEX_NAMES = {
    "0d_": frozenset(),
    "1d_": frozenset({"space"}),
    "2d_": frozenset({"space", "space_2"}),
    "ts_": frozenset({"time", "space", "space_2"}),
}


def netcdfPrefixForDims(dims):
    """Get the netCDF variable name prefix for a set of index names.

    This is the one place the netCDF naming rule lives. It mirrors
    :func:`_leafToIndexedData`, which is what builds the names in the first
    place. A list valued parameter such as componentLimitID is written as a
    one-dimensional array over an unnamed dimension, so it falls to "0d_".

    :param dims: index names of the variable, that is its dimensions plus its
        scalar coordinates
    :type dims: iterable of strings

    :return: one of "0d_", "1d_", "2d_" and "ts_"
    :rtype: string
    """
    names = {str(name) for name in dims}
    if "time" in names:
        return "ts_"
    if {"space", "space_2"} <= names:
        return "2d_"
    if "space" in names:
        return "1d_"
    return "0d_"


def indexNamesOfVariable(data_array, netcdfPrefix=None):
    """Get the index names of one variable, dimensions and scalar coordinates.

    :param data_array: the variable, as it sits in the dataset of one component
    :type data_array: xr.DataArray

    :param netcdfPrefix: the prefix of the netCDF variable name, if the name
        carries one. It decides which scalar coordinates belong to the variable.
        |br| * the default value is None
    :type netcdfPrefix: None or string

    :return: the index names, in the order the netCDF layout puts them in
    :rtype: list of strings
    """
    names = [str(dim) for dim in data_array.dims]
    allowed = _PREFIX_INDEX_NAMES.get(netcdfPrefix)
    for name in INDEX_NAMES:
        if name in names or name not in data_array.coords:
            continue
        if data_array.coords[name].ndim != 0:
            continue
        if allowed is not None and name not in allowed:
            continue
        names = _insertIndexName(names, name)
    return names


def _insertIndexName(names, name):
    """Put a recovered index name where the netCDF layout would have put it."""
    position = INDEX_NAMES.index(name)
    for index, existing in enumerate(names):
        if existing not in INDEX_NAMES or INDEX_NAMES.index(existing) > position:
            return [*names[:index], name, *names[index:]]
    return [*names, name]


def _castObjectArray(data_array):
    """Cast one object-dtype array to a string or a numeric array.

    Zarr has no object dtype. A variable whose values are all strings becomes a
    unicode array, anything else becomes float64, and a value that cannot be read
    as a number becomes NaN.
    """
    values = data_array.values.flatten()
    present = [value for value in values if value is not None and pd.notna(value)]
    if present and all(isinstance(value, str) for value in present):
        return data_array.astype("U")
    if present and all(isinstance(value, (bool, np.bool_)) for value in present):
        # a boolean parameter has to stay boolean. hasIsBuiltBinaryVariable=0.0 is
        # rejected by the component constructor, which wants a bool.
        return data_array.astype(bool)
    try:
        return data_array.astype("float64")
    except (ValueError, TypeError):
        numeric = pd.to_numeric(data_array.values.ravel(), errors="coerce")
        return xr.DataArray(
            numeric.reshape(data_array.shape),
            dims=data_array.dims,
            coords=data_array.coords,
            name=data_array.name,
            attrs=data_array.attrs,
        )


def _normaliseDtypes(dataset):
    """Give every variable and coordinate of a dataset a dtype Zarr can store.

    :param dataset: dataset to normalise. It is copied, not changed in place.
    :type dataset: xr.Dataset

    :return: the normalised dataset
    :rtype: xr.Dataset
    """
    dataset = dataset.copy()

    for name, variable in dataset.data_vars.items():
        if variable.dtype == object:
            dataset[name] = _castObjectArray(variable)

    for name, coordinate in dataset.coords.items():
        if coordinate.dtype == object:
            dataset = dataset.assign_coords({name: _castObjectArray(coordinate)})

    return dataset


def _expandIndexNames(data_array, indexNames):
    """Undo squeeze(), so a per-component scalar coordinate never reaches the concat.

    A scalar coordinate cannot be concatenated with a dimension of the same name,
    and it is dropped by the concatenation. Turn it back into the length-1
    dimension it was before squeeze(), and put the dimensions into the order the
    netCDF layout uses, which is the order the netCDF reader unstacks them in.

    :param data_array: the variable, as it sits in the dataset of one component
    :type data_array: xr.DataArray

    :param indexNames: index names of the variable, see :func:`indexNamesOfVariable`
    :type indexNames: list of strings

    :return: the variable with one dimension per index name and no other coordinate
    :rtype: xr.DataArray
    """
    scalars = {
        name: data_array.coords[name].item()
        for name in indexNames
        if name not in data_array.dims
    }
    attrs = dict(data_array.attrs)
    data_array = data_array.reset_coords(drop=True)
    for name, value in scalars.items():
        data_array = data_array.expand_dims({name: [value]})
    if list(data_array.dims) != list(indexNames):
        data_array = data_array.transpose(*indexNames)
    data_array.attrs = attrs
    return data_array


def stackComponents(components, prefixed):
    """Concatenate the datasets of one model class into one stacked dataset.

    This is the inverse of :func:`unstackComponents`.

    :param components: the datasets of the class, {component name: xr.Dataset}
    :type components: dict

    :param prefixed: states if the variable names carry a netCDF prefix, which is
        the case for Input and not for Results
    :type prefixed: boolean

    :return: the stacked dataset, or None if there is nothing to concatenate
    :rtype: xr.Dataset or None
    """
    if not components:
        return None

    names = list(components)
    prepared = {}
    indexNames = {}
    attributes = {}

    for component in names:
        dataset = components[component]
        prepared[component] = {}
        indexNames[component] = {}
        attributes[component] = {}
        for name, variable in dataset.data_vars.items():
            prefix = str(name)[:3] if prefixed else None
            parameter = str(name)[3:] if prefixed else str(name)
            if prefixed and variable.dtype == object and _isAllMissing(variable):
                # the parameter is None. Leave it out, so variable_present marks
                # it absent and no other component's dtype can turn it into a
                # value. Writing NaN into a string array yields the string "nan".
                # Input only: an absent Input parameter keeps its default, but an
                # absent result variable changes the shape of the summary.
                continue
            wanted = indexNamesOfVariable(variable, prefix)
            indexNames[component][parameter] = wanted
            prepared[component][parameter] = (
                _encodeListVariable(variable)
                if _isListValued(wanted, prefixed)
                else _expandIndexNames(variable, wanted)
            )
            if variable.attrs:
                attributes[component][parameter] = dict(variable.attrs)

    # first appearance, not sorted. A parameter that holds a dict, such as
    # commodityConversionFactors, is written as one variable per key, and the
    # order of those variables is the order the keys come back in. Sorting them
    # would reorder the dict, which the netCDF layout does not do.
    parameters = list(
        dict.fromkeys(name for entry in indexNames.values() for name in entry)
    )

    datasets = _standardiseForConcat(prepared, names, parameters)

    if len(datasets) == 1:
        stacked = datasets[0].expand_dims({COMPONENT_DIMENSION: [names[0]]})
    else:
        stacked = xr.concat(
            datasets,
            dim=pd.Index(names, name=COMPONENT_DIMENSION),
            join="outer",
            coords="minimal",
            fill_value=np.nan,
        )
    stacked = _normaliseDtypes(stacked)

    maskDims = [COMPONENT_DIMENSION, PARAMETER_DIMENSION]
    maskCoords = {COMPONENT_DIMENSION: names, PARAMETER_DIMENSION: parameters}
    stacked[PRESENCE_MASK] = xr.DataArray(
        np.array(
            [
                [name in indexNames[component] for name in parameters]
                for component in names
            ],
            dtype=bool,
        ).reshape(len(names), len(parameters)),
        dims=maskDims,
        coords=maskCoords,
    )
    stacked[DIMENSION_MASK] = xr.DataArray(
        np.array(
            [
                [
                    _DIMENSION_SEPARATOR.join(indexNames[component].get(name, ()))
                    for name in parameters
                ]
                for component in names
            ],
            dtype="U",
        ).reshape(len(names), len(parameters)),
        dims=maskDims,
        coords=maskCoords,
    )
    stacked.attrs[PREFIXED_ATTRIBUTE] = bool(prefixed)
    stacked.attrs[VARIABLE_ATTRIBUTES] = json.dumps(attributes)
    return stacked


def _isAllMissing(variable):
    """Say if a variable holds no value at all, that is None or NaN everywhere.

    Only object arrays are asked. None reaches xarray as an object, and a numeric
    array is left alone so a large time series is not computed to answer this.

    :param variable: the variable to test
    :type variable: xr.DataArray

    :return: True if every entry is None or NaN
    :rtype: boolean
    """
    values = np.asarray(variable.values).ravel()
    if values.size == 0:
        return False
    return bool(pd.isna(values).all())


def _isListValued(indexNames, prefixed):
    """Say if the index names describe a list, not a position in the model.

    The Input layout knows three index names, the ones in INDEX_NAMES. Any other
    dimension there is anonymous, that is one xarray named when it built an array
    from a list, which is how a list valued parameter such as componentLimitID
    arrives. Every name has to be such a one, so a variable that also holds a
    position in the model keeps its array.

    Results are never treated this way. They carry index names of their own, such
    as locationIn and locationOut, which are positions in the model like any other.

    :param indexNames: index names of the variable, see :func:`indexNamesOfVariable`
    :type indexNames: iterable of strings

    :param prefixed: states if the variable names carry a netCDF prefix, which is
        the case for Input and not for Results
    :type prefixed: boolean

    :return: True if the variable holds a list
    :rtype: boolean
    """
    if not prefixed:
        return False
    names = [str(name) for name in indexNames]
    return bool(names) and all(name not in INDEX_NAMES for name in names)


def _encodeListVariable(variable):
    """Put the entries of a list valued variable into one cell, as JSON.

    An array cannot hold two entries in one row and three in the next, and the
    length of such a list is a property of the component, so the components of one
    class disagree on it. One cell per component always concatenates.

    JSON rather than a joined string: an ID is chosen by the user and may hold the
    separator, and JSON tells an empty list from a list holding an empty string.

    :param variable: the variable, whose dimensions are all its own
    :type variable: xr.DataArray

    :return: a scalar string variable holding the JSON
    :rtype: xr.DataArray
    """
    values = np.asarray(variable.values).ravel().tolist()
    return xr.DataArray(json.dumps([None if pd.isna(v) else v for v in values]))


def _decodeListVariable(cell, indexNames):
    """Rebuild a list valued variable from the one cell :func:`_encodeListVariable` wrote.

    :param cell: the scalar string variable holding the JSON
    :type cell: xr.DataArray

    :param indexNames: index names the variable had, one per dimension
    :type indexNames: list of strings

    :return: the variable, one dimensional again
    :rtype: xr.DataArray
    """
    values = json.loads(str(np.asarray(cell.values).item()))
    return xr.DataArray(np.array(values), dims=indexNames[:1] or ["dim_0"])


def _standardiseForConcat(prepared, names, parameters):
    """Give every component the same variables with the same dtypes.

    A variable that is a string for one component has to be a string for all of
    them, otherwise the concatenation produces an object array that Zarr cannot
    store. A variable that is boolean for every component stays boolean, because
    the component constructors reject a float where they want a bool. A component
    that does not hold a variable gets a scalar placeholder, which the presence
    mask marks as absent.

    :param prepared: {component: {parameter: xr.DataArray}}
    :type prepared: dict

    :param names: the component names, in the order they are concatenated
    :type names: list of strings

    :param parameters: every parameter name of the class, in the order they are written
    :type parameters: list of strings

    :return: one dataset per component, ready to concatenate
    :rtype: list of xr.Dataset
    """
    cast = {
        component: {
            parameter: (
                _castObjectArray(variable) if variable.dtype == object else variable
            )
            for parameter, variable in variables.items()
        }
        for component, variables in prepared.items()
    }

    dtypes = {}
    for parameter in parameters:
        found = [
            variables[parameter].dtype
            for variables in cast.values()
            if parameter in variables
        ]
        if any(np.issubdtype(dtype, np.str_) for dtype in found):
            dtypes[parameter] = "U"
        elif all(dtype is np.dtype(bool) for dtype in found):
            dtypes[parameter] = bool
        else:
            dtypes[parameter] = "float64"

    empty = {"U": "", bool: False, "float64": np.nan}
    datasets = []
    for component in names:
        variables = {}
        for parameter in parameters:
            variable = cast[component].get(parameter)
            if variable is None:
                variables[parameter] = xr.DataArray(empty[dtypes[parameter]])
            else:
                variables[parameter] = variable.astype(dtypes[parameter])
        datasets.append(xr.Dataset(variables))
    return datasets


def unstackComponents(dataset):
    """Split a stacked dataset back into one dataset per component.

    This is the inverse of :func:`stackComponents`. It restores the netCDF
    variable names when the group was written from prefixed names, so the result
    is exactly what
    :func:`~fine.IOManagement.xarrayIO.convertDatasetsToEnergySystemModel` reads.

    :param dataset: one stacked model class, as written by :func:`stackComponents`
    :type dataset: xr.Dataset

    :return: {component name: xr.Dataset}
    :rtype: dict
    """
    prefixed = _asBool(dataset.attrs.get(PREFIXED_ATTRIBUTE, False))
    attributes = json.loads(dataset.attrs.get(VARIABLE_ATTRIBUTES, "{}"))

    components = [str(name) for name in dataset[COMPONENT_DIMENSION].values]
    parameters = [str(name) for name in dataset[PARAMETER_DIMENSION].values]
    present = np.asarray(dataset[PRESENCE_MASK].values)
    recorded = np.asarray(dataset[DIMENSION_MASK].values)

    xr_dss = {}
    for index, component in enumerate(components):
        componentDataset = dataset.isel({COMPONENT_DIMENSION: index}, drop=True)
        componentAttributes = attributes.get(component, {})
        data_vars = {}
        for position, parameter in enumerate(parameters):
            if not bool(present[index, position]):
                continue
            if parameter not in componentDataset.data_vars:
                continue
            wanted = [
                name
                for name in str(recorded[index, position]).split(_DIMENSION_SEPARATOR)
                if name
            ]
            variable = componentDataset[parameter].reset_coords(drop=True)
            if _isListValued(wanted, prefixed):
                # the cell holds the whole list. It is not squeezed afterwards:
                # a list of one entry has to stay a list.
                variable = _decodeListVariable(variable, wanted)
                variable.attrs = dict(componentAttributes.get(parameter, {}))
                data_vars[
                    f"{netcdfPrefixForDims(wanted)}{parameter}"
                    if prefixed
                    else parameter
                ] = variable
                continue
            for dim in variable.dims:
                # a dimension the parameter never had. Take the first entry rather
                # than dropna: a genuine NaN is not padding, and dropna cannot tell
                # the two apart.
                if dim not in wanted:
                    variable = variable.isel({dim: 0}, drop=True)
            variable.attrs = dict(componentAttributes.get(parameter, {}))
            name = parameter
            if prefixed:
                # squeeze() again, because that is what the prefixed layout holds.
                # The Results layout is never squeezed, so it is left alone.
                variable = variable.squeeze()
                name = f"{netcdfPrefixForDims(wanted)}{parameter}"
            data_vars[name] = variable
        xr_dss[component] = xr.Dataset(data_vars)
    return xr_dss


def _asBool(value):
    """Read a group attribute back as a bool, whether it was stored as one or not."""
    if isinstance(value, str):
        return value == "True"
    return bool(value)


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

    # a dimensionless attribute whose value is a list, such as componentLimitID,
    # is stored as a one-dimensional array. It has no scalar to unwrap and no
    # empty-string case to skip.
    if var_value.ndim > 0:
        class_name, comp_name = component.split("; ")
        key_list = getKeyHierarchyOfNestedDict(variable)
        key_list[0] = key_list[0][3:]
        setInDict(component_dict[class_name][comp_name], key_list, var_value.tolist())
        return component_dict

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
