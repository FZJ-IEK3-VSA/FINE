import math
import numpy as np
import pandas as pd
import xarray as xr


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

    for row in series.iteritems():

        # n_seperators = row[0].count("_")

        # if (n_seperators % 2) == 0:
        #     raise ValueError(
        #         "Please rename your locations to contain same number of _s in each location name"
        #     )

        # else:
        #     # get the point of cut -> would be the middle _
        #     _cut = math.ceil(n_seperators / 2)

        #     split_id_list = row[0].split("_")
        #     id_1 = "_".join(split_id_list[:_cut])
        #     id_2 = "_".join(split_id_list[_cut:])

        #     df.loc[id_1, id_2] = row[1]

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


def generateIterationDicts(component_dict):
    """Creates iteration dictionaries that contain descriptions of all
    dataframes, series, and constants present in component_dict.

    :param component_dict: dictionary containing information about the esM instance's components
    :type component_dict: dict

    :return: df_iteration_dict, series_iteration_dict, constants_iteration_dict
    """

    df_iteration_dict, series_iteration_dict, constants_iteration_dict = {}, {}, {}

    # Loop through every class-component-variable combination
    for classname in component_dict:

        for component in component_dict[classname]:

            for variable_description, data in component_dict[classname][
                component
            ].items():
                description_tuple = (classname, component)

                # private function to check if the current variable is a dict, df, series or constant.
                # If its a dict (in the case of commodityConversionFactors), this is unpacked and the
                # the function is run on each value in dict
                def _append_to_iteration_dicts(_variable_description, _data):

                    if isinstance(_data, dict):
                        for key, value in _data.items():
                            nested_variable_description = f"{_variable_description}.{key}"  # NOTE: a . is introduced in the variable here

                            _append_to_iteration_dicts(
                                nested_variable_description, value
                            )

                    elif isinstance(_data, pd.DataFrame):
                        if _variable_description not in df_iteration_dict.keys():
                            df_iteration_dict[_variable_description] = [
                                description_tuple
                            ]
                        else:
                            df_iteration_dict[_variable_description].append(
                                description_tuple
                            )

                    # NOTE: transmission components are series in component_dict
                    # (example index - cluster_0_cluster_2)

                    elif isinstance(_data, pd.Series):
                        if _variable_description not in series_iteration_dict.keys():
                            series_iteration_dict[_variable_description] = [
                                description_tuple
                            ]
                        else:
                            series_iteration_dict[_variable_description].append(
                                description_tuple
                            )

                    else:
                        if _variable_description not in constants_iteration_dict.keys():
                            constants_iteration_dict[_variable_description] = [
                                description_tuple
                            ]
                        else:
                            constants_iteration_dict[_variable_description].append(
                                description_tuple
                            )

                _append_to_iteration_dicts(variable_description, data)

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
                data = component_dict[classname][component][var_name][subvar_name]
            else:
                data = component_dict[classname][component][variable_description]

            multi_index_dataframe = data.stack()
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
            if "." in variable_description:
                [var_name, subvar_name] = variable_description.split(".")
                data = component_dict[classname][component][var_name][subvar_name]
            else:
                data = component_dict[classname][component][variable_description]

            # Only ['Transmission', 'LinearOptimalPowerFlow'] are 2d classes.
            # So, if classname is one of these, append the data to space_space_dict
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
                if locations == sorted(data.index.values):
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


def addConstantsToXarray(xr_ds, component_dict, constants_iteration_dict):
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

            if "." in variable_description:
                [var_name, subvar_name] = variable_description.split(".")
                data = component_dict[classname][component][var_name][subvar_name]
            else:
                data = component_dict[classname][component][variable_description]

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

    if "." in variable:
        [var_name, nested_var_name] = variable.split(".")
        component_dict[class_name][comp_name][var_name[3:]][
            nested_var_name
        ] = df.sort_index()
        # NOTE: Thanks to PowerDict(), the nested dictionaries need not be created before adding the data.

    else:
        component_dict[class_name][comp_name][variable[3:]] = df.sort_index()

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

        if "." in variable:
            [var_name, nested_var_name] = variable.split(".")
            component_dict[class_name][comp_name][var_name[3:]][
                nested_var_name
            ] = series.sort_index()
        else:
            component_dict[class_name][comp_name][variable[3:]] = series.sort_index()

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

    if "." in variable:
        [var_name, nested_var_name] = variable.split(".")
        component_dict[class_name][comp_name][var_name[3:]][
            nested_var_name
        ] = series.sort_index()
    else:
        component_dict[class_name][comp_name][variable[3:]] = series.sort_index()

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

        if "." in variable:
            [var_name, nested_var_name] = variable.split(".")
            component_dict[class_name][comp_name][var_name[3:]][
                nested_var_name
            ] = var_value.item()
        else:
            component_dict[class_name][comp_name][variable[3:]] = var_value.item()

    return component_dict
