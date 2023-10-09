"""
Functions to aggregate region data for a reduced set 
of regions obtained as a result of spatial grouping of regions. 
"""

import logging
import warnings
from copy import deepcopy
import numpy as np
import xarray as xr
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
import pandas as pd

logger_representation = logging.getLogger("spatial_representation")


def aggregate_geometries(xr_data_array_in, sub_to_sup_region_id_dict):
    """
    For each region group, aggregates their geometries to form one super geometry.

    :param xr_data_array_in: subset of the xarray dataset data that corresponds to geometry variable
    :type xr_data_array_in: xr.DataArray

    :param sub_to_sup_region_id_dict: Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'01_reg_02_reg': ['01_reg','02_reg'],\n
            '03_reg_04_reg': ['03_reg','04_reg']}

    :type sub_to_sup_region_id_dict: Dict[str, List[str]]

    :returns: xr_data_array_out

        * Contains new geometries as values
        * Coordinates correspond to new regions

        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    :rtype: xr.DataArray
    """

    space = list(sub_to_sup_region_id_dict.keys())

    shape_list = []

    for sub_region_id_list in sub_to_sup_region_id_dict.values():
        temp_shape_list = list(xr_data_array_in.sel(space=sub_region_id_list).values)

        shape_union = cascaded_union(temp_shape_list)

        shape_list.append(shape_union)

    if len(shape_list) == 1:
        shape_list = (
            pd.Series(shape_list, index=space).to_xarray().rename({"index": "space"})
        )

    shape_list = np.array(shape_list, dtype=object)

    xr_data_array_out = xr.DataArray(shape_list, coords=[space], dims=["space"])

    return xr_data_array_out


def aggregate_time_series_spatially(
    xr_data_array_in,
    sub_to_sup_region_id_dict,
    mode="mean",
    xr_weight_array=None,
):
    """
    For each region group, aggregates the given time series variable.

    :param xr_data_array_in: subset of the xarray dataset data that corresponds to a time series variable
    :type xr_data_array_in: xr.DataArray

    :param sub_to_sup_region_id_dict: Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'01_reg_02_reg': ['01_reg','02_reg'],\n
            '03_reg_04_reg': ['03_reg','04_reg']}

    :type sub_to_sup_region_id_dict: Dict[str, List[str]]

    **Default arguments:**

    :param mode: Specifies how the time series should be aggregated
        |br| * the default value is 'mean'
    :type mode: str, one of {"mean", "weighted mean", "sum"}

    :param xr_weight_array: Required if `mode` is "weighted mean". `xr_weight_array` in this case would provide weights.
        The dimensions and coordinates of it should be same as `xr_data_array_in`
        |br| * the default value is None
    :type xr_weight_array: xr.DataArray

    :returns: xr_data_array_out

        * Contains aggregated time series as values
        * Coordinates correspond to new regions

        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    :rtype: xr.DataArray
    """
    space_coords = list(sub_to_sup_region_id_dict.keys())
    aggregated_coords = {
        key: value.values for key, value in xr_data_array_in.coords.items()
    }
    aggregated_coords["space"] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.empty(tuple(len(coord) for coord in aggregated_coords.values()))
    data_out_dummy[:] = np.nan

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list)

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        sub_region_da = xr_data_array_in.sel(space=sub_region_id_list)
        # drop regions that contains only NAs. These correspond to locationally ineligible regions
        sub_region_da = sub_region_da.dropna(dim="space", how="all")

        if mode == "weighted mean":
            # xr_data_array_in dytpe is set as float, this to avoid the division by zero error when dytpe=object
            sub_region_da = sub_region_da.astype(dtype=float)

            # get weights
            sub_region_weight_da = xr_weight_array.sel(space=sub_region_id_list)
            ## drop regions that contains only NAs. These correspond to locationally ineligible regions
            sub_region_weight_da = sub_region_weight_da.dropna(dim="space", how="all")

            weighted_sub_region_da = sub_region_da * sub_region_weight_da

            xr_data_array_out.loc[
                dict(space=sup_region_id)
            ] = weighted_sub_region_da.sum(dim="space") / sub_region_weight_da.sum(
                dim="space"
            )

        elif mode == "mean":
            xr_data_array_out.loc[dict(space=sup_region_id)] = sub_region_da.mean(
                dim="space"
            ).values

        elif mode == "sum":
            xr_data_array_out.loc[dict(space=sup_region_id)] = sub_region_da.sum(
                dim="space"
            ).values

        else:
            logger_representation.error(
                'Please select one of the modes "weighted mean", "mean", or "sum"'
            )

    # NOTE: If theres a 0 in weight (ex: capacity being 0),
    # it results in na during weighted mean calculation. These are converted to 0s
    if mode == "weighted mean":
        xr_data_array_out = xr_data_array_out.fillna(0)

    return xr_data_array_out


def aggregate_values_spatially(
    xr_data_array_in, sub_to_sup_region_id_dict, mode="mean"
):
    """
    For each region group, aggregates the given 1d variable.

    :param xr_data_array_in: subset of the xarray dataset data that corresponds to a 1d variable
    :type xr_data_array_in: xr.DataArray

    :param sub_to_sup_region_id_dict: Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'01_reg_02_reg': ['01_reg','02_reg'],\n
            '03_reg_04_reg': ['03_reg','04_reg']}

    :type sub_to_sup_region_id_dict: Dict[str, List[str]]

    **Default arguments:**

    :param mode: Specifies how the values should be aggregated
        |br| * the default value is 'mean'
    :type mode: str, one of {"mean", "sum", "bool"}

    :returns: xr_data_array_out

        * Contains aggregated 1d variable as values
        * Coordinates correspond to new regions

        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    :rtype: xr.DataArray
    """

    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = {
        key: value.values for key, value in xr_data_array_in.coords.items()
    }

    aggregated_coords["space"] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.zeros(tuple(len(coord) for coord in aggregated_coords.values()))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list)

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        sub_region_da = xr_data_array_in.sel(space=sub_region_id_list)

        if mode == "mean":
            xr_data_array_out.loc[dict(space=sup_region_id)] = sub_region_da.mean(
                dim="space"
            ).values
        else:
            _sum_xr = sub_region_da.sum(dim="space")

            if mode == "sum":
                xr_data_array_out.loc[dict(space=sup_region_id)] = _sum_xr

            elif mode == "bool":
                xr_data_array_out.loc[dict(space=sup_region_id)] = _sum_xr.where(
                    np.logical_or(_sum_xr.isnull(), _sum_xr == 0), 1
                )  # only replace positive non nas

            else:
                logger_representation.error(
                    'Please select one of the modes "mean", "bool" or "sum"'
                )

    xr_data_array_out = xr_data_array_out.fillna(0)
    return xr_data_array_out


def aggregate_connections(xr_data_array_in, sub_to_sup_region_id_dict, mode="bool"):
    """
    For each region group, aggregates the given 2d variable.

    :param xr_data_array_in: subset of the xarray dataset that corresponds to a 2d variable
    :type xr_data_array_in: xr.DataArray

    :param sub_to_sup_region_id_dict: Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'01_reg_02_reg': ['01_reg','02_reg'],\n
            '03_reg_04_reg': ['03_reg','04_reg']}

    :type sub_to_sup_region_id_dict: Dict[str, List[str]]

    **Default arguments:**

    :param mode: Specifies how the connections should be aggregated
        |br| * the default value is 'bool'
    :type mode: str, one of {"bool", "mean", "sum"}

    :returns: xr_data_array_out

        * Contains aggregated 2d variable as values
        * Coordinates correspond to new regions

        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    :rtype: xr.DataArray
    """

    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = {
        key: value.values for key, value in xr_data_array_in.coords.items()
    }

    aggregated_coords["space"] = space_coords
    aggregated_coords["space_2"] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.zeros(tuple(len(coord) for coord in aggregated_coords.values()))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list)

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        for sup_region_id_2, sub_region_id_list_2 in sub_to_sup_region_id_dict.items():
            sub_region_da = xr_data_array_in.sel(
                space=sub_region_id_list, space_2=sub_region_id_list_2
            )

            if mode == "mean":
                xr_data_array_out.loc[
                    dict(space=sup_region_id, space_2=sup_region_id_2)
                ] = sub_region_da.mean(dim=["space", "space_2"]).values

            else:
                _sum_xr = sub_region_da.sum(dim=["space", "space_2"])

                if mode == "sum":
                    xr_data_array_out.loc[
                        dict(space=sup_region_id, space_2=sup_region_id_2)
                    ] = _sum_xr

                elif mode == "bool":
                    xr_data_array_out.loc[
                        dict(space=sup_region_id, space_2=sup_region_id_2)
                    ] = _sum_xr.where(
                        np.logical_or(_sum_xr.isnull(), _sum_xr == 0), 1
                    )  # only replace positive non nas

                else:
                    logger_representation.error(
                        'Please select one of the modes "mean", "bool" or "sum"'
                    )

            # set diagonal values to 0
            if sup_region_id == sup_region_id_2:
                xr_data_array_out.loc[
                    dict(space=sup_region_id, space_2=sup_region_id_2)
                ] = 0

    xr_data_array_out = xr_data_array_out.fillna(0)
    return xr_data_array_out


def aggregate_esm_parameters_spatially(
    param_df_in, old_locations, sub_to_sup_region_id_dict, mode="mean"
):
    """
    For each region group, aggregates the given esm init parameter data.

    :param param_df_in: the dataframe with parameter data
    :type param_df_in: pd.DataFrame

    :param old_locations: list of former unaggregated regions
    :type old_locations: list

    :param sub_to_sup_region_id_dict: Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'01_reg_02_reg': ['01_reg','02_reg'],\n
            '03_reg_04_reg': ['03_reg','04_reg']}

    :type sub_to_sup_region_id_dict: Dict[str, List[str]]

    **Default arguments:**

    :param mode: Specifies how the data should be aggregated
        |br| * the default value is 'mean'
    :type mode: str, one of {"mean", "sum"}

    :returns: param_df_out
        * Contains aggregated data
    :rtype: pd.DataFrame
    """

    new_col_names = list(sub_to_sup_region_id_dict.keys())

    new_col_names.extend([x for x in param_df_in.columns if x not in old_locations])

    param_df_out = pd.DataFrame(data=0, index=param_df_in.index, columns=new_col_names)

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        if mode == "mean":
            param_df_out[sup_region_id] = param_df_in[sub_region_id_list].mean(axis=1)

        if mode == "sum":
            param_df_out[sup_region_id] = param_df_in[sub_region_id_list].sum(axis=1)
    return param_df_out


def aggregate_based_on_sub_to_sup_region_id_dict(
    xarray_datasets, sub_to_sup_region_id_dict, aggregation_function_dict
):
    """
    After spatial grouping, for each region group, spatially aggregates the data.

    :param xarray_datasets: The dictionary of xarray datasets holding esM's info
    :type xarray_datasets: Dict[str, xr.Dataset]

    :param sub_to_sup_region_id_dict: Dictionary new regions' ids and their corresponding group of regions

        * Ex.: {'01_reg_02_reg': ['01_reg','02_reg'],\n
            '03_reg_04_reg': ['03_reg','04_reg']}

    :type sub_to_sup_region_id_dict: Dict[str, List[str]]

    :param aggregation_function_dict: Contains information regarding the mode of aggregation for each individual variable, component, and component class combination.\n
        * Aggregation possibilities: mean, weighted mean, sum, bool(boolean OR).
        * Format of the dictionary:\n
             {<component_class>: {<component_name>: {<variable_name>: (<mode_of_aggregation>, <weights>),\n
                                                    <variable_name>: (<mode_of_aggregation>, None)}}}\n
          <weights> is required only if <mode_of_aggregation> is
          'weighted mean'. The name of the variable that should act as weights should be provided. Can be None otherwise.

    :type aggregation_function_dict: Dict[str, Tuple(str, None/str)]

    :returns: aggregated_xr_dataset

        * New xarray dataset with aggregated information
        * Coordinates correspond to new regions

        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    :rtype: xr.Dataset
    """

    # private function to get aggregation mode for a particular variable name
    def _get_aggregation_mode(varname, comp=None, comp_ds=None):
        # If aggregation_function_dict is passed AND the current variable is in it...
        if (aggregation_function_dict is not None) and (
            varname in aggregation_function_dict.keys()
        ):
            ## Get the mode and weight
            aggregation_mode = aggregation_function_dict[varname][0]
            aggregation_weight = aggregation_function_dict[varname][1]

            ## If the mode is "weighted mean"...
            if aggregation_mode == "weighted mean":
                ## raise error if weight is not specified
                if aggregation_weight is None:
                    raise TypeError(
                        "Weights must be passed in order to perform weighted mean"
                    )
                ## get corresponding weight data if another variable is supposed to be the weight
                elif isinstance(aggregation_weight, str):
                    if varname[:3] == "2d_":
                        try:
                            aggregation_weight = comp_ds[f"2d_{aggregation_weight}"]
                        except:
                            warnings.warn(
                                f"Aggregation mode for {comp} component's {varname[3:]} set to mean instead of \
                                weighted mean because corresponding weight: {aggregation_weight} variable is not found"
                            )

                            aggregation_mode = "mean"

                    else:
                        try:
                            aggregation_weight = comp_ds[f"1d_{aggregation_weight}"]
                        except:
                            warnings.warn(
                                f"Aggregation mode for {comp} component's {varname[3:]} set to mean instead of \
                                weighted mean because corresponding weight: {aggregation_weight} variable is not found"
                            )

                            aggregation_mode = "mean"

                else:
                    raise TypeError(
                        "Aggregation mode for {comp} component's {varname[3:]} is weighted mean, but the \
                        corresponding weight provided is not valid."
                    )

        # If aggregation_function_dict is not passed OR the current variable is not in it, set default
        else:
            aggregation_mode = "mean"
            aggregation_weight = None

        return aggregation_mode, aggregation_weight

    # Make a copy of xarray_dataset
    aggregated_xr_dataset = deepcopy(xarray_datasets)

    # update esM Parameters
    parameters_dict = aggregated_xr_dataset.get("Parameters").attrs

    for varname, vardata in parameters_dict.items():
        if varname == "locations":
            parameters_dict[varname] = set(sub_to_sup_region_id_dict.keys())

        elif isinstance(vardata, pd.DataFrame):
            old_locations = xarray_datasets.get("Parameters").attrs["locations"]
            if all([x in vardata.columns for x in old_locations]):
                aggregation_mode, aggregation_weight = _get_aggregation_mode(varname)

                aggregated_vardata = aggregate_esm_parameters_spatially(
                    vardata,
                    old_locations,
                    sub_to_sup_region_id_dict,
                    mode=aggregation_mode,
                )

                parameters_dict[varname] = aggregated_vardata

    # Aggregate geometries
    aggregated_xr_dataset["Geometry"] = aggregate_geometries(
        xarray_datasets.get("Geometry")["geometries"], sub_to_sup_region_id_dict
    )

    # Aggregate input data
    for comp_class, comp_dict in xarray_datasets.get("Input").items():
        for comp, comp_ds in comp_dict.items():
            aggregated_comp_ds = xr.Dataset()

            for varname, da in comp_ds.data_vars.items():
                # Check and set aggregation mode and weights
                aggregation_mode, aggregation_weight = _get_aggregation_mode(
                    varname, comp, comp_ds
                )

                # only aggregate data corresponding to regions that are locationally eligible
                var_dim = varname[:3]
                if var_dim != "0d_":
                    if var_dim == "2d_":
                        locational_eligibility = comp_ds["2d_locationalEligibility"]
                    else:
                        locational_eligibility = comp_ds["1d_locationalEligibility"]

                    da = da.where(locational_eligibility != 0)

                    if aggregation_weight is not None:
                        aggregation_weight = aggregation_weight.where(
                            locational_eligibility != 0
                        )

                # check if multiple investment periods exist
                if "Period" in da.coords:
                    if not da.coords["Period"].values == np.array(0):
                        raise NotImplementedError(
                            "Spatial aggregation currently does not support multiple investment periods."
                        )
                    else:
                        ## drop the period coordinate
                        da = da.reset_coords("Period", drop=True)

                ## Time series
                if var_dim == "ts_":
                    da = aggregate_time_series_spatially(
                        da,
                        sub_to_sup_region_id_dict,
                        mode=aggregation_mode,
                        xr_weight_array=aggregation_weight,
                    )

                ## 1d variables
                elif var_dim == "1d_":
                    da = aggregate_values_spatially(
                        da,
                        sub_to_sup_region_id_dict,
                        mode=aggregation_mode,
                    )

                ## 2d variables
                elif var_dim == "2d_":
                    da = aggregate_connections(
                        da,
                        sub_to_sup_region_id_dict,
                        mode=aggregation_mode,
                    )

                aggregated_comp_ds[varname] = da

            aggregated_xr_dataset["Input"][comp_class][comp] = aggregated_comp_ds

    return aggregated_xr_dataset
