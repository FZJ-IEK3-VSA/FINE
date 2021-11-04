"""Representation algorithms to represent region data for a reduced set 
of regions obtained as a result of spatial grouping of regions. 
"""

import logging
import warnings
from copy import deepcopy

import numpy as np
import shapely
import xarray as xr
from shapely.ops import cascaded_union

import FINE.spagat.utils as spu

logger_representation = logging.getLogger("spatial_representation")


def aggregate_geometries(xr_data_array_in, sub_to_sup_region_id_dict):
    """For each region group, aggregates their geometries to form one super geometry.

    Parameters
    ----------
    xr_data_array_in :  xr.DataArray
        subset of the xarray dataset data that corresponds to geometry variable
    sub_to_sup_region_id_dict : Dict[str, List[str]]
        Dictionary new regions' ids and their corresponding group of regions
        Ex. {'01_reg_02_reg': ['01_reg','02_reg'],
             '03_reg_04_reg': ['03_reg','04_reg']}
    Returns
    -------
    xr_data_array_out :  xr.DataArray
        Contains new geometries as values
        Coordinates correspond to new regions
        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    """
    space = list(sub_to_sup_region_id_dict.keys())

    shape_list = []

    for sub_region_id_list in sub_to_sup_region_id_dict.values():

        temp_shape_list = list(xr_data_array_in.sel(space=sub_region_id_list).values)

        shape_union = cascaded_union(temp_shape_list)

        shape_list.append(shape_union)

    data = None
    if len(shape_list) == 1:
        data = np.array([shapely.geometry.asMultiPolygon(x) for x in shape_list])
    else:
        data = np.array(shape_list)

    xr_data_array_out = xr.DataArray(data, coords=[space], dims=["space"])

    return xr_data_array_out


def aggregate_time_series_spatially(
    xr_data_array_in, sub_to_sup_region_id_dict, mode="mean", xr_weight_array=None
):
    """For each region group, aggregates the given time series variable.

    Parameters
    ----------
    xr_data_array_in :  xr.DataArray
        subset of the xarray dataset data that corresponds to a time series variable
    sub_to_sup_region_id_dict :  Dict[str, List[str]]
        Dictionary new regions' ids and their corresponding group of regions
        Ex. {'01_reg_02_reg': ['01_reg','02_reg'],
             '03_reg_04_reg': ['03_reg','04_reg']}
    mode : {"mean", "weighted mean", "sum"}, optional
        Specifies how the time series should be aggregated
    xr_weight_array : xr.DataArray
        Required if `mode` is "weighted mean". `xr_weight_array` in this case would provide weights.
        The dimensions and coordinates of it should be same as `xr_data_array_in`

    Returns
    -------
    xr_data_array_out :  xr.DataArray
        Contains aggregated time series as values
        Coordinates correspond to new regions
        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
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
        if mode == "mean":
            xr_data_array_out.loc[dict(space=sup_region_id)] = (
                xr_data_array_in.sel(space=sub_region_id_list).mean(dim="space").values
            )

        if mode == "weighted mean":
            weighted_xr_data_array_in = xr_data_array_in * xr_weight_array

            xr_data_array_out.loc[
                dict(space=sup_region_id)
            ] = weighted_xr_data_array_in.sel(space=sub_region_id_list).sum(
                dim="space", skipna=False
            ) / xr_weight_array.sel(
                space=sub_region_id_list
            ).sum(
                dim="space", skipna=False
            )

        if mode == "sum":
            xr_data_array_out.loc[dict(space=sup_region_id)] = (
                xr_data_array_in.sel(space=sub_region_id_list)
                .sum(
                    dim="space", skipna=False
                )  # INFO: if skipna=Flase not specified, sum of nas results in 0.
                .values
            )

    # NOTE: If theres a 0 in weight (ex: capacity being 0),
    # it results in na during weighted mean calculation. These are converted to 0s
    if mode == "weighted mean":
        xr_data_array_out = xr_data_array_out.fillna(0)

    return xr_data_array_out


def aggregate_values_spatially(
    xr_data_array_in, sub_to_sup_region_id_dict, mode="mean"
):
    """For each region group, aggregates the given 1d variable.

    Parameters
    ----------
    xr_data_array_in : xr.DataArray
        subset of the xarray dataset data that corresponds to a 1d variable
    sub_to_sup_region_id_dict :  Dict[str, List[str]]
        Dictionary new regions' ids and their corresponding group of regions
        Ex. {'01_reg_02_reg': ['01_reg','02_reg'],
             '03_reg_04_reg': ['03_reg','04_reg']}
    mode : {"mean", "sum", "bool"}, optional
        Specifies how the values should be aggregated

    Returns
    -------
    xr_data_array_out :  xr.DataArray
        Contains aggregated 1d variable as values
        Coordinates correspond to new regions
        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
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
        if mode == "mean":
            xr_data_array_out.loc[dict(space=sup_region_id)] = (
                xr_data_array_in.sel(space=sub_region_id_list).mean(dim="space").values
            )
        else:
            _sum_xr = xr_data_array_in.sel(space=sub_region_id_list).sum(
                dim="space", skipna=False
            )

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

    return xr_data_array_out


def aggregate_connections(xr_data_array_in, sub_to_sup_region_id_dict, mode="bool"):
    """For each region group, aggregates the given 2d variable.

    Parameters
    ----------
    xr_data_array_in : xr.DataArray
        subset of the xarray dataset that corresponds to a 2d variable
    sub_to_sup_region_id_dict :  Dict[str, List[str]]
        Dictionary new regions' ids and their corresponding group of regions
        Ex. {'01_reg_02_reg': ['01_reg','02_reg'],
             '03_reg_04_reg': ['03_reg','04_reg']}
    mode : {"bool", "mean", "sum"}, optional
        Specifies how the connections should be aggregated

    Returns
    -------
    xr_data_array_out :  xr.DataArray
        Contains aggregated 2d variable as values
        Coordinates correspond to new regions
        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
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

            if mode == "mean":
                xr_data_array_out.loc[
                    dict(space=sup_region_id, space_2=sup_region_id_2)
                ] = (
                    xr_data_array_in.sel(
                        space=sub_region_id_list, space_2=sub_region_id_list_2
                    )
                    .mean(dim=["space", "space_2"])
                    .values
                )

            else:
                _sum_xr = xr_data_array_in.sel(
                    space=sub_region_id_list, space_2=sub_region_id_list_2
                ).sum(dim=["space", "space_2"], skipna=False)

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

    return xr_data_array_out


def aggregate_based_on_sub_to_sup_region_id_dict(
    xarray_datasets, sub_to_sup_region_id_dict, aggregation_function_dict=None
):
    """After spatial grouping, for each region group, spatially aggregates the data.

    Parameters
    ----------
    xarray_datasets : Dict[str, xr.Dataset]
        The dictionary of xarray datasets holding esM's info
    sub_to_sup_region_id_dict :  Dict[str, List[str]]
        Dictionary new regions' ids and their corresponding group of regions
        Ex. {'01_reg_02_reg': ['01_reg','02_reg'],
             '03_reg_04_reg': ['03_reg','04_reg']}
    aggregation_function_dict : Dict[str, Tuple(str, None/xr.DataArray)]
        - Contains information regarding the mode (sum, mean, bool, etc.) of aggregation for each individual variable.
        - Format of the dictionary - {<variable_name>: (<mode_of_aggregation>, <weights>),
                                      <variable_name>: (<mode_of_aggregation>, None)}
          <weights>, which is a xr.DataArray, is required only if <mode_of_aggregation> is
          'weighted mean'. Can be None otherwise.

    Returns
    -------
    aggregated_xr_dataset :  xr.Dataset
        New xarray dataset with aggregated information
        Coordinates correspond to new regions
        (In the above example, '01_reg_02_reg', '03_reg_04_reg' form new coordinates)
    """

    # xarray_dataset has prefix 1d_, 2d_ and ts_
    # Therefore, in order to match that, the prefix is added 
    # in aggregation_function_dict for each variable
    if aggregation_function_dict != None:
        aggregation_function_dict = {
            f"{dimension}_{key}": value
            for key, value in aggregation_function_dict.items()
            for dimension in ["ts", "1d", "2d"]
        }

    # private function to get aggregation mode for a particular variable name 
    def _get_aggregation_mode(varname, comp, comp_ds):
        # If aggregation_function_dict is passed AND the current variable is in it...
        if (aggregation_function_dict is not None) and (
            varname in aggregation_function_dict.keys()
        ):

            ## Get the mode and weight
            aggregation_mode = aggregation_function_dict[varname][0]
            aggregation_weight = aggregation_function_dict[varname][1]

            ## If variable is related to locationalEligibility, the mode must be "bool"
            if varname in ["1d_locationalEligibility", "2d_locationalEligibility"]:
                if aggregation_mode != "bool":
                    warnings.warn(
                        f"Aggregation mode for {comp} component's {varname} set to bool as only binary values are acceptable for this variable"
                    )
                    aggregation_mode = "bool"
                    aggregation_weight = None

            ## If the mode is "weighted mean"...
            if aggregation_mode == "weighted mean":
                ## raise error if weight is not specified
                if aggregation_weight is None:
                    raise TypeError(
                        "Weights must be passed in order to perform weighted mean"
                    )
                ## get corresponding weight data if another variable is supposed to be the weight
                ##INFO: User would only give the name of this weight variable. It should be
                #  matched based on the current variable's dimension
                elif isinstance(aggregation_weight, str):
                    if varname[:3] == "2d_":
                        try:
                            aggregation_weight = comp_ds[f"2d_{aggregation_weight}"]
                        except:
                            warnings.warn(
                                f"Aggregation mode for {comp} component's {varname[3:]} set to mean instead of \
                                weighted mean because corresponding weight: {aggregation_weight} variable is not found")

                            aggregation_mode = "mean"
                    
                    else:
                        try:
                            aggregation_weight = comp_ds[f"1d_{aggregation_weight}"]
                        except:
                            warnings.warn(
                                f"Aggregation mode for {comp} component's {varname[3:]} set to mean instead of \
                                weighted mean because corresponding weight: {aggregation_weight} variable is not found")

                            aggregation_mode = "mean"
                        

        # If aggregation_function_dict is not passed OR the current variable is not in it
        else:
            ## If variable is related to locationalEligibility, set the mode to "bool"
            if varname in ["1d_locationalEligibility", "2d_locationalEligibility"]:
                aggregation_mode = "bool"
                aggregation_weight = None

                warnings.warn(
                    f"Aggregation mode for {varname} set to bool as only binary \
                    values are acceptable for this variable"
                )

            ## For all other variables set default
            else:
                aggregation_mode = "sum"
                aggregation_weight = None

        return aggregation_mode, aggregation_weight

    # Make a copy of xarray_dataset
    aggregated_xr_dataset = deepcopy(xarray_datasets)

    # update locations in 'Parameters'
    aggregated_xr_dataset.get('Parameters').attrs['locations'] = set(sub_to_sup_region_id_dict.keys())

    # Aggregate geometries 
    aggregated_xr_dataset['Geometry'] = aggregate_geometries(
                                                xarray_datasets.get('Geometry')['geometries'], 
                                                sub_to_sup_region_id_dict
                                            )

    # Aggregate input data 
    for comp_class, comp_dict in xarray_datasets.get('Input').items():
        for comp, comp_ds in comp_dict.items():

            aggregated_comp_ds = xr.Dataset()

            for varname, da in comp_ds.data_vars.items():

                # Check and set aggregation mode and weights
                aggregation_mode, aggregation_weight = _get_aggregation_mode(varname, comp, comp_ds)
                
                ## Time series
                if varname[:3] == "ts_":
                    da = aggregate_time_series_spatially(da,
                                                        sub_to_sup_region_id_dict,
                                                        mode=aggregation_mode,
                                                        xr_weight_array=aggregation_weight,
                                                    )

                ## 1d variables
                elif varname[:3] == "1d_":
                    da = aggregate_values_spatially(da,
                                                    sub_to_sup_region_id_dict,
                                                    mode=aggregation_mode,
                                                )

                ## 2d variables
                elif varname[:3] == "2d_":
                    da = aggregate_connections(da,
                                            sub_to_sup_region_id_dict,
                                            mode=aggregation_mode,
                                            )

                ## aggregated or 0d variables
                aggregated_comp_ds[varname] = da
            
            aggregated_xr_dataset['Input'][comp_class][comp] = aggregated_comp_ds


    return aggregated_xr_dataset
