import logging

import dask
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely.geometry import LineString
from shapely.ops import cascaded_union, unary_union

import metis_utils.io_tools as ito
import metis_utils.time_tools as tto
import spagat.dataset as spd

logger_representation = logging.getLogger('spagat_representation')



def add_region_centroids(sds, spatial_dim='space'):
    gpd_centroids = pd.Series([geom.centroid for geom in sds.xr_dataset.gpd_geometries.values])
    sds.xr_dataset['gpd_centroids'] = (spatial_dim, gpd_centroids.values)


def add_centroid_distances(sds, spatial_dim='space'):
    data_out_dummy = np.zeros((len(sds.xr_dataset[spatial_dim]), len(sds.xr_dataset[spatial_dim])))

    space = sds.xr_dataset[spatial_dim].values

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=[space, space], dims=[
        'space', 'space_2'])

    for region_id_1 in sds.xr_dataset[spatial_dim]:
        for region_id_2 in sds.xr_dataset[spatial_dim]:
            centroid_1 = sds.xr_dataset.sel(space=region_id_1).gpd_centroids.item(0)
            centroid_2 = sds.xr_dataset.sel(space=region_id_2).gpd_centroids.item(0)
            xr_data_array_out.loc[dict(space=region_id_1, space_2=region_id_2)
                                  ] = centroid_1.distance(centroid_2) / 1e3  # distances in km

    sds.xr_dataset['centroid_distances'] = (['space', 'space_2'], xr_data_array_out.values)


def aggregate_based_on_sub_to_sup_region_id_dict(sds, sub_to_sup_region_id_dict, aggregation_function_dict=None, 
                                                 spatial_dim='space', time_dim='TimeStep'):
    """Spatially aggregates all variables of the sds.xr_dataset according to sub_to_sup_region_id_dict using aggregation functions defined by aggregation_function_dict"""

    sds_2 = spd.SpagatDataSet()

    for varname, da in sds.xr_dataset.data_vars.items():
        if aggregation_function_dict is None:
            aggregation_mode = 'sum'
            aggregation_weight = None
        else:
            if varname in aggregation_function_dict.keys():
                aggregation_mode = aggregation_function_dict[varname][0] # TODO: implement this properly
                aggregation_weight_varname = aggregation_function_dict[varname][1]
                if aggregation_weight_varname is not None:
                    aggregation_weight = sds.xr_dataset[aggregation_weight_varname].fillna(1)
                else:
                    aggregation_weight = None
            else:
                aggregation_mode = 'sum'
                aggregation_weight = None

        if varname == 'gpd_geometries':
            shapes_aggregated = aggregate_geometries(sds.xr_dataset[varname], sub_to_sup_region_id_dict)
            sds_2.add_region_data(list(sub_to_sup_region_id_dict.keys()))
            sds_2.add_objects(description='gpd_geometries', dimension_list=(spatial_dim), object_list=shapes_aggregated)
            add_region_centroids(sds_2)

        elif varname != 'gpd_centroids':
            if spatial_dim in da.dims and time_dim in da.dims:
                da = aggregate_time_series(sds.xr_dataset[varname], sub_to_sup_region_id_dict, mode=aggregation_mode, 
                                           xr_weight_array=aggregation_weight)
                sds_2.xr_dataset[varname] = da

            if spatial_dim in da.dims and f'{spatial_dim}_2' not in da.dims:
                da = aggregate_values(sds.xr_dataset[varname], sub_to_sup_region_id_dict, mode=aggregation_mode)
                sds_2.xr_dataset[varname] = da

            if f'{spatial_dim}' in da.dims and f'{spatial_dim}_2' in da.dims:
                da = aggregate_connections(sds.xr_dataset[varname], sub_to_sup_region_id_dict, mode=aggregation_mode)

                sds_2.xr_dataset[varname] = da

    return sds_2


def aggregate_geometries(xr_data_array_in, sub_to_sup_region_id_dict):
    """Aggregates shapes given in a xr_data_array based on the dictionary"""
    space = list(sub_to_sup_region_id_dict.keys())

    # multipolygon_dimension = [0, 1, 2, 3]
    # TODO: maybe iteratively add increasing buffer size to avoid multipolygons

    shape_list = []
    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():

        temp_shape_list = list(xr_data_array_in.sel(space=sub_region_id_list).values)

        shape_union = cascaded_union(temp_shape_list)

        shape_list.append(shape_union)

    data = np.array(shape_list)

    # TODO: understand the multipolygon_dimension's origin: Why does shapely do these four polygons instead of one?
    # xr_data_array_out = xr.DataArray(data, coords=[space, multipolygon_dimension],
    #                                  dims=['space', 'multipolygon_dimension'])
    xr_data_array_out = xr.DataArray(data, coords=[space],
                                     dims=['space'])

    return xr_data_array_out


def aggregate_time_series(xr_data_array_in, sub_to_sup_region_id_dict, mode='mean', xr_weight_array=None, 
                          spatial_dim='space', time_dim='TimeStep'):
    """Aggregates all data of a data array containing time series with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: maybe add this to SpagatDataset as method?
    # TODO: generalize dims -> 'space' could be replaced by sth more general such as 'locs'

    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = {key: value.values for key, value in xr_data_array_in.coords.items()}

    aggregated_coords['space'] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.zeros(tuple(len(coord) for coord in aggregated_coords.values()))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list) 

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        if mode == 'mean':
            xr_data_array_out.loc[dict(space=sup_region_id)] = xr_data_array_in.sel(
                space=sub_region_id_list).mean(dim=spatial_dim).values

        if mode == 'weighted mean':
            weighted_xr_data_array_in = xr_data_array_in * xr_weight_array

            # TODO: implement weighted mean aggregation properly
            xr_data_array_out.loc[dict(space=sup_region_id)] = weighted_xr_data_array_in.sel(space=sub_region_id_list).sum(dim=spatial_dim, skipna=True) / xr_weight_array.sel(space=sub_region_id_list).sum(dim=spatial_dim, skipna=True)

        if mode == 'sum':
            xr_data_array_out.loc[dict(space=sup_region_id)] = xr_data_array_in.sel(
                space=sub_region_id_list).sum(dim=spatial_dim).values

    return xr_data_array_out


def aggregate_values(xr_data_array_in, sub_to_sup_region_id_dict, mode='mean', output_unit='GW'):
    """Aggregates all data of a data array containing time series with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: maybe add this to SpagatDataset as method?
    # TODO: add unit information to xr_data_array_out

    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = {key: value.values for key, value in xr_data_array_in.coords.items()}

    aggregated_coords['space'] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.zeros(tuple(len(coord) for coord in aggregated_coords.values()))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list) 

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        if mode == 'mean':
            xr_data_array_out.loc[dict(space=sup_region_id)] = xr_data_array_in.sel(
                space=sub_region_id_list).mean(dim='space').values
        elif mode == 'sum':
            xr_data_array_out.loc[dict(space=sup_region_id)] = xr_data_array_in.sel(
                space=sub_region_id_list).sum(dim='space').values
        elif mode == 'bool':
            xr_data_array_out.loc[dict(space=sup_region_id)] = xr_data_array_in.sel(
                space=sub_region_id_list).any(dim='space').values
        else:
            logger_representation.error('Please select one of the modes "mean", "bool" or "sum"')

    if output_unit == 'GW':
        return xr_data_array_out
    elif output_unit == 'KW':
        return xr_data_array_out


def aggregate_connections(xr_data_array_in, sub_to_sup_region_id_dict, mode='bool', set_diagonal_to_zero=True, spatial_dim='space'):
    """Aggregates all data of a data array containing connections with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: make sure that region and region_2 ids don't get confused
    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = {key: value.values for key, value in xr_data_array_in.coords.items()}

    aggregated_coords[f'{spatial_dim}'] = space_coords
    aggregated_coords[f'{spatial_dim}_2'] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.zeros(tuple(len(coord) for coord in aggregated_coords.values()))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list) 

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        for sup_region_id_2, sub_region_id_list_2 in sub_to_sup_region_id_dict.items():
            if mode == 'mean':
                xr_data_array_out.loc[dict(space=sup_region_id,
                                           space_2=sup_region_id_2)] = xr_data_array_in.sel(space=sub_region_id_list, 
                                                                                            space_2=sub_region_id_list_2).mean().values

            elif mode == 'bool':
                bool_array = xr_data_array_in.sel(
                    space=sub_region_id_list, space_2=sub_region_id_list_2).any()
                
                xr_data_array_out.loc[dict(space=sup_region_id,
                                           space_2=sup_region_id_2)] = bool_array

            elif mode == 'sum':
                xr_data_array_out.loc[dict(space=sup_region_id,
                                           space_2=sup_region_id_2)] = xr_data_array_in.sel(
                    space=sub_region_id_list, space_2=sub_region_id_list_2).sum()
            else:
                logger_representation.error('Please select one of the modes "mean", "bool" or "sum"')

            if set_diagonal_to_zero and sup_region_id == sup_region_id_2:
                xr_data_array_out.loc[dict(space=sup_region_id,
                                           space_2=sup_region_id_2)] = 0

    return xr_data_array_out


# spagat.output:
def create_grid_shapefile(sds, filename='AC_lines.shp', spatial_dim='space', locational_eligibility='2d_locationalEligibility'):
    # TODO: move this to spr or so
    # TODO: add check, whether gpd_centroids exist

    add_region_centroids(sds)

    buses_0 = []
    buses_1 = []
    geoms = []

    for region_id_1 in sds.xr_dataset[f'{spatial_dim}'].values:
        for region_id_2 in sds.xr_dataset[f'{spatial_dim}_2'].values:
            if sds.xr_dataset[locational_eligibility].sel(space=region_id_1, space_2=region_id_2).isel(component=0).values:
                buses_0.append(region_id_1)
                buses_1.append(region_id_2)

                point_1 = sds.xr_dataset.gpd_centroids.sel(space=region_id_1).item(0)
                point_2 = sds.xr_dataset.gpd_centroids.sel(space=region_id_2).item(0)
                line = LineString([(point_1.x, point_1.y), (point_2.x, point_2.y)])

                geoms.append(line)

    # TODO: understand what s_nom and x stand for (look into FINE?) and add or delete them below
    df = pd.DataFrame(
        {'bus0': buses_0,
            'bus1': buses_1,
            #      's_nom': ,
            #      'x': ,
         })

    ito.create_gdf(df, geoms, crs=3035, filepath=filename)