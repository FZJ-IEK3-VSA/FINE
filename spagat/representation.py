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


def add_region_centroids(sds):
    gpd_centroids = pd.Series([geom.centroid for geom in sds.xr_dataset.gpd_geometries.values])
    sds.xr_dataset['gpd_centroids'] = ('regions', gpd_centroids.values)


def add_centroid_distances(sds):
    data_out_dummy = np.zeros((len(sds.xr_dataset.regions), len(sds.xr_dataset.regions)))

    regions = sds.xr_dataset.regions.values

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=[regions, regions], dims=[
        'regions', 'regions_2'])

    for region_id_1 in sds.xr_dataset.regions:
        for region_id_2 in sds.xr_dataset.regions:
            centroid_1 = sds.xr_dataset.sel(regions=region_id_1).gpd_centroids.item(0)
            centroid_2 = sds.xr_dataset.sel(regions=region_id_2).gpd_centroids.item(0)
            xr_data_array_out.loc[dict(regions=region_id_1, regions_2=region_id_2)
                                  ] = centroid_1.distance(centroid_2) / 1e3  # distances in km

    sds.xr_dataset['centroid_distances'] = (['regions', 'regions_2'], xr_data_array_out.values)


def aggregate_based_on_sub_to_sup_region_id_dict(sds, sub_to_sup_region_id_dict):
    # TODO: do this automatically, e.g. recognize cf time series and aggregate them accordingly ('mean' or 'weighted mean')

    e_load_aggregated = aggregate_time_series(sds.xr_dataset.e_load, sub_to_sup_region_id_dict, mode='sum')
    h_load_aggregated = aggregate_time_series(sds.xr_dataset['Hydrogen demand, operationRateFix'],
                                              sub_to_sup_region_id_dict, mode='sum')

    res_wind_cf_aggregated = aggregate_time_series(
        sds.xr_dataset['wind cf'], sub_to_sup_region_id_dict, mode='weighted mean',
        xr_weight_array=sds.xr_dataset['wind capacity'])
    res_solar_cf_aggregated = aggregate_time_series(
        sds.xr_dataset['solar cf'], sub_to_sup_region_id_dict, mode='weighted mean',
        xr_weight_array=sds.xr_dataset['solar capacity'])

    res_wind_capacity_aggregated = aggregate_values(
        sds.xr_dataset['wind capacity'], sub_to_sup_region_id_dict, mode='sum')
    res_solar_capacity_aggregated = aggregate_values(
        sds.xr_dataset['solar capacity'], sub_to_sup_region_id_dict, mode='sum')

    ac_grid_capacity_aggregated = aggregate_connections(sds.xr_dataset.AC_cable_capacity,
                                                        sub_to_sup_region_id_dict, mode='sum')
    ac_grid_incidence_aggregated = aggregate_connections(sds.xr_dataset.AC_cable_incidence,
                                                         sub_to_sup_region_id_dict, mode='bool')

    h_pipelines_distances_aggregated = aggregate_connections(sds.xr_dataset['Pipelines, distances'],
                                                             sub_to_sup_region_id_dict, mode='sum')
    h_pipelines_incidence_aggregated = aggregate_connections(sds.xr_dataset['Pipelines, eligibility'],
                                                             sub_to_sup_region_id_dict, mode='bool')

    shapes_aggregated = aggregate_geometries(sds.xr_dataset.gpd_geometries, sub_to_sup_region_id_dict)

    sds_2 = spd.SpagatDataSet()

    sds_2.add_region_data(list(sub_to_sup_region_id_dict.keys()))

    sds_2.xr_dataset['AC_cable_incidence'] = (('regions', 'regions_2'), ac_grid_incidence_aggregated)
    sds_2.xr_dataset['AC_cable_capacity'] = (('regions', 'regions_2'), ac_grid_capacity_aggregated)

    sds_2.xr_dataset['Pipelines, eligibility'] = (('regions', 'regions_2'), h_pipelines_incidence_aggregated)
    sds_2.xr_dataset['Pipelines, distances'] = (('regions', 'regions_2'), h_pipelines_distances_aggregated)

    sds_2.xr_dataset['AC_cable_incidence'] = (('regions', 'regions_2'), ac_grid_incidence_aggregated)
    sds_2.xr_dataset['AC_cable_incidence'] = (('regions', 'regions_2'), ac_grid_incidence_aggregated)

    sds_2.xr_dataset['AC_cable_incidence'] = (('regions', 'regions_2'), ac_grid_incidence_aggregated)
    sds_2.xr_dataset['AC_cable_incidence'] = (('regions', 'regions_2'), ac_grid_incidence_aggregated)

    sds_2.xr_dataset.coords['time'] = e_load_aggregated.time

    sds_2.xr_dataset['e_load'] = (('regions', 'time'), e_load_aggregated.T)
    sds_2.xr_dataset['Hydrogen demand, operationRateFix'] = (('regions', 'time'), h_load_aggregated.T)

    sds_2.xr_dataset['wind cf'] = (('regions', 'time'), res_wind_cf_aggregated.T)
    sds_2.xr_dataset['solar cf'] = (('regions', 'time'), res_solar_cf_aggregated.T)

    sds_2.xr_dataset['wind capacity'] = (('regions'), res_wind_capacity_aggregated)
    sds_2.xr_dataset['solar capacity'] = (('regions'), res_solar_capacity_aggregated)

    sds_2.add_objects(description='gpd_geometries', dimension_list=('regions'), object_list=shapes_aggregated)

    add_region_centroids(sds_2)

    return sds_2


def aggregate_geometries(xr_data_array_in, sub_to_sup_region_id_dict):
    """Aggregates shapes given in a xr_data_array based on the dictionary"""
    regions = list(sub_to_sup_region_id_dict.keys())

    # multipolygon_dimension = [0, 1, 2, 3]
    # TODO: maybe iteratively add increasing buffer size to avoid multipolygons

    shape_list = []
    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():

        temp_shape_list = list(xr_data_array_in.sel(regions=sub_region_id_list).values)

        shape_union = cascaded_union(temp_shape_list)

        shape_list.append(shape_union)

    data = np.array(shape_list)

    # TODO: understand the multipolygon_dimension's origin: Why does shapely do these four polygons instead of one?
    # xr_data_array_out = xr.DataArray(data, coords=[regions, multipolygon_dimension],
    #                                  dims=['regions', 'multipolygon_dimension'])
    xr_data_array_out = xr.DataArray(data, coords=[regions],
                                     dims=['regions'])

    return xr_data_array_out


def aggregate_time_series(xr_data_array_in, sub_to_sup_region_id_dict, mode='mean', xr_weight_array=None):
    """Aggregates all data of a data array containing time series with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: maybe add this to SpagatDataset as method?
    # TODO: generalize dims -> 'regions' could be replaced by sth more general such as 'locs'

    time = xr_data_array_in.time

    regions = list(sub_to_sup_region_id_dict.keys())

    data_out_dummy = np.zeros((len(regions), time.shape[0]))

    xr_data_array_out = xr.DataArray(data_out_dummy.T, coords=[time, regions], dims=['time', 'regions'])

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        if mode == 'mean':
            xr_data_array_out.loc[dict(regions=sup_region_id)] = xr_data_array_in.sel(
                region_ids=sub_region_id_list).mean(dim='region_ids').values

        if mode == 'weighted mean':
            weighted_xr_data_array_in = xr_data_array_in * xr_weight_array

            xr_data_array_out.loc[dict(regions=sup_region_id)] = weighted_xr_data_array_in.sel(
                region_ids=sub_region_id_list).sum(dim='region_ids').values / xr_weight_array.sel(
                region_ids=sub_region_id_list).sum(dim='region_ids').values

        if mode == 'sum':
            xr_data_array_out.loc[dict(regions=sup_region_id)] = xr_data_array_in.sel(
                region_ids=sub_region_id_list).sum(dim='region_ids').values

    return xr_data_array_out


def aggregate_values(xr_data_array_in, sub_to_sup_region_id_dict, mode='mean', output_unit='GW'):
    """Aggregates all data of a data array containing time series with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: maybe add this to SpagatDataset as method?
    # TODO: add unit information to xr_data_array_out

    regions = list(sub_to_sup_region_id_dict.keys())

    data_out_dummy = np.zeros((len(regions)))

    xr_data_array_out = xr.DataArray(data_out_dummy.T, coords=[regions], dims=['regions'])

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        if mode == 'mean':
            xr_data_array_out.loc[dict(regions=sup_region_id)] = xr_data_array_in.sel(
                regions=sub_region_id_list).mean(dim='regions').values
        if mode == 'sum':
            xr_data_array_out.loc[dict(regions=sup_region_id)] = xr_data_array_in.sel(
                regions=sub_region_id_list).sum(dim='regions').values

    if output_unit == 'GW':
        return xr_data_array_out
    elif output_unit == 'KW':
        return xr_data_array_out


def aggregate_connections(xr_data_array_in, sub_to_sup_region_id_dict, mode='bool', set_diagonal_to_zero=True):
    """Aggregates all data of a data array containing connections with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: make sure that region and region_2 ids don't get confused
    regions = list(sub_to_sup_region_id_dict.keys())

    data_out_dummy = np.zeros((len(regions), len(regions)))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=[regions, regions], dims=['regions',
                                                                                      'regions_2'])

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        for sup_region_id_2, sub_region_id_list_2 in sub_to_sup_region_id_dict.items():
            if mode == 'mean':
                xr_data_array_out.loc[dict(regions=sup_region_id,
                                           regions_2=sup_region_id_2)] = xr_data_array_in.sel(
                    regions=sub_region_id_list, regions_2=sub_region_id_list_2).mean().values

            elif mode == 'bool':
                xr_data_array_out.loc[dict(regions=sup_region_id,
                                           regions_2=sup_region_id_2)] = xr_data_array_in.sel(
                    regions=sub_region_id_list, regions_2=sub_region_id_list_2).any()

            elif mode == 'sum':
                xr_data_array_out.loc[dict(regions=sup_region_id,
                                           regions_2=sup_region_id_2)] = xr_data_array_in.sel(
                    regions=sub_region_id_list, regions_2=sub_region_id_list_2).sum()
            else:
                logger_representation.error('Please select one of the modes "mean", "bool" or "sum"')

            if set_diagonal_to_zero and sup_region_id == sup_region_id_2:
                xr_data_array_out.loc[dict(regions=sup_region_id,
                                           regions_2=sup_region_id_2)] = 0

    return xr_data_array_out


# spagat.output:
def create_grid_shapefile(sds, filename='AC_lines.shp'):
    # TODO: move this to spr or so
    # TODO: add check, whether gpd_centroids exist

    add_region_centroids(sds)

    buses_0 = []
    buses_1 = []
    geoms = []

    for region_id_1 in sds.xr_dataset.regions.values:
        for region_id_2 in sds.xr_dataset.regions_2.values:
            if sds.xr_dataset.AC_cable_incidence.sel(regions=region_id_1, regions_2=region_id_2).values:
                buses_0.append(region_id_1)
                buses_1.append(region_id_2)

                point_1 = sds.xr_dataset.gpd_centroids.sel(regions=region_id_1).item(0)
                point_2 = sds.xr_dataset.gpd_centroids.sel(regions=region_id_2).item(0)
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
