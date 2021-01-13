"""Representation algorithms to represent region data for a reduced set of regions.

"""

import logging
import warnings

import numpy as np
import pandas as pd
import copy
import shapely
import xarray as xr
from sklearn.cluster import AgglomerativeClustering
from shapely.geometry import LineString
from shapely.ops import cascaded_union

import FINE.spagat.utils as spu
import FINE.spagat.dataset as spd

logger_representation = logging.getLogger('spagat_representation')


def add_region_centroids(sds, spatial_dim="space"):
    gpd_centroids = pd.Series(
        [geom.centroid for geom in sds.xr_dataset.gpd_geometries.values]
    )
    sds.xr_dataset["gpd_centroids"] = (spatial_dim, gpd_centroids.values)


def add_centroid_distances(sds, spatial_dim="space"):
    data_out_dummy = np.zeros(
        (len(sds.xr_dataset[spatial_dim]), len(sds.xr_dataset[spatial_dim]))
    )

    space = sds.xr_dataset[spatial_dim].values

    xr_data_array_out = xr.DataArray(
        data_out_dummy, coords=[space, space], dims=["space", "space_2"]
    )

    for region_id_1 in sds.xr_dataset[spatial_dim]:
        for region_id_2 in sds.xr_dataset[spatial_dim]:
            centroid_1 = sds.xr_dataset.sel(space=region_id_1).gpd_centroids.item(0)
            centroid_2 = sds.xr_dataset.sel(space=region_id_2).gpd_centroids.item(0)
            xr_data_array_out.loc[dict(space=region_id_1, space_2=region_id_2)] = (
                centroid_1.distance(centroid_2) / 1e3
            )  # distances in km

    sds.xr_dataset["centroid_distances"] = (
        ["space", "space_2"],     #TODO: Use spatial_dim instead ??
        xr_data_array_out.values,
    )


def aggregate_geometries(xr_data_array_in, sub_to_sup_region_id_dict):
    """Aggregates shapes given in a xr_data_array based on the dictionary"""
    space = list(sub_to_sup_region_id_dict.keys())
    
    # multipolygon_dimension = [0, 1, 2, 3]
    # TODO: maybe iteratively add increasing buffer size to avoid multipolygons

    shape_list = []
    
    for sub_region_id_list in sub_to_sup_region_id_dict.values():
        
        temp_shape_list = list(xr_data_array_in.sel(space=sub_region_id_list).values)
        
        shape_union = cascaded_union(temp_shape_list)
        
        shape_list.append(shape_union)

    # TODO: why the convertion to numpy array results in a list of polygons when n_regions = 1 ???
    data = None
    if len(shape_list) == 1:
        data = np.array([shapely.geometry.asMultiPolygon(x) for x in shape_list])
    else:
        data = np.array(shape_list)
    
    # TODO: understand the multipolygon_dimension's origin: Why does shapely do these four polygons instead of one?
    # xr_data_array_out = xr.DataArray(data, coords=[space, multipolygon_dimension],
    #                                  dims=['space', 'multipolygon_dimension'])
    xr_data_array_out = xr.DataArray(data, coords=[space], dims=["space"])

    return xr_data_array_out

def aggregate_time_series(xr_data_array_in,
                        sub_to_sup_region_id_dict,
                        mode="mean",
                        xr_weight_array=None,
                        spatial_dim="space",
                        time_dim="TimeStep"):
    """Aggregates all data of a data array containing time series with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: maybe add this to SpagatDataset as method?
    # TODO: generalize dims -> 'space' could be replaced by sth more general such as 'locs'

    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = { key: value.values 
                            for key, value in xr_data_array_in.coords.items()
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
                xr_data_array_in.sel(space=sub_region_id_list)
                .mean(dim=spatial_dim)
                .values
            )

        if mode == "weighted mean":
            weighted_xr_data_array_in = xr_data_array_in * xr_weight_array

            # TODO: implement weighted mean aggregation properly
            xr_data_array_out.loc[
                dict(space=sup_region_id)
            ] = weighted_xr_data_array_in.sel(space=sub_region_id_list).sum(
                dim=spatial_dim, skipna=True
            ) / xr_weight_array.sel(
                space=sub_region_id_list
            ).sum(
                dim=spatial_dim, skipna=True
            )

        if mode == "sum":
            xr_data_array_out.loc[dict(space=sup_region_id)] = (
                xr_data_array_in.sel(space=sub_region_id_list)
                .sum(dim=spatial_dim)
                .values
            )

    return xr_data_array_out


def aggregate_values(xr_data_array_in, 
                    sub_to_sup_region_id_dict, 
                    mode="mean", 
                    output_unit="GW"):
    """Aggregates all data of a data array containing capacities corresponding 
    to time series with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: maybe add this to SpagatDataset as method?
    # TODO: add unit information to xr_data_array_out

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
        elif mode == "sum":
            xr_data_array_out.loc[dict(space=sup_region_id)] = (
                xr_data_array_in.sel(space=sub_region_id_list).sum(dim="space").values
            )
        elif mode == "bool":
            xr_data_array_out.loc[dict(space=sup_region_id)] = (
                xr_data_array_in.sel(space=sub_region_id_list).any(dim="space").values
            )
        else:
            logger_representation.error(
                'Please select one of the modes "mean", "bool" or "sum"'
            )

    if output_unit == "GW":
        return xr_data_array_out
    elif output_unit == "KW":
        return xr_data_array_out


def aggregate_connections(xr_data_array_in,
                        sub_to_sup_region_id_dict,
                        mode="bool",
                        set_diagonal_to_zero=True,
                        spatial_dim="space"):
    """Aggregates all data of a data array containing connections with dimension 'sub_regions' to new data_array with
    dimension 'regions"""
    # TODO: make sure that region and region_2 ids don't get confused
    space_coords = list(sub_to_sup_region_id_dict.keys())

    aggregated_coords = {
        key: value.values for key, value in xr_data_array_in.coords.items()
    }

    aggregated_coords[f"{spatial_dim}"] = space_coords
    aggregated_coords[f"{spatial_dim}_2"] = space_coords

    coord_list = [value for value in aggregated_coords.values()]
    dim_list = [key for key in aggregated_coords.keys()]

    data_out_dummy = np.zeros(tuple(len(coord) for coord in aggregated_coords.values()))

    xr_data_array_out = xr.DataArray(data_out_dummy, coords=coord_list, dims=dim_list)

    for sup_region_id, sub_region_id_list in sub_to_sup_region_id_dict.items():
        for sup_region_id_2, sub_region_id_list_2 in sub_to_sup_region_id_dict.items():  #TODO: aggregates both ways (ex. sum -> (a+b) + (b+a)), is this required ? maybe it is for birectional connections
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
            elif mode == "bool":
                sum_array = xr_data_array_in.sel(
                    space=sub_region_id_list, space_2=sub_region_id_list_2
                ).sum(dim=["space", "space_2"])

                xr_data_array_out.loc[
                    dict(space=sup_region_id, space_2=sup_region_id_2)
                ] = sum_array.where(sum_array == 0, 1)

            elif mode == "sum":
                xr_data_array_out.loc[
                    dict(space=sup_region_id, space_2=sup_region_id_2)
                ] = xr_data_array_in.sel(
                    space=sub_region_id_list, space_2=sub_region_id_list_2
                ).sum(
                    dim=["space", "space_2"]
                )
            else:
                logger_representation.error(
                    'Please select one of the modes "mean", "bool" or "sum"'
                )

            if set_diagonal_to_zero and sup_region_id == sup_region_id_2:
                xr_data_array_out.loc[
                    dict(space=sup_region_id, space_2=sup_region_id_2)
                ] = 0

                # TODO: make sure, that setting NAN values to 0 does not cause troubles
                # -> find a better, such that only non-nan diagonal entries are set to zero

    return xr_data_array_out



def aggregate_based_on_sub_to_sup_region_id_dict(sds,
                                                sub_to_sup_region_id_dict,
                                                aggregation_function_dict=None,
                                                spatial_dim="space",
                                                time_dim="TimeStep" ):

    """Spatially aggregates all variables of the sds.xr_dataset according to 
    sub_to_sup_region_id_dict using aggregation functions defined by 
    aggregation_function_dict"""

    aggregated_sds = spd.SpagatDataset()
    
    
    for varname, da in sds.xr_dataset.data_vars.items():

        #STEP 1. Set aggregation mode and weights
        # If aggregation_function_dict is passed AND the current variable is in it   
        if ((aggregation_function_dict is not None) and (varname in aggregation_function_dict.keys())):
            
            ## Get the mode and weight 
            aggregation_mode = aggregation_function_dict[varname][0]  
            aggregation_weight = aggregation_function_dict[varname][1]

            ## If variable is related to locationalEligibility, set the mode to "bool"
            if varname in ["1d_locationalEligibility", "2d_locationalEligibility"]:
                if aggregation_mode != "bool":
                    warnings.warn(f"Aggregation mode for {varname} set to bool as only binary values are acceptable for this variable")
                    aggregation_mode = "bool"
                    aggregation_weight = None 

            ## If the mode is "weighted mean", raise error if weight is not specified 
            if ((aggregation_mode == "weighted mean") and (aggregation_weight is None)):
                raise TypeError("Weights must be passed in order to perform weighted mean")
            
        # If aggregation_function_dict is not passed OR the current variable is not in it    
        else:
            ## If variable is related to locationalEligibility, set the mode to "bool"
            if varname in ["1d_locationalEligibility", "2d_locationalEligibility"]:
                aggregation_mode = "bool"
                aggregation_weight = None 

                print(f"Aggregation mode for {varname} set to bool as only binary values are acceptable for this variable")

            ## For all other variables set default
            else:
                aggregation_mode = "sum"
                aggregation_weight = None    

        #STEP 2. Aggregation 
        #STEP 2a. Aggregate geometries if varname == "gpd_geometries"    
        if varname == "gpd_geometries":
            shapes_aggregated = aggregate_geometries(sds.xr_dataset[varname], 
                                                    sub_to_sup_region_id_dict)

            aggregated_sds.add_region_data(list(sub_to_sup_region_id_dict.keys()))

            aggregated_sds.add_objects(description="gpd_geometries", #TODO: based on the current and possible future use of add_objects() simplify the method
                                    dimension_list=(spatial_dim),  #TODO: check why the brackets are necessary
                                    object_list=shapes_aggregated)

            add_region_centroids(aggregated_sds)
        
        #STEP 2b. For other variables except "gpd_centroids", call respective 
        # aggregation functions based on dimensions  
        elif varname != "gpd_centroids":
            ## Time series 
            if spatial_dim in da.dims and time_dim in da.dims:  
                da = aggregate_time_series(sds.xr_dataset[varname],
                                        sub_to_sup_region_id_dict,
                                        mode=aggregation_mode,
                                        xr_weight_array=aggregation_weight)

                aggregated_sds.xr_dataset[varname] = da

            ## 1d variables
            if (spatial_dim in da.dims and f"{spatial_dim}_2" not in da.dims):  
                da = aggregate_values(sds.xr_dataset[varname],
                                    sub_to_sup_region_id_dict,
                                    mode=aggregation_mode)
                aggregated_sds.xr_dataset[varname] = da
            
            ## 2d variables
            if (f"{spatial_dim}" in da.dims and f"{spatial_dim}_2" in da.dims):  
                da = aggregate_connections(sds.xr_dataset[varname],
                                        sub_to_sup_region_id_dict,
                                        mode=aggregation_mode)

                aggregated_sds.xr_dataset[varname] = da

    return aggregated_sds


# spagat.output:
def create_grid_shapefile(sds,
                        file_path, 
                        files_name = "AC_lines",
                        spatial_dim="space",
                        eligibility_variable="2d_locationalEligibility",
                        eligibility_component=None):
    # TODO: dataset class
    
    add_region_centroids(sds)

    buses_0 = []
    buses_1 = []
    geoms = []

    if eligibility_component is not None:
        eligibility_xr_array = sds.xr_dataset[eligibility_variable].sel(
            component=eligibility_component)
    else:
        eligibility_xr_array = sds.xr_dataset[eligibility_variable]  

    for region_id_1 in sds.xr_dataset[f"{spatial_dim}"].values:
        for region_id_2 in sds.xr_dataset[f"{spatial_dim}_2"].values:
            if eligibility_xr_array.sel(space=region_id_1, space_2=region_id_2).values: #TODO: if eligibility_component is not specified, an error is raised here. 
                                                                                        # either make eligibility_component nondefault or change this line to look for at least one non nan value 
                buses_0.append(region_id_1)
                buses_1.append(region_id_2)

                point_1 = sds.xr_dataset.gpd_centroids.sel(space=region_id_1).item(0)
                point_2 = sds.xr_dataset.gpd_centroids.sel(space=region_id_2).item(0)
                line = LineString([(point_1.x, point_1.y), (point_2.x, point_2.y)])

                geoms.append(line)

    # TODO: understand what s_nom and x stand for (look into FINE?) and add or delete them below
    df = pd.DataFrame(
        {
            "bus0": buses_0,
            "bus1": buses_1,
            #      's_nom': ,
            #      'x': ,
        }
    )

    spu.create_gdf(df, geoms, crs=3035, file_path=file_path, files_name=files_name)

    

################################################################################################################
###########                        REPRESENTATION OF RE TIME SERIES                              ###############
################################################################################################################

def represent_RE_technology(capfac_dataarray, 
                            cap_dataarray, 
                            n_timeSeries_perRegion,
                            skip_data_restructure = False):
    '''
    Function represents RE time series and their corresponding capacities using time series clustering methods.
    Clustering method: agglomerative hierarchical clustering, Linkage criteria: Average 
    Distance measure used: Euclidean distance 

    '''

    #STEP 1. Create DataArrays to store the represented time series and capacities
    ## DataArray to store the represented time series
    region_ids_list = capfac_dataarray.region_ids.values #TODO: generalize "region_ids" (can be space or locs)
    time_steps = capfac_dataarray.time.values #TODO: generalize "time" (can be timeStep)

    n_regions = len(region_ids_list)
    n_timeSteps = len(time_steps)

    TS_ids = [f'TS_{i}' for i in range(n_timeSeries_perRegion)] #TODO: change TS to something else ?
    data = np.zeros(n_timeSteps, n_regions, n_timeSeries_perRegion)

    represented_timeSeries = xr.DataArray(data, [('time', time_steps),
                                                  ('region_ids', region_ids_list),
                                                  ('TS_ids', TS_ids)])

    ## DataArray to store the represented capacities
    data = np.zeros((len(region_ids_list), n_timeSeries_perRegion))

    represented_capacities = xr.DataArray(data, [('region_ids', region_ids_list),
                                                  ('TS_ids', TS_ids)])

    #STEP 2. Representation in every region...
    for region in region_ids_list:
        #STEP 2a. Get time series and capacities of current region 
        region_capfac_dataarray = capfac_dataarray.sel(region_ids=region)
        region_cap_dataarray = cap_dataarray.sel(region_ids=region)
        
        #STEP 2b. Preprocess DataArrays 

        #STEP 2b (i). Restructure data
        #INFO: The clustering model, takes <= 2 dimensions. So, x and y coordinates are fused 
        # Transposing dimensions to make sure clustering is performed along x_y dimension (i.e., space not time)
        # Requires skipping during testing. In test data, dimensions are already fused and transposed 
        if skip_data_restructure == False:
            region_capfac_dataarray = region_capfac_dataarray.stack(x_y = ['x', 'y']) 
            region_capfac_dataarray = region_capfac_dataarray.transpose(transpose_coords= True) 
            
            region_cap_dataarray = region_cap_dataarray.stack(x_y = ['x', 'y'])
            region_cap_dataarray = region_cap_dataarray.transpose(transpose_coords= True)

        #STEP 2b (ii). Remove all time series with 0 values 
        region_capfac_dataarray = region_capfac_dataarray.where(region_cap_dataarray>0)
        region_cap_dataarray = region_cap_dataarray.where(region_cap_dataarray>0)
        
        #STEP 2b (iii). Drop NAs 
        region_capfac_dataarray = region_capfac_dataarray.dropna(dim='x_y')
        region_cap_dataarray = region_cap_dataarray.dropna(dim='x_y')

        #Print out number of time series in the region 
        n_ts = len(region_capfac_dataarray['x_y'].values)
        print(f'Number of time series in {region}: {n_ts}')
        
        #STEP 2c. Get power curves from capacity factor time series and capacities 
        region_power_dataarray = region_cap_dataarray * region_capfac_dataarray
        
        #STEP 2d. Clustering  
        agg_cluster = AgglomerativeClustering(n_clusters=n_timeSeries_perRegion, 
                                              affinity="euclidean",  
                                              linkage="average")
        agglomerative_model = agg_cluster.fit(region_capfac_dataarray)
        
        # #store the cluster labels 
        # labels = agglomerative_model.labels_
        # results_labels[region] = labels.tolist()
        
        #STEP 2e. Aggregation
        for i in range(np.unique(agglomerative_model.labels_).shape[0]):
            ## Aggregate capacities 
            cluster_cap = region_cap_dataarray[agglomerative_model.labels_ == i]
            cluster_cap_total = cluster_cap.sum(dim = 'x_y').values
            
            represented_timeSeries.loc[region, TS_ids[i]] = cluster_cap_total
            
            #aggregate capacity factor 
            cluster_power = region_power_dataarray[agglomerative_model.labels_ == i]
            cluster_power_total = cluster_power.sum(dim = 'x_y').values
            cluster_capfac_total = cluster_power_total/cluster_cap_total
            
            represented_capacities.loc[:,region, TS_ids[i]] = cluster_capfac_total
            
    return represented_timeSeries, represented_capacities #, results_labels 
