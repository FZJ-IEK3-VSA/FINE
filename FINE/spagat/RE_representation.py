'''
Functions to represent RE time series 
'''

import numpy as np
import xarray as xr
from sklearn.cluster import AgglomerativeClustering

import FINE.spagat.utils as spu
import FINE.spagat.RE_representation_utils as RE_rep_utils

@spu.timer
def represent_RE_technology(gridded_RE_ds, 
                            shp_file,
                            n_timeSeries_perRegion, 
                            linkage='average', 
                            index_col='region_ids', 
                            geometry_col='geometry',
                            longitude='x', 
                            latitude='y'):
    """Represents RE time series and their corresponding capacities, within each region
    using time series clustering methods.
    - Clustering method: agglomerative hierarchical clustering
    - Distance measure: Euclidean distance

    Parameters
    ----------
    gridded_RE_ds : str/xr.Dataset 
        Either the path to the dataset or the read-in xr.Dataset
        - Dimensions in this data - latitude, longitude, and 'time' 
        - variables - 'capfac' and 'capacity' #TODO: generalize this
    shp_file : str/Shapefile
        Either the path to the shapefile or the read-in shapefile 
        that should be added to `gridded_RE_ds`
    n_timeSeries_perRegion : strictly int 
        The number of time series to which the original set should be reduced,
        within each region 
    linkage : str, optional (default='average') 
        The linkage criterion to be used with agglomerative hierarchical clustering. 
        Can be 'complete', 'single', etc. 
    index_col : str, optional (default='region_ids')
        The column in `shp_file` that needs to be taken as location-index in `gridded_RE_ds`
    geometry_col : str, optional (default='geometry')
        The column in `shp_file` that holds geometries 
    longitude : str, optional (default='x')
        The dimension name in `gridded_RE_ds` that corresponds to longitude 
    latitude : str, optional (default='y')
        The dimension name in `gridded_RE_ds` that corresponds to latitude

    
    Returns
    -------
    represented_RE_ds : xr.Dataset 
        - Dimensions in this data - 'time', 'region_ids', 'TS_ids'
        - The dimension 'region_ids' has its coordinates corresponding to `index_col`
        -  Within each region, different time seires are indicated by the 'TS_ids'   
    """
    if isinstance(gridded_RE_ds, str): 
        gridded_RE_ds = xr.open_dataset(gridded_RE_ds)
    elif not isinstance(gridded_RE_ds, xr.Dataset):
        raise TypeError("gridded_RE_ds must either be a path to a netcdf file or xarray dataset")
    
    #STEP 1. Rasterize the dataset
    rasterized_RE_ds = RE_rep_utils.add_shapes_from_shp(gridded_RE_ds, 
                                                        shp_file, 
                                                        index_col=index_col, 
                                                        geometry_col=geometry_col,
                                                        longitude=longitude, 
                                                        latitude=latitude)

    #STEP 2. Create DataArrays to store the represented time series and capacities
    ## DataArray to store the represented time series
    #TODO: maybe instead of dataarrays, you can directly output a data dict for FINE here.
    time_steps = rasterized_RE_ds["time"].values  #quickfix might be required - X.time.values = range(8760)
    region_ids = rasterized_RE_ds["region_ids"].values

    n_regions = len(region_ids)
    n_timeSteps = len(time_steps)

    TS_ids = [f'TS_{i}' for i in range(n_timeSeries_perRegion)] #TODO: change TS to something else ?
    data = np.zeros((n_timeSteps, n_regions, n_timeSeries_perRegion))

    represented_timeSeries = xr.DataArray(data, [('time', time_steps),
                                                  ('region_ids', region_ids),
                                                  ('TS_ids', TS_ids)])

    ## DataArray to store the represented capacities
    data = np.zeros((len(region_ids), n_timeSeries_perRegion))

    represented_capacities = xr.DataArray(data, [('region_ids', region_ids),
                                                  ('TS_ids', TS_ids)])
    
    #STEP 3. Representation in every region...
    for region in region_ids:
        #STEP 3a. Get time series and capacities of current region 
        regional_ds = rasterized_RE_ds.sel(region_ids = region)
        regional_capfac_da = regional_ds.capfac.where(regional_ds.rasters == 1)
        regional_capacity_da = regional_ds.capacity.where(regional_ds.rasters == 1)

        #STEP 3b. Preprocess regional capfac and capacity dataArrays 

        #STEP 3b (i). Restructure data
        #INFO: The clustering model, takes <= 2 dimensions. So, x and y coordinates are fused 
        # Transposing dimensions to make sure clustering is performed along x_y dimension (i.e., space not time)
        regional_capfac_da = regional_capfac_da.stack(x_y = ['x', 'y']) 
        regional_capfac_da = regional_capfac_da.transpose(transpose_coords= True) 

        regional_capacity_da = regional_capacity_da.stack(x_y = ['x', 'y'])
        regional_capacity_da = regional_capacity_da.transpose(transpose_coords= True)

        #STEP 3b (ii). Remove all time series with 0 values 
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da > 0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da > 0)

        #STEP 3b (iii). Drop NAs 
        regional_capfac_da = regional_capfac_da.dropna(dim='x_y')
        regional_capacity_da = regional_capacity_da.dropna(dim='x_y')

        #Print out number of time series in the region 
        n_ts = len(regional_capfac_da['x_y'].values)
        print(f'Number of time series in {region}: {n_ts}')

        #STEP 3c. Get power curves from capacity factor time series and capacities 
        region_power_da = regional_capacity_da * regional_capfac_da

        #STEP 3d. Clustering  
        agg_cluster = AgglomerativeClustering(n_clusters=n_timeSeries_perRegion, 
                                              affinity="euclidean",  
                                              linkage=linkage)
        agglomerative_model = agg_cluster.fit(regional_capfac_da)

        #STEP 3e. Aggregation
        for i in range(np.unique(agglomerative_model.labels_).shape[0]):
            ## Aggregate capacities 
            cluster_capacity = regional_capacity_da[agglomerative_model.labels_ == i]
            cluster_capacity_total = cluster_capacity.sum(dim = 'x_y').values

            represented_capacities.loc[region, TS_ids[i]] = cluster_capacity_total

            #aggregate capacity factor 
            cluster_power = region_power_da[agglomerative_model.labels_ == i]
            cluster_power_total = cluster_power.sum(dim = 'x_y').values
            cluster_capfac_total = cluster_power_total/cluster_capacity_total

            represented_timeSeries.loc[:,region, TS_ids[i]] = cluster_capfac_total
            
        #STEP 4. Create resulting dataset 
        represented_RE_ds = xr.Dataset({'capacity': represented_capacities,
                                        'capfac': represented_timeSeries}) 
    
    return represented_RE_ds 


@spu.timer
def get_one_REtech_per_region(gridded_RE_ds, 
                            shp_file, 
                            index_col='region_ids', 
                            geometry_col='geometry',
                            longitude='x', 
                            latitude='y'):
    """Performs simple aggregation: 
    Within every region, calculates the weighted mean of capfac 
    time series (capacities being weights), and sums the capacities. 
    
    Parameters
    ----------
    gridded_RE_ds : str/xr.Dataset 
        Either the path to the dataset or the read-in xr.Dataset
        - Dimensions in this data - latitude, longitude, and 'time' 
        - variables - 'capfac' and 'capacity' 
    shp_file : str/Shapefile
        Either the path to the shapefile or the read-in shapefile 
        that should be added to `gridded_RE_ds`
    index_col : str, optional (default='region_ids')
        The column in `shp_file` that needs to be taken as location-index in `gridded_RE_ds`
    geometry_col : str, optional (default='geometry')
        The column in `shp_file` that holds geometries 
    longitude : str, optional (default='x')
        The dimension name in `gridded_RE_ds` that corresponds to longitude 
    latitude : str, optional (default='y')
        The dimension name in `gridded_RE_ds` that corresponds to latitude

    
    Returns
    -------
    aggregated_RE_ds : xr.Dataset 
        - Dimensions in this data - 'time', 'region_ids'
        - The dimension 'region_ids' has its coordinates corresponding to `index_col` 
    """
    if isinstance(gridded_RE_ds, str): 
        gridded_RE_ds = xr.open_dataset(gridded_RE_ds)
    elif not isinstance(gridded_RE_ds, xr.Dataset):
        raise TypeError("gridded_RE_ds must either be a path to a netcdf file or xarray dataset")
    
    #STEP 1. Rasterize the dataset
    rasterized_RE_ds = RE_rep_utils.add_shapes_from_shp(gridded_RE_ds, 
                                                        shp_file, 
                                                        index_col=index_col, 
                                                        geometry_col=geometry_col,
                                                        longitude=longitude, 
                                                        latitude=latitude)

    #STEP 2. Create DataArrays to store the aggregated time series and capacities
    ## DataArray to store the aggregated time series
    region_ids = rasterized_RE_ds['region_ids'].values
    time_steps = rasterized_RE_ds['time'].values

    data = np.zeros((len(time_steps), len(region_ids)))

    aggr_capfac_da = xr.DataArray(data, [('time', time_steps),
                                         ('region_ids', region_ids)])

    ## DataArray to store aggregated capacities
    data = np.zeros((len(region_ids)))

    aggr_capacity_da = xr.DataArray(data, [('region_ids', region_ids)])
    
    #STEP 3. Aggregation in every region...
    for region in region_ids:
        #STEP 3a. Get time series and capacities of current region 
        regional_ds = rasterized_RE_ds.sel(region_ids = region)
        regional_capfac_da = regional_ds.capfac.where(regional_ds.rasters == 1)
        regional_capacity_da = regional_ds.capacity.where(regional_ds.rasters == 1)
        
        #STEP 3b. Preprocess regional capfac and capacity dataArrays 
        
        #STEP 3b (i). Restructure data
        #INFO: The clustering model, takes <= 2 dimensions. So, x and y coordinates are fused 
        # Transposing dimensions to make sure clustering is performed along x_y dimension (i.e., space not time)
        regional_capfac_da = regional_capfac_da.stack(x_y = ['x', 'y']) 
        regional_capfac_da = regional_capfac_da.transpose(transpose_coords= True) 
        
        regional_capacity_da = regional_capacity_da.stack(x_y = ['x', 'y'])
        regional_capacity_da = regional_capacity_da.transpose(transpose_coords= True)
            
        #STEP 3b (ii). Remove all time series with 0 values 
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da>0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da>0)
        
        #STEP 3b (iii). Drop NAs 
        regional_capfac_da = regional_capfac_da.dropna(dim='x_y')
        regional_capacity_da = regional_capacity_da.dropna(dim='x_y')
        
        #Print out number of time series in the region 
        n_ts = len(regional_capfac_da['x_y'].values)
        print(f'Number of time series in {region}: {n_ts}')
        
        #STEP 3c. Get power curves from capacity factor time series and capacities 
        regional_power_da = regional_capacity_da * regional_capfac_da
        
        #STEP 3d. Aggregation
        ## capacity
        capacity_total = regional_capacity_da.sum(dim = 'x_y').values
        aggr_capacity_da.loc[region] = capacity_total
        
        ## capacity factor 
        power_total = regional_power_da.sum(dim = 'x_y').values
        capfac_total = power_total/capacity_total
        
        aggr_capfac_da.loc[:,region] = capfac_total
        
    #STEP 4. Create resulting dataset 
    aggregated_RE_ds = xr.Dataset({'capacity': aggr_capacity_da,
                             'capfac': aggr_capfac_da}) 
            
    return aggregated_RE_ds