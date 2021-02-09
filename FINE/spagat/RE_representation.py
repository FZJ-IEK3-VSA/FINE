'''
Functions to represent RE time series 
'''

import numpy as np
import xarray as xr
from sklearn.cluster import AgglomerativeClustering

import FINE.spagat.utils as spu

@spu.timer
def represent_RE_technology(rasterized_RE_ds, n_timeSeries_perRegion, linkage='average'):
    """ Represents RE time series and their corresponding capacities using time series clustering methods.
    Clustering method: agglomerative hierarchical clustering, Linkage criteria: Average 
    Distance measure used: Euclidean distance.
    """
    #STEP 1. Create DataArrays to store the represented time series and capacities
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
    
    #STEP 2. Representation in every region...
    for region in region_ids:
        #STEP 2a. Get time series and capacities of current region 
        regional_ds = rasterized_RE_ds.sel(region_ids = region)
        regional_capfac_da = regional_ds.capfac.where(regional_ds.rasters == 1)
        regional_capacity_da = regional_ds.capacity.where(regional_ds.rasters == 1)

        #STEP 2b. Preprocess regional capfac and capacity dataArrays 

        #STEP 2b (i). Restructure data
        #INFO: The clustering model, takes <= 2 dimensions. So, x and y coordinates are fused 
        # Transposing dimensions to make sure clustering is performed along x_y dimension (i.e., space not time)
        regional_capfac_da = regional_capfac_da.stack(x_y = ['x', 'y']) 
        regional_capfac_da = regional_capfac_da.transpose(transpose_coords= True) 

        regional_capacity_da = regional_capacity_da.stack(x_y = ['x', 'y'])
        regional_capacity_da = regional_capacity_da.transpose(transpose_coords= True)

        #STEP 2b (ii). Remove all time series with 0 values 
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da > 0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da > 0)

        #STEP 2b (iii). Drop NAs 
        regional_capfac_da = regional_capfac_da.dropna(dim='x_y')
        regional_capacity_da = regional_capacity_da.dropna(dim='x_y')

        #Print out number of time series in the region 
        n_ts = len(regional_capfac_da['x_y'].values)
        print(f'Number of time series in {region}: {n_ts}')

        #STEP 2c. Get power curves from capacity factor time series and capacities 
        region_power_da = regional_capacity_da * regional_capfac_da

        #STEP 2d. Clustering  
        agg_cluster = AgglomerativeClustering(n_clusters=n_timeSeries_perRegion, 
                                              affinity="euclidean",  
                                              linkage=linkage)
        agglomerative_model = agg_cluster.fit(regional_capfac_da)

        #STEP 2e. Aggregation
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
def get_one_REtech_per_region(rasterized_RE_ds):
    """Simple aggregation: Within every region, take the weighted mean of capfac 
    time series (capacities being weights), and sum of capacities. 
    
    """
    #STEP 1. Create DataArrays to store the aggregated time series and capacities
    ## DataArray to store the aggregated time series
    region_ids = rasterized_RE_ds['region_ids'].values
    time_steps = rasterized_RE_ds['time'].values

    data = np.zeros((len(time_steps), len(region_ids)))

    aggr_capfac_da = xr.DataArray(data, [('time', time_steps),
                                         ('region_ids', region_ids)])

    ## DataArray to store aggregated capacities
    data = np.zeros((len(region_ids)))

    aggr_capacity_da = xr.DataArray(data, [('region_ids', region_ids)])
    
    #STEP 2. Aggregation in every region...
    for region in region_ids:
        #STEP 2a. Get time series and capacities of current region 
        regional_ds = rasterized_RE_ds.sel(region_ids = region)
        regional_capfac_da = regional_ds.capfac.where(regional_ds.rasters == 1)
        regional_capacity_da = regional_ds.capacity.where(regional_ds.rasters == 1)
        
        #STEP 2b. Preprocess regional capfac and capacity dataArrays 
        
        #STEP 2b (i). Restructure data
        #INFO: The clustering model, takes <= 2 dimensions. So, x and y coordinates are fused 
        # Transposing dimensions to make sure clustering is performed along x_y dimension (i.e., space not time)
        regional_capfac_da = regional_capfac_da.stack(x_y = ['x', 'y']) 
        regional_capfac_da = regional_capfac_da.transpose(transpose_coords= True) 
        
        regional_capacity_da = regional_capacity_da.stack(x_y = ['x', 'y'])
        regional_capacity_da = regional_capacity_da.transpose(transpose_coords= True)
            
        #STEP 2b (ii). Remove all time series with 0 values 
        regional_capfac_da = regional_capfac_da.where(regional_capacity_da>0)
        regional_capacity_da = regional_capacity_da.where(regional_capacity_da>0)
        
        #STEP 2b (iii). Drop NAs 
        regional_capfac_da = regional_capfac_da.dropna(dim='x_y')
        regional_capacity_da = regional_capacity_da.dropna(dim='x_y')
        
        #Print out number of time series in the region 
        n_ts = len(regional_capfac_da['x_y'].values)
        print(f'Number of time series in {region}: {n_ts}')
        
        #STEP 2c. Get power curves from capacity factor time series and capacities 
        regional_power_da = regional_capacity_da * regional_capfac_da
        
        #STEP 2d. Aggregation
        ## capacity
        capacity_total = regional_capacity_da.sum(dim = 'x_y').values
        aggr_capacity_da.loc[region] = capacity_total
        
        ## capacity factor 
        power_total = regional_power_da.sum(dim = 'x_y').values
        capfac_total = power_total/capacity_total
        
        aggr_capfac_da.loc[:,region] = capfac_total
        
    #STEP 4. Create resulting dataset 
    aggr_RE_ds = xr.Dataset({'capacity': aggr_capacity_da,
                             'capfac': aggr_capfac_da}) 
            
    return aggr_RE_ds