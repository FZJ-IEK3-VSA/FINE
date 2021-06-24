import numpy as np 
import geopandas as gpd 
from rasterio import features
from affine import Affine
import xarray as xr

import weighted
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.cbook import violin_stats


def rasterize_geometry(geometry, coords, latitude='y', longitude='x'):
    """Given a geometry and geolocations, it masks the geolocations 
    such that all the geolocations within the geometry are indicated
    by a 1 and rest are NAs. 

    Parameters
    ----------
    geometry : a polygon or a multiploygon 
    coords : Dict-like 
        Holds latitudes and longitudes 
    latitude : str, optional (default='y')  
        The description of latitude in `coords` 
    longitude : str, optional (default='x') 
        The description of longitude in `coords`

    Returns
    -------
    raster : np.ndarray 
        A 2d matrix of size latitudes * longitudes 
        If a latitude-longitude pair falls within the `geometry` then 
        the value at this point in the matrix is 1, otherwise NA 
    """

    #STEP 1. Get the affine transformation
    lat = np.asarray(coords[latitude])
    lon = np.asarray(coords[longitude])

    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    transform = trans * scale
    
    #STEP 2. Get the raster mask 
    out_shape = (len(lat), len(lon))
    
    raster = features.rasterize([geometry], 
                                out_shape=out_shape,
                                fill=np.nan, 
                                transform=transform,
                                dtype=float)
    
    return raster 


def rasterize_xr_ds(gridded_RE_ds, 
                    CRS_attr,
                    shp_file, 
                    index_col='region_ids', 
                    geometry_col='geometry',
                    longitude='x', 
                    latitude='y'):
    """Converts the gridded data in an xarray dataset to raster data 
    based on the given shapefile.
    
    Parameters
    ----------
    gridded_RE_ds : str/xr.Dataset 
        Either the path to the dataset or the read-in xr.Dataset 
        2 mandatory dimensions in this data - `latitude` and `longitude`  
    CRS_attr : str
        The attribute in `gridded_RE_ds` that holds its 
        Coordinate Reference System (CRS) information 
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
    rasterized_RE_ds : xr.Dataset 
        - Additional dimension with name `index_col` 
        - Additional variable with name 'rasters' and values as rasters 
          corresponding to each geometry in `shp_file`
    """

    #STEP 1. Read in the files 
    if isinstance(gridded_RE_ds, str): 
        gridded_RE_ds = xr.open_dataset(gridded_RE_ds)
    elif not isinstance(gridded_RE_ds, xr.Dataset):
        raise TypeError("gridded_RE_ds must either be a path to a netcdf file or xarray dataset")
    
    if isinstance(shp_file, str): 
        shp_file = gpd.read_file(shp_file)
    elif not isinstance(shp_file, gpd.geodataframe.GeoDataFrame):
        raise TypeError("shp_file must either be a path to a shapefile or a geopandas dataframe")

    #STEP 2. Match the CRS of shapefile to that of the dataset
    shp_file = shp_file.to_crs({'init': gridded_RE_ds.attrs[CRS_attr]})
    
    #STEP 3. rasterize each geometry and add it to new data_var "rasters"

    region_geometries = shp_file[geometry_col]
    region_indices = shp_file[index_col]

    rasterized_RE_ds = gridded_RE_ds.expand_dims({'region_ids' : region_indices}) 

    coords = rasterized_RE_ds.coords 
    
    rasterized_RE_ds['rasters'] = (['region_ids', latitude, longitude],
                                      [rasterize_geometry(geometry, 
                                                        coords, 
                                                        longitude=longitude, 
                                                        latitude=latitude)
                                            for geometry in region_geometries])

    return rasterized_RE_ds
    
#PLOTS =================================================================================================
#TODO: update docstrings 
#VIOLIN PLOT ------------------------------------------------------
def vdensity_with_weights(weights):
    ''' Outer function allows innder function access to weights. Matplotlib
    needs function to take in data and coords, so this seems like only way
    to 'pass' custom density function a set of weights '''
    
    def vdensity(data, coords):
        ''' Custom matplotlib weighted violin stats function '''
        # Using weights from closure, get KDE fomr statsmodels
        weighted_cost = sm.nonparametric.KDEUnivariate(data)
        weighted_cost.fit(fft=False, weights=weights)

        # Return y-values for graph of KDE by evaluating on coords
        return weighted_cost.evaluate(coords)
    return vdensity

def custom_violin_stats(data, weights):
    # Get weighted median and mean (using weighted module for median)
    median = weighted.quantile_1D(data, weights, 0.5)
    mean, sumw = np.ma.average(data, weights=list(weights), returned=True)
    
    # Use matplotlib violin_stats, which expects a function that takes in data and coords
    # which we get from closure above
    results = violin_stats(data, vdensity_with_weights(weights))
    
    # Update result dictionary with our updated info
    results[0][u"mean"] = mean
    results[0][u"median"] = median
    
    # No need to do this, since it should be populated from violin_stats
    # results[0][u"min"] =  np.min(data)
    # results[0][u"max"] =  np.max(data)

    return results


def plot_violin_plot(data, weights, position, color, ax, widths=2):
    
    vpstats1 = custom_violin_stats(data=data, weights=weights)

    vplot = ax.violin(vpstats1, [position], vert=True, showmeans=False,
                      showextrema=False, showmedians=False, widths=widths)
    
    for pc in vplot['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha = 0.3

#TODO: require testing (with optimization results) (all functions below)
# PERCENTILE PLOT ----------------------------------------------------
def plot_percentile_plot(original_capfac, RESULTS_PATH, data = 'Wind', clusters_list=['mean', 2, 5, 10]):
    
    x = np.arange(0,8760,1)

    n = 5
    if data=='Wind':
        colormap = cm.Blues 
    else:
        colormap = cm.Oranges
    percentile_list = [(0, 100), (5, 95), (10, 90), (20, 80), (40, 60)]

    half = int((n-1)/2)

    fig, ax = plt.subplots(nrows=len(clusters_list)+1, ncols=1, sharey=True, sharex=True, figsize=(19,8))

    #original data
    y = np.percentile(original_capfac, 50, axis=1)
    ax[0].plot(x, y, color='black')
    for i, percentiles in enumerate(percentile_list):
        y1 = np.percentile(original_capfac, percentiles[0], axis=1)
        y2 = np.percentile(original_capfac, percentiles[1], axis=1)
        
        ax[0].fill_between(x, y1,y2,color=colormap(i/half))
        ax[0].set_title('Original data')  
    ax_no = 1   
    for cluster in clusters_list:
        
        if cluster=='mean':
            mean_capfac = xr.open_dataarray(RESULTS_PATH  / 'Mean' / f'{data}_mean_capfac.nc4')
            ax[ax_no].plot(range(8760), mean_capfac, color='black') 
            ax[ax_no].set_title('Mean of time series')
        else:
            #clustered data
            clustered_capfac = xr.open_dataarray(RESULTS_PATH  / 'Hier_Ward' / f'{data}_{cluster}cl_capfac.nc4')
            capfac_clustered = clustered_capfac.sel(region_ids='01_es').values

            y = np.percentile(capfac_clustered, 50, axis=1)
            ax[ax_no].plot(x, y, color='black')
            
            for i, percentiles in enumerate(percentile_list):
                y1 = np.percentile(capfac_clustered, percentiles[0], axis=1)
                y2 = np.percentile(capfac_clustered, percentiles[1], axis=1)

                ax[ax_no].fill_between(x, y1,y2,color=colormap(i/half))
            ax[ax_no].set_title(f'{cluster} time series')
        ax_no+=1
    #get common x and y labels 
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Capacity factor', fontsize=16)
    
    fig.tight_layout()


# STACKED BAR PLOT ----------------------------------------------------------
def stacked_bar_plot(x, y_list, labels=None, width=1.3):

    y_pos = x[:]
    y_pos[0] = 0  #change 'mean' to pos 0
    
    x_tick_labels = x[:]
    x_tick_labels[0] = 1
    
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    p1 = plt.bar(y_pos, y_list[0], width, linewidth=0)

    y_bottom = y_list[0]

    p_list = [p1]
    
    for n in np.arange(1, len(labels)):
        pn = plt.bar(y_pos, y_list[n], width, bottom=y_bottom, linewidth=0)

        p_list.append(pn)

        y_bottom = [y_bottom[i] + y_list[n][i] for i in np.arange(len(x))]

    plt.xticks(y_pos, x_tick_labels)
    pn0_set_for_legend = (p[0] for p in p_list)
    label_set_for_legend = (label for label in labels)
    
    ax.set_xlabel('Number of time series', fontsize=14)
    ax.set_ylabel('TAC (1e9 Euro/a)', fontsize=14)
    
    #grid lines 
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='-', alpha=0.3)
    
    plt.legend(pn0_set_for_legend, labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=4)
    
    return fig, ax

# LINE PLOT ------------------------------------------------
def plot_line_plot(x, y):
    
    y_pos = x[:]
    y_pos[0] = 0  #change 'mean' to pos 0
    
    x_tick_labels = x[:]
    x_tick_labels[0] = 1
    
    fig, ax = plt.subplots(figsize=(15, 3))
    p1 = plt.plot(y_pos, y, 'black')
    
    plt.xticks(y_pos, x_tick_labels)
    
    ax.set_xlabel('Number of time series per region', fontsize=14)
    ax.set_ylabel('Time taken (in minutes)', fontsize=14)
