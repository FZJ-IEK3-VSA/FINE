import warnings

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import preprocessing as prep
from scipy.cluster import hierarchy
from sklearn import metrics
from typing import Dict, List

def get_scaled_array(array):
    """Scale the given matrix to [0,1].

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be scaled
    
    Returns
    -------
    np.ndarray
        Scaled matrix 
    """

    scaled_min, scaled_max = 0, 1

    if np.max(array) == np.min(array): 
        return array

    return ((array - np.min(array)) / (np.max(array) - np.min(array))) * (scaled_max - scaled_min) + scaled_min

def preprocess_time_series(vars_dict):
    """Preprocess time series variables.

    Parameters
    ----------
    vars_dict : Dict[str, xr.DataArray]
        Dictionary of each time series variable and it's corresponding data in a xr.DataArray. 
        - Dimensions of xr.DataArray - 'component', 'space', 'time'

    Returns
    -------
    ds_ts : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary 
        consists of each valid component name and the corresponding scaled (between [0, 1]) data matrix
        - Size of each matrix: n_regions * n_timesteps 
    """
     
    ds_ts = {}
    
    # For each time series variable-data pair...
    for var_name, da in vars_dict.items():
        #STEP 1. Add the variable to the resuting dict 
        ds_ts.update({var_name: {}})

        #STEP 2. Find the corresponding valid components: valid_component_weight=1, otherwise=0
        var_mean_df = da.mean(dim='space').mean(dim='time').to_dataframe()
        valid_components = list(var_mean_df[var_mean_df[var_name].notna()].index.values)

        #STEP 2. For each valid component, scale the corresponding matrix. Add to resulting dict 
        for comp_name in valid_components:
            scaled_comp_matrix = get_scaled_array(da.sel(component=comp_name).values) 

            ds_ts.get(var_name).update({comp_name : scaled_comp_matrix})
           
    return ds_ts

def preprocess_1d_variables(vars_dict):
    """Preprocess 1-dimensional variables.

    Parameters
    ----------
    vars_dict : Dict[str, xr.DataArray]
        Dictionary of each 1-dimensional variable and it's corresponding data in a xr.DataArray. 
        - Dimensions of xr.DataArray - 'component', 'space'

    Returns
    -------
    ds_1d : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary 
        consists of each valid component name and the corresponding scaled (between [0, 1]) data array
        - Size of each array: n_regions   
    """
    ds_1d = {}
    
    # For each 1d variable-data pair...
    for var_name, da in vars_dict.items():
        #STEP 1. Add the variable to the resuting dict 
        ds_1d.update({var_name: {}})

        #STEP 2. Find the corresponding valid components: valid_comp_weight=1, otherwise=0
        var_mean_df = da.mean(dim='space').to_dataframe()
        valid_components = list(var_mean_df[var_mean_df[var_name].notna()].index.values)
        
        #STEP 2. For each valid component, scale the corresponding matrix. Add to resulting dict 
        for comp_name in valid_components:
            #data = da.sel(component=comp_name).values
            scaled_comp_array = get_scaled_array(da.sel(component=comp_name).values) 

            ds_1d.get(var_name).update({comp_name : scaled_comp_array})

    return ds_1d

def preprocess_2d_variables(vars_dict):  
    """Preprocess 2-dimensional variables.

    Parameters
    ----------
    vars_dict : Dict[str, xr.DataArray]
        Dictionary of each 2-dimensional variable and it's corresponding connectivity data in a xr.DataArray. 
        - Dimensions of xr.DataArray - 'space','space_2'
    
    Returns
    -------
    ds_2d : Dict[str, Dict[str, np.ndarray]]
        For each key (variable name), the corresponding value is a dictionary. This dictionary consists of 
        each valid component name and the corresponding data scaled (between [0, 1]), converted to vector form,
        and translated to distance meaning.
        - Size of each data array: n_regions
    
    Notes
    -----
    For each variable, find it's valid components (components without all NAs)
    For each of these variable-valid component pair, a symmetric connectivity matrix of n_regions * n_regions is obtained
    - Flatten each matrix in `ds_2d` to obtain it's vector form: 
                    [[0.  0.1 0.2]
                    [0.1 0.  1. ]       -->  [0.1 0.2 1. ]   (only the elements from upper or lower triangle 
                    [0.2 1.  0. ]]                            as the other is always redundant in a dist matrix )
    - Translate connectivity (similarity) to distance (dissimilarity) : (1- connectivity vector)
    - Update `ds_2d`    
    """
    ds_2d = {}

    # For each 2d variable-data pair...
    for var_name, da in vars_dict.items():
        #STEP 1. Add the variable to the resuting dict 
        ds_2d.update({var_name: {}})
        
        space1 = da.space.values
        space2 = da.space_2.values
        
        #STEP 2. Find the corresponding valid components: valid_comp_weight=1, otherwise=0
        var_mean_df = da.mean(dim='space').mean(dim='space_2').to_dataframe()
        valid_components = list(var_mean_df[var_mean_df[var_name].notna()].index.values)
        
        #STEP 2. For each valid component...
        for comp_name in valid_components:
            #STEP 2a. Get the corresponding data 
            data_matrix = da.sel(component=comp_name).values # square matrix (dim: space and space_2)
            
            #STEP 2b: Make sure order of space and space_2 is the same
            data_df = pd.DataFrame(data=data_matrix, columns=space2)            
            data_df = data_df[space1]
            
            #STEP 2c: Scale the matrix
            scaled_comp_matrix = get_scaled_array(data_df.to_numpy())

            #STEP 4d. Obtain the vector form of this symmetric connectivity matrix
            scaled_comp_vector = hierarchy.distance.squareform(scaled_comp_matrix, checks=False)   

            #STEP 4c. Convert the value of connectivity (similarity) to distance (dissimilarity)
            scaled_comp_vector = 1 - scaled_comp_vector

            #STEP 4d. Add to resulting dict
            ds_2d.get(var_name).update({comp_name : scaled_comp_vector})

    return ds_2d

    
def preprocess_dataset(sds): 
    """Preprocess xarray dataset.

    Parameters
    ----------
    sds : Instance of SpagatDataset
        Refer to SpagatDataset class in dataset.py for more information 

    Returns
        dict_ts, dict_1d, dict_2d : Dict
            Dictionaries obtained from 
                preprocess_time_series(), 
                preprocess_1d_variables(),
            and preprocess_2d_variables(), respectively
    """
    ds_extracted = sds.xr_dataset

    component_list = list(ds_extracted['component'].values)
    n_components = len(component_list)
    n_regions = len(ds_extracted['space'].values)

    #STEP 0. Traverse all variables in the dataset, and put them in separate categories
    #NOTE: vars_ts, vars_1d, vars_2d -> dicts of variables and their corresponding dataArrays
    vars_ts = {}
    vars_1d = {}
    vars_2d = {}

    for varname, da in ds_extracted.data_vars.items():
        
        if sorted(da.dims) == sorted(('component', 'time', 'space')):   
            da = da.transpose('component','space','time') #require sorting dimensions 
            vars_ts[varname] = da

        elif sorted(da.dims) == sorted(('component','space')):
            vars_1d[varname] = da

        elif sorted(da.dims) == sorted(('component','space','space_2')):
            vars_2d[varname] = da

        else:
            warnings.warn(f'Variable {varname} has dimensions {str(da.dims)} which are not considered for spatial aggregation.')

    #STEP 1. Preprocess Time Series
    dict_ts = preprocess_time_series(vars_ts)

    #STEP 2. Preprocess 1d Variables
    dict_1d = preprocess_1d_variables(vars_1d)

    #STEP 3. Preprocess 2d Variables 
    dict_2d = preprocess_2d_variables(vars_2d)

    return dict_ts, dict_1d, dict_2d
    

def get_custom_distance(dict_ts, dict_1d, dict_2d, 
                        n_regions, 
                        region_index_x, 
                        region_index_y,
                        weights=None):  
    """Custom distance function.

    Parameters
    ----------
    dict_ts, dict_1d, dict_2d : Dict
        Dictionaries obtained as a result of preprocess_dataset()
    n_regions : int
        Total number of regions in the given data 
    region_index_x, region_index_y : int 
        Indicate the two regions between which the custom distance is to be calculated 
        range of these indices - [0, n_regions)
    weights : Dict
        weights for each variable-component pair 

    Returns
    -------
    float 
        Custom distance value 
    """

    #STEP 1. Check if weights are specified correctly 
    if weights != None:
        if 'components' not in weights.keys():
            raise ValueError("weights dictionary must contain a 'components' dictionary within it")

        if not set(weights.keys()).issubset({'components', 'variables'}):
            raise ValueError("Something is wrong with weights dictionary. Please refer to the its template in the doc string")

        if 'variables' in weights.keys():
            var_weights = weights.get('variables')
            if isinstance(var_weights, str):
                if var_weights != 'all':
                    warnings.warn("Unrecognised string for variable weights. All variables will be weighted")
                    weights['variables'] = 'all'
        
        else:
            warnings.warn("variable list not found in weights dictionary. All variables will be weighted")
            weights.update({'variables' : 'all'})     


    def _get_var_comp_weight(var_name, comp_name):
        """Private function to get weight corresponding to a variable-component pair
        """  

        wgt = 1

        if weights != None:

            [var_category, var] = var_name.split("_") #strip the category and take only var 
            [comp_class, comp] = comp_name.split(", ") #strip the model class and take only comp 

            var_weights = weights.get('variables')
            comp_weights = weights.get('components')

            if (var_weights == 'all') or (var in var_weights):
                if comp_weights.get('all') != None:
                    wgt = comp_weights.get('all')
                elif comp_weights.get(comp) != None:
                    wgt = comp_weights.get(comp)

        return wgt  

    #STEP 2. Find distance for each variable category separately 

    #STEP 3a. Distance of Time Series category
    distance_ts = 0
    for var_name, var_dict in dict_ts.items():
        for comp_name, data_matrix in var_dict.items():
            # (i) Get weight
            var_comp_weight = _get_var_comp_weight(var_name, comp_name)

            # (ii) Extract data corresponding to the variable-component pair in both regions  
            region_x_data = data_matrix[region_index_x]
            region_y_data = data_matrix[region_index_y]
            
            # (ii) Calculate distance 
            #INFO: ts_region_x and ts_region_y are vectors, 
            # subtract the vectors, square each element and add all elements. And multiply with its weight
            distance_ts += sum(np.power((region_x_data - region_y_data),2))  * var_comp_weight  

    #STEP 3b. Distance of 1d Variables category
    distance_1d = 0
    for var_name, var_dict in dict_1d.items():
        for comp_name, data_array in var_dict.items():
            # (i) Get weight
            var_comp_weight = _get_var_comp_weight(var_name, comp_name)

            # (ii) Extract data corresponding to the variable in both regions 
            region_x_data = data_array[region_index_x]
            region_y_data = data_array[region_index_y]

            # (iii) Calculate distance
            distance_1d += pow(region_x_data - region_y_data, 2)  * var_comp_weight  

    #STEP 3c. Distance of 2d Variables category
    distance_2d = 0

    #STEP 3c (i). Since ds_2d is a condensed matrix, we have to get dist. corresponding to the two given regions
    region_index_x_y = region_index_x * (n_regions - region_index_x) + (region_index_y - region_index_x) -1                

    for var_name, var_dict in dict_2d.items():
        for comp_name, data_array in var_dict.items():
            # (i) Get weight
            var_comp_weight = _get_var_comp_weight(var_name, comp_name)

            # (ii) Extract data corresponding to the variable in both regions 
            dist = data_array[region_index_x_y]

            if not np.isnan(dist):        #INFO: if the regions are not connected the value will be na
                # Calculate the distance 
                distance_2d += pow(dist, 2) * var_comp_weight

    #STEP 4. Add all three distances 
    return distance_ts + distance_1d + distance_2d 


def get_custom_distance_matrix(ds_ts, 
                            ds_1d, 
                            ds_2d, 
                            n_regions,
                            weights=None):

    """For every region combination, calculates the custom distance by calling get_custom_distance().
                                                      
    Parameters
    ----------
    ds_ts, ds_1d, ds_2d : Dict
        Dictionaries obtained as a result of preprocess_dataset() 
    n_regions : int
        Total number of regions in the given data 
        range of these indices - [0, n_regions)
    weights : Dict
        weights for each variable-component pair 
        
    Returns
    -------
    distMatrix : np.ndarray 
        A n_regions by n_regions hollow, symmetric distance matrix 
    """
    distMatrix = np.zeros((n_regions,n_regions))

    #STEP 1. For every region pair, calculate the distance 
    for i in range(n_regions):
        for j in range(i+1,n_regions):
            distMatrix[i,j] = get_custom_distance(ds_ts,
                                                ds_1d, 
                                                ds_2d, 
                                                n_regions, 
                                                i,j,
                                                weights)

    #STEP 2. Only upper triangle has values, reflect these values in lower triangle to make it a hollow, symmetric matrix
    distMatrix += distMatrix.T - np.diag(distMatrix.diagonal())  

    return distMatrix


def get_connectivity_matrix(sds):
    """Generates connectiviy matrix. #TODO: update docstring
        - In this matrix, if two regions are connected, it is indicated as 1 and 0 otherwise. 
        - For every region pair, as long as they have atleast one non-zero 2d-variable value, 
          related to 'pipeline' component, they are regarded as connected.
        - If no component related to 'pipeline' is present in the data, then all components
          are considered.
                                                      
    Parameters
    ----------
    sds : Instance of SpagatDataset
        Refer to SpagatDataset class in dataset.py for more information
    
    Returns
    -------
    connectivity_matrix : np.ndarray 
        A n_regions by n_regions symmetric matrix 
    """
    ds_extracted = sds.xr_dataset
    
    n_regions = len(ds_extracted['space'].values)

    connectivity_matrix = np.zeros((n_regions,n_regions))

    #STEP 1: Check for contiguous neighbors 
    geometries = gpd.GeoSeries(ds_extracted['gpd_geometries']) #NOTE: disjoint seems to work only on geopandas or geoseries object
    for ix, geom in enumerate(geometries):
        neighbors = geometries[~geometries.disjoint(geom)].index.tolist()
        connectivity_matrix[ix, neighbors] = 1
    
    #STEP 2: Find nearest neighbor for island regions   
    for row in range(len(connectivity_matrix)):
        if np.count_nonzero(connectivity_matrix[row,:] == 1) == 1: #if a region is connected only to itself
            
            #get the nearest neighbor based on regions centroids 
            centroid_distances = ds_extracted.centroid_distances.values[row,:] 
            nearest_neighbor_idx = np.argmin(centroid_distances[np.nonzero(centroid_distances)])
            
            #make the connection between the regions (both ways to keep it symmetric)
            connectivity_matrix[row,nearest_neighbor_idx], connectivity_matrix[nearest_neighbor_idx, row] = 1, 1
            
    #STEP 3: Additionally, check if there are transmission between regions that are not yet connected in the 
    #connectivity matrix 
    for data_var in ds_extracted.data_vars:
        if data_var[:3] == "2d_":
            comp_xr = ds_extracted[data_var]
            valid_comps_xr = comp_xr.dropna(dim='component') #drop invalid components

            valid_comps_xr = valid_comps_xr.fillna(0) #for safety purpose, does not affect further steps anyway

            valid_comps_xr = valid_comps_xr.sum(dim=['component'])
            
            connectivity_matrix[valid_comps_xr.values>0] = 1 #if a pos, non-zero value exits, make a connection!
    
    return connectivity_matrix


