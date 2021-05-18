import warnings

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import preprocessing as prep
from scipy.cluster import hierarchy
from sklearn import metrics
from typing import Dict, List

def get_scaled_matrix(matrix, scaled_min = 0, scaled_max = 1):
    """Scale the given matrix to specificied range.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to be scaled
    scaled_min : int, optional (default=0) 
        Minimum value in the scaled range.  
    scaled_max: int, optional (default=1) 
        Maximum value in the scaled range.

    Returns
    -------
    np.ndarray
        Scaled matrix 
    """

    if np.max(matrix) == np.min(matrix): 
        return matrix

    return ((matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))) * (scaled_max - scaled_min) + scaled_min

def preprocess_time_series(vars_dict, n_regions, n_components):
    """Preprocess time series variables.

    Parameters
    ----------
    vars_dict : Dict[str, xr.DataArray]
        Dictionary of each time series variable and it's corresponding data in a xr.DataArray. 
        - Dimensions of xr.DataArray - 'component','space','TimeStep'
    n_regions : int
        Number of regions present in the data. Corresponds to 'space' dimension 
    n_components : int
        Number of components present in the data. Corresponds to 'component' dimension

    Returns
    -------
    ds_ts : Dict[str, np.ndarray]
        - For each key (variable name), the corresponding value is a scaled and flattened data 
          matrix based on its valid components
        - Size of the matrix: Row (n_regions) * Column (n_valid_components * n_timesteps)
        - The matrix sub-blocks corresponding to each valid component are scaled to [0,1]     
    """
     
    ds_ts = {}
    
    # For each time series variable, data pair...
    for var, da in vars_dict.items():
        matrix_var = np.array([np.zeros(n_regions)]).T

        #STEP 1. Find the corresponding valid components: valid_component_weight=1, otherwise=0
        var_mean_df = da.mean(dim="space").mean(dim="TimeStep").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])

        #STEP 2. For each valid component...
        for comp_id in valid_component_ids:
            #STEP 2a. Scale the corresponding matrix
            matrix_var_c = get_scaled_matrix(da[comp_id].values) 

            #STEP 2b. Concatenate the resulting matrix to the final matrix of the corresponding variable
            matrix_var = np.concatenate((matrix_var, matrix_var_c), axis=1)
        
        #STEP 3. Delete the first column of zeros (created initially) and add the matrix it to final dict 
        matrix_var = np.delete(matrix_var,0,1)
        ds_ts[var] = matrix_var
           
    return ds_ts

def preprocess_1d_variables(vars_dict, n_components):
    """Preprocess 1-dimensional variables.

    Parameters
    ----------
    vars_dict : Dict[str, xr.DataArray]
        Dictionary of each 1-dimensional variable and it's corresponding data in a xr.DataArray. 
        - Dimensions of xr.DataArray - 'component','space'
    n_components : int
        Number of components present in the data. Corresponds to 'component' dimension

    Returns
    -------
    ds_1d : Dict[str, np.ndarray]
        - For each key (variable name), the corresponding value is a matrix for all valid component data
        - Size of the matrix: Row (n_regions) * Column (n_valid_components)
        - Each column in the matrix (corresponding to a valid component) is scaled to [0,1]     
    """
    ds_1d = {}

    min_max_scaler = prep.MinMaxScaler()
    
    # For each 1d variable, data pair...
    for var, da in vars_dict.items():
        
        #STEP 1. Find the corresponding valid components: valid_comp_weight=1, otherwise=0
        var_mean_df = da.mean(dim="space").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])
        
        #STEP 2. Retain only the valid components' data 
        data = da.values[valid_component_ids]

        #STEP 3. Scale data and add it to final dict 
        ds_1d[var] = min_max_scaler.fit_transform(data.T)

    return ds_1d

def preprocess_2d_variables(vars_dict, n_components):  
    """Preprocess 2-dimensional variables.

    Parameters
    ----------
    vars_dict : Dict[str, xr.DataArray]
        Dictionary of each 2-dimensional variable and it's corresponding connectivity data in a xr.DataArray. 
        - Dimensions of xr.DataArray - 'space','space_2'
    n_components : int
        Number of components present in the data. Corresponds to 'component' dimension
    
    Returns
    -------
    ds_2d : Dict[str, Dict[int, np.ndarray]]
        - For each key (variable name), the corresponding value is a dictionary with:
            - key: index of the valid component
            - value: - a flattened vector of scaled distances, scaled to [0,1]  
                     - size of the vector: (n_regions) 
    
    Notes
    -----
    For each variable, find it's valid components (components without all NAs)
    For each of these variable-valid component pair, a symmetric connectivity matrix of n_regions * n_regions is obtained
    - Flatten each matrix in `ds_2d` as such to obtain it's vector form: 
                    [[0.  0.1 0.2]
                    [0.1 0.  1. ]       -->  [0.1 0.2 1. ]   (only the elements from upper or lower triangle 
                    [0.2 1.  0. ]]                            as the other is always redundant in a dist matrix )
    - Translate connectivity (similarity) to distance (dissimilarity) : (1- connectivity vector)
    - Update `ds_2d`    
    """
    ds_2d = {}

    # For each 2d variable, data pair...
    for var, da in vars_dict.items():
        
        ds_2d_var = {}
        
        space1 = da.space.values
        space2 = da.space_2.values
        
        #STEP 1. Find the valid components
        var_mean_df = da.mean(dim="space").mean(dim="space_2").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])
        
        #STEP 2. For each valid component...
        for comp_id in valid_component_ids:
            
            var_matr = da[comp_id].values # square matrix (dim: space and space_2)
            
            #STEP 2a: Make sure order of space and space_2 is the same
            da_comp_df = pd.DataFrame(data=var_matr,columns=space2)            
            da_comp_df = da_comp_df[space1]
            
            #STEP 2b: Scale the matrix
            ds_2d_var[comp_id] = get_scaled_matrix(da_comp_df.to_numpy())
        
        #STEP 3. Add it to resultant dict
        ds_2d[var] = ds_2d_var


    # For each var, var_dict pair in the resultant dict created above...
    for var, var_dict in ds_2d.items():
        
        # For each component, matrix pair in var_dict...
        for comp, data in var_dict.items():
            
            #STEP 4a. Obtain the vector form of this symmetric connectivity matrix
            vec = hierarchy.distance.squareform(data, checks=False)    

            #STEP 4b. Convert the value of connectivity (similarity) to distance (dissimilarity)
            vec = 1 - vec
            
            # Add it to resultant dict 
            ds_2d[var][comp] = vec
  
    return ds_2d

    
def preprocess_dataset(sds): 
    """Preprocess xarray dataset.

    Parameters
    ----------
    sds : Instance of SpagatDataset
        Refer to SpagatDataset class in dataset.py for more information 

    Returns
        ds_timeseries, ds_1d_vars, ds_2d_vars : Dict
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
        
        if sorted(da.dims) == sorted(('component','Period','TimeStep', 'space')):   #TODO: maybe space should be generalized with additional variable - dimension_description ?
            da = da.transpose('Period','component','space','TimeStep')[0]  #NOTE: eg. (component: 4, Period: 1, TimeStep: 2, space: 3) converted to (component: 4, space: 3, TimeStep: 2) (in coordinates period is still shown without *)
            vars_ts[varname] = da

        elif sorted(da.dims) == sorted(('component','space')):
            vars_1d[varname] = da

        elif sorted(da.dims) == sorted(('component','space','space_2')):
            vars_2d[varname] = da

        else:
            warnings.warn(f'Variable {varname} has dimensions {str(da.dims)} which are not considered for spatial aggregation.')

    #STEP 1. Preprocess Time Series
    ds_timeseries = preprocess_time_series(vars_ts, n_regions, n_components)

    #STEP 2. Preprocess 1d Variables
    ds_1d_vars = preprocess_1d_variables(vars_1d, n_components)

    #STEP 3. Preprocess 2d Variables 
    ds_2d_vars = preprocess_2d_variables(vars_2d, n_components)

    return ds_timeseries, ds_1d_vars, ds_2d_vars
    

def get_custom_distance(ds_ts, ds_1d, ds_2d, 
                n_regions, 
                region_index_x, 
                region_index_y,
                var_category_weights=None,
                var_weights=None):  
    """Custom distance function.

    Parameters
    ----------
    ds_ts, ds_1d, ds_2d : Dict
        Dictionaries obtained as a result of preprocess_dataset()
    n_regions : int
        Total number of regions in the given data 
    region_index_x, region_index_y : int 
        Indicate the two regions between which the custom distance is to be calculated 
        range of these indices - [0, n_regions)
    var_category_weights : None/Dict, optional (default=None)
        A dictionay with weights for different variable categories i.e., ts, 1d and 2d 
        Template: {'ts_vars' : 1, '1d_vars' : 1, '2d_vars' : 1}
        A subset of these can be provided too, rest are considered to be 1 
    var_weights : None/Dict, optional (default=None)
        A dictionay with weights for different variables. For the variables not found 
        in dictionary, a default weight of 1 is assigned. 

    Returns
    -------
    float 
        Custom distance value 
    """

    if var_category_weights is None:
        var_category_weights = {'ts_vars' : 1, '1d_vars' : 1, '2d_vars' : 1}


    if var_weights is None:
        var_list = list(ds_ts.keys()) + list(ds_1d.keys()) + list(ds_2d.keys())
        var_weights = dict.fromkeys(var_list, 1)

    else:
        #INFO: xarray dataset has prefix 1d_,  2d_ and ts_
        # Therefore, in order to match that, the prefix is added here for each variable  
        var_weights = {f"{dimension}_{key}": value      
                                        for key, value in var_weights.items()
                                            for dimension in ["ts", "1d", "2d"]}
    

    #STEP 3. Find distance for each variable category separately 

    #STEP 3a. Distance of Time Series category
    distance_ts = 0
    for var, var_matr in ds_ts.items():

        var_weight = var_weights[var] if var in var_weights.keys() else 1 

        # (i) Extract data corresponding to the variable in both regions  
        region_x_data = var_matr[region_index_x]
        region_y_data = var_matr[region_index_y]
        
        # (ii) Calculate distance 
        #INFO: ts_region_x and ts_region_y are vectors, 
        # subtract the vectors, square each element and add all elements. 
        # (notice subtraction happens per time step, per component)
        distance_ts += sum(np.power((region_x_data - region_y_data),2))  * var_weight  

    #STEP 3b. Distance of 1d Variables category
    distance_1d = 0
    for var, var_matr in ds_1d.items():

        var_weight = var_weights[var] if var in var_weights.keys() else 1 

        # (i) Extract data corresponding to the variable in both regions 
        region_x_data = var_matr[region_index_x]
        region_y_data = var_matr[region_index_y]

        # (ii) Calculate distance
        #INFO: same as previous but subtraction happens per component
        distance_1d += sum(np.power((region_x_data - region_y_data),2))  * var_weight  

    #STEP 3c. Distance of 2d Variables category
    distance_2d = 0

    #STEP 3c (i). Since ds_2d is a condensed matrix, we have to get dist. corresponding to the two given regions
    region_index_x_y = region_index_x * (n_regions - region_index_x) + (region_index_y - region_index_x) -1                
                                                                      
    for var, var_dict in ds_2d.items():

        var_weight = var_weights[var] if var in var_weights.keys() else 1 
        
        #STEP 3c (ii). For each var, component pair...
        for component, data in var_dict.items():
            # Find the corresponding distance value for the given regions 
            value_var_c = data[region_index_x_y]

            if not np.isnan(value_var_c):        #INFO: if the regions are not connected the value will be na
                # Calculate the distance 
                distance_2d += (value_var_c*value_var_c) * var_weight

    #STEP 4. Add all three distances with weights for each variable category
    var_ts_weight = var_category_weights['ts_vars'] if 'ts_vars' in var_category_weights.keys() else 1
    var_1d_weight = var_category_weights['1d_vars'] if '1d_vars' in var_category_weights.keys() else 1
    var_2d_weight = var_category_weights['2d_vars'] if '2d_vars' in var_category_weights.keys() else 1

    return distance_ts * var_ts_weight + distance_1d * var_1d_weight + distance_2d * var_2d_weight

def get_custom_distance_matrix(ds_ts, 
                            ds_1d, 
                            ds_2d, 
                            n_regions,
                            var_category_weights=None,
                            var_weights=None):

    """For every region combination, calculates the custom distance by calling get_custom_distance().
                                                      
    Parameters
    ----------
    ds_ts, ds_1d, ds_2d : Dict
        Dictionaries obtained as a result of preprocess_dataset() 
    n_regions : int
        Total number of regions in the given data 
        range of these indices - [0, n_regions)
    var_category_weights : None/Dict, optional (default=None)
        A dictionay with weights for different variable categories i.e., ts, 1d and 2d 
        Template: {'ts_vars' : 1, '1d_vars' : 1, '2d_vars' : 1}
        A subset of these can be provided too, rest are considered to be 1 
    var_weights : None/Dict, optional (default=None)
        A dictionay with weights for different variables. For the variables not found 
        in dictionary, a default weight of 1 is assigned. 
        
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
                                                var_category_weights,
                                                var_weights)

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


