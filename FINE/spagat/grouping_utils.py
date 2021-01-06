import xarray as xr
import numpy as np
import pandas as pd
import geopandas
import warnings

from sklearn import preprocessing as prep
from scipy.cluster import hierarchy
from sklearn import metrics

def matrix_MinMaxScaler(X, x_min=0, x_max=1):
    ''' Standardize a numpy matrix to range [0,1], NOT column-wise, but matrix-wise!
    '''
    if np.max(X) == np.min(X): 
        return X
    return ((X - np.min(X)) / (np.max(X) - np.min(X))) * (x_max - x_min) + x_min

def preprocessTimeSeries(vars_dict, n_regions, n_components):
    '''Preprocess the input dictionary of time series variables
        - Input vars_dic: dimensions of each variable value are 'component','space','TimeStep'
        - Output ds_ts: a dictionary containing all 2d variables
            - For each variable: the value is a flattened data feature matrix based on its valid components
                - size: Row (n_regions) * Column (n_components * n_timesteps)
            - matrix block for each valid component of one particular variable: 
                - the value is a numpy array of size n_regions * TimeStep         
                - the array matrix is normalized to scale [0,1]     
    '''
     
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
            matrix_var_c = matrix_MinMaxScaler(da[comp_id].values) 

            #STEP 2b. Concatenate the resulting matrix to the final matrix of the corresponding variable
            matrix_var = np.concatenate((matrix_var, matrix_var_c), axis=1)
        
        #STEP 3. Delete the first column of zeros (created initially) and add the matrix it to final dict 
        matrix_var = np.delete(matrix_var,0,1)
        ds_ts[var] = matrix_var
           
    return ds_ts

def preprocess1dVariables(vars_dict, n_components):
    ''' Preprocess 1-dimensional variables
        - return a dictionary containing a numpy matrix for all 1d variables
        - each value is a numpy array of size n_regions * n_valid_components_of_each_variable, 
            e.g. 96*6 for '1d_capacityFix'
        - the numpy arrays are standardized, rescaling to the range [0,1] in column-wise, i.e. rescaling for each component
    '''
    ds_1d = {}

    min_max_scaler = prep.MinMaxScaler()
    
    # For each time series variable, data pair...
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

def preprocess2dVariables(vars_dict, n_components, handle_mode='toDissimilarity'):  
    ''' Preprocess matrices of 2d-vars with one of the following mode:
        - Firstly: Adjust the region order of space_2, i.e. order of columns
        - Obtain ds_2d: a dictionary containing all variables
            - For each variable: the value is a dictionary containing all its valid components
            - For each variable and each valid component: 
                - the value is a numpy array of size n_regions*n_regions
                - the matrix is symmetrical. (undirected graph), with zero diagonal values
                - all the values in the matrices are NON-Negative!

        - Return: a dictionary containing a matrix / vector for each 2d var and each component
            - standardize the vector (rescaling!)
            - Possible TO-DO: add the matrics together?

        How to handle the matrices:
            - handle_mode == 'toDissimilarity': Convert matrices to a distance matrix by transforming the connectivity values to distance meaning
            - handle_mode='toAffinity': extract the matrices of all variables and add them up as one adjacency matrix for spectral clustering
    '''
    
    ds_2d = {}

    # For each time series variable, data pair...
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
            
            var_matr = da[comp_id].values # square matrix (dim: space_1 and space_2)
            
            #STEP 2a: Make sure order of space_1 and space_2 is the same
            da_comp_df = pd.DataFrame(data=var_matr,columns=space2)            
            da_comp_df = da_comp_df[space1]
            
            # Scale the matrix
            ds_2d_var[comp_id] = matrix_MinMaxScaler(da_comp_df.to_numpy())
        
        #STEP 3. Add it to resultant dict
        ds_2d[var] = ds_2d_var

    #STEP 4. Additional steps only if handle_mode is 'toDissimilarity'
    if handle_mode == 'toDissimilarity':
        ''' Convert similarity matrix (original symmetric connectivity matrix) into dissimilarity matrix (1-dim Distance vector)
            - higher values for connectivity means: the two regions are more likely to be grouped -> can be regarded as smaller distance!
            
            - Rescaling the connectivity matrix to range [0,1], where 1 means maximum similarity
            - dissim(x) = 1 - sim(x)
        '''

        # For each var, var_dict pair in the resultant dict created above...
        for var, var_dict in ds_2d.items():
            
            # For each component, matrix pair in var_dict...
            for comp, data in var_dict.items():
                
                #STEP 4a. Obtain the vector form of this symmetric connectivity matrix
                #INFO: [[0.  0.1 0.2]
                #       [0.1 0.  1. ]       -->  [0.1 0.2 1. ]   (only the elements from upper or lower triangle 
                #       [0.2 1.  0. ]]                            as the other is always redundant in a dist matrix )
                vec = hierarchy.distance.squareform(data, checks=False)    

                #STEP 4b. Convert the value of connectivity (similarity) to distance (dissimilarity)
                vec = 1 - vec
                
                # Add it to resultant dict 
                ds_2d[var][comp] = vec
  
    return ds_2d

    
def preprocessDataset(sds, handle_mode, vars='all', dims='all', var_weightings=None): 
    '''Preprocess the Xarray dataset: Separate the dataset into 3 parts: time series data, 1d-var data, and 2d-var data
        - vars_ts: Time series variables: a feature matrix for each ts variable
        - vars_1d: a features matrix for each 1d variable
        - vars_2d: 2d variables showing connectivity between regions
            - Hierarchical: directly transformed as distance matrix (values showing dissimilarity) and combine with vars_ts and vars_1d
            - Spectral: extract this part as an affinity matrix for the (un)directed graph (indicating similarity, here using adjacency matrix)
            - handle_mode: decide which method to preprocess vars_2d

        - Return: the three parts separately 
    '''
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
    ds_timeseries = preprocessTimeSeries(vars_ts, n_regions, n_components)

    #STEP 2. Preprocess 1d Variables
    ds_1d_vars = preprocess1dVariables(vars_1d, n_components)

    #=================== handle_mode == 'toDissimilarity' ===================#  

    if handle_mode == 'toDissimilarity':

        #STEP 3. Preprocess 2d Variables (handle_mode='toDissimilarity')
        ds_2d_vars = preprocess2dVariables(vars_2d, n_components, handle_mode='toDissimilarity')

        return ds_timeseries, ds_1d_vars, ds_2d_vars
    
    #=================== handle_mode == 'toAffinity' ===================#  

    if handle_mode == 'toAffinity':

        #STEP 3. If var_weightings is not specified, set default 
        #INFO: default var_weightings -> {'operationFixRate': 1, '1d_capacity': 1, '2d_distance': 1} 
        if not var_weightings:
            vars_list = list(vars_ts.keys()) + list(vars_1d.keys()) + list(vars_2d.keys())
            var_weightings = dict.fromkeys(vars_list, 1)    

        #STEP 4a. For each Time series variable (in ds_timeseries):
        # convert it's corresponding matrix into weighted matrix and concatenate
        matrix_ts = np.array([np.zeros(n_regions)]).T

        for var, var_matrix in ds_timeseries.items():
            weight = var_weightings[var]
            matrix_ts = np.concatenate((matrix_ts, var_matrix * weight), axis=1)
        
        # Delete the first column of zeros (created initially) 
        matrix_ts = np.delete(matrix_ts, 0, 1)     
        #TODO: if var_weightings is default, this matrix should be the same as ds_timeseries.values()
        # check if it is the same. If yes, reduce these lines (add to previous if statement)

        #STEP 4b. For each 1d varriable:  
        # convert it's corresponding matrix into weighted matrix and concatenate
        matrix_1d = np.array([np.zeros(n_regions)]).T   

        for var, var_matrix in ds_1d_vars.items():
            weight = var_weightings[var]
            matrix_1d = np.concatenate((matrix_1d, var_matrix * weight),axis=1)
        
        # Delete the first column of zeros (created initially)
        matrix_1d = np.delete(matrix_1d,0,1)

        #STEP 4c (i). Preprocess 2d variables (handle_mode='toAffinity')
        ds_2d_vars = preprocess2dVariables(vars_2d, n_components, handle_mode='toAffinity')
 
        matrix_2d = np.zeros((n_regions,n_regions))   
        #STEP 4c. (ii) a. For each 2d varriable... 
        for var, var_dict in ds_2d_vars.items():
            weight = var_weightings[var]

            #STEP 4c. (ii) b. For each corresponding component, data pair:
            # # convert it's corresponding matrix into weighted matrix and concatenate 
            for component, data in var_dict.items():
                matrix_2d += data * weight

        return matrix_ts, matrix_1d, matrix_2d


def selfDistance(ds_ts, ds_1d, ds_2d, 
                n_regions, 
                region_index_x, 
                region_index_y, 
                var_weightings=None, 
                part_weightings=None):  
    ''' Custom distance function: #TODO: change parameters in doc string according to function 
        - parameters a, b: region ids, a < b, a,b in [0, n_regions) 
        - return: distance between a and b = distance_ts + distance_1d + distance_2d
            - distance for time series: ** dist_var_component_timestep ** -> value subtraction
            - distance for 1d vars: sum of (value subtraction in ** dist_var_component ** )
            - distance for 2d vars: corresponding value in ** dist_var_component **
            - each partial distance need to be divided by sum of variable weight factors, 
                i.e. number of variables when all weight factors are 1, 
                to reduce the effect of variables numbers on the final distance.
        ---
        Metric space properties :  -> at the same level of data structure! in the same distance space!
        - Non-negativity: d(i,j) > 0
        - Identity of indiscernibles: d(i,i) = 0   ---> diagonal must be 0!
        - Symmetry: d(i,j) = d(j,i)  ---> the part_2 must be Symmetrical!
        - Triangle inequality: d(i,j) <= d(i,k) + d(k,j)  ---> NOT SATISFIED!!!
    '''

    #STEP 1. If weights for variables are not passed (via var_weightings) then assign default 1 to each 
    if not var_weightings:                                    
        vars_list = list(ds_ts.keys()) + list(ds_1d.keys()) + list(ds_2d.keys())
        var_weightings = dict.fromkeys(vars_list,1)

    #STEP 2. If weights for the 3 variable categories are not passed (via part_weightings) then assign default 1 to each
    if not part_weightings:                             
        part_weightings = [1,1,1]

    #STEP 3. Find distance for each variable category separately 

    #STEP 3a. Distance of Time Series category
    distance_ts = 0
    for var, var_matr in ds_ts.items():
        var_weight_factor = var_weightings[var]

        # (i) Extract data corresponding to the variable in both regions  
        region_x_data = var_matr[region_index_x]
        region_y_data = var_matr[region_index_y]
        
        # (ii) Calculate distance 
        #INFO: ts_region_x and ts_region_y are vectors, 
        # subtract the vectors, square each element and add all elements. 
        # (notice subtraction happens per time step, per component)
        distance_ts += sum(np.power((region_x_data - region_y_data),2)) * var_weight_factor    

    #STEP 3b. Distance of 1d Variables category
    distance_1d = 0
    for var, var_matr in ds_1d.items():

        var_weight_factor = var_weightings[var]

        # (i) Extract data corresponding to the variable in both regions 
        region_x_data = var_matr[region_index_x]
        region_y_data = var_matr[region_index_y]

        # (ii) Calculate distance
        #INFO: same as previous but subtraction happens per component
        distance_1d += sum(np.power((region_x_data - region_y_data),2)) * var_weight_factor  

    #STEP 3c. Distance of 2d Variables category
    distance_2d = 0

    #STEP 3c (i). Since ds_2d is a condensed matrix, we have to get dist. corresponding to the two given regions
    region_index_x_y = region_index_x * (n_regions - region_index_x) + (region_index_y - region_index_x) -1                
                                                                      
    for var, var_dict in ds_2d.items():
        var_weight_factor = var_weightings[var]
        
        #STEP 3c (ii). For each var, component pair...
        for component, data in var_dict.items():
            # Find the corresponding distance value for the given regions 
            value_var_c = data[region_index_x_y]

            if not np.isnan(value_var_c):        #INFO: if the regions are not connected the value will be na
                # Calculate the distance 
                distance_2d += (value_var_c*value_var_c) * var_weight_factor

    #STEP 4. Add all three distances with part_weightings of each category
    return distance_ts * part_weightings[0] + distance_1d * part_weightings[1] + distance_2d * part_weightings[2]

def selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions, var_weightings=None):
    ''' Return a n_regions by n_regions symmetric distance matrix X #TODO: better description => for every region combination, calculates the custom distance (by calling selfDistance())
                                                                                                 Returns a n_regions by n_regions symmetric distance matrix 
    '''

    distMatrix = np.zeros((n_regions,n_regions))

    #STEP 1. For every region pair, calculate the distance 
    for i in range(n_regions):
        for j in range(i+1,n_regions):
            distMatrix[i,j] = selfDistance(ds_ts,ds_1d, ds_2d, n_regions, i,j, var_weightings=var_weightings)

    #STEP 2. Only upper triangle has values, reflect these values in lower triangle to make it a hollow, symmetric matrix
    distMatrix += distMatrix.T - np.diag(distMatrix.diagonal())  

    return distMatrix


def generateConnectivityMatrix(sds):
    ''' Generate an adjacency matrix to show the neighboring structure
        - For every index pair of regions, as long as they have a non-zero 2d-variable value, related to pipeline component, they are regarded as connected.
        - 1 means connected, otherwise 0
        - If no component related to pipeline, then consider all other components
    '''
    ds_extracted = sds.xr_dataset

    vars_2d = {}

    for varname, da in ds_extracted.data_vars.items():
        if da.dims == ('component','space','space_2'):
            vars_2d[varname] = da

    n_regions = len(ds_extracted['space'].values)
    component_list = list(ds_extracted['component'].values)
    n_components = len(component_list)

    #STEP 1. Preprocess 2d variables #TODO: why is this required again? isn't this done initially ?
    ds_2d = preprocess2dVariables(vars_2d, n_components, handle_mode='toAffinity')

    #STEP 2. First check if a component called 'pipeline' exists.
    # If it does, take it as connect_components
    connect_components = []
    for i in range(n_components):            
        if 'pipeline' in component_list[i].lower():
            connect_components.append(i)

    #STEP 3. If 'pipeline' does not exist, consider all existing components.
    if not connect_components:
        connect_components = list(range(n_components))  

    adjacencyMatrix = np.zeros((n_regions,n_regions))
    # For each region pair checkConnectivity (call the function)
    for i in range(n_regions):
        for j in range(i+1,n_regions):
            if checkConnectivity(i,j, ds_2d, connect_components):  #INFO: checkConnectivity returns true or false 
                adjacencyMatrix[i,j] = 1

    #STEP 4. adjacencyMatrix is upper triangular. 
    # so, take transpose, subtract the diagonal elements and add it back to adjacencyMatrix. 
    # Now, it is symmetrical
    adjacencyMatrix += adjacencyMatrix.T - np.diag(adjacencyMatrix.diagonal())  

    #STEP 5. Set the diagonal values to 1
    np.fill_diagonal(adjacencyMatrix, 1)  

    return adjacencyMatrix

def checkConnectivity(region_index_x, 
                    region_index_y,
                    ds_2d, 
                    connect_components):   
    '''Check if region i is neighboring to region j, based on the components related to pipelines.
        - as 1 if there exists at least one non-zero value in any matrix at the position [i,j]
        - if no components related to pipelines, then the connect_components is the list of all existing components.
    '''
    
    for var_dict in ds_2d.values():
        for c, data in var_dict.items():
            if (c in connect_components) and (data[region_index_x, region_index_y] != 0):  
                return True                              #INFO: checks for each var, each component. 
                                                         #      Returns TRUE if atleast one var, component pair has non-zero value.
                                                         # 
                                                         #
    return False                                         # After checking for all, returns false if none had non-zero value

def computeModularity(adjacency, regions_label_list):
    ''' Compute the modularity of the partitioned graph
        - graph's weighted adjacency matrix with entries defined by the edge weights
        - regions_label_list to show the graph partition
    '''
    #INFO: bounds -> [-0.5; 1). Higher the score, the better the partition 
    np.fill_diagonal(adjacency, 0)
    n_regions = len(regions_label_list)

    #STEP 1. Take values in the adjacency matrix as edge weights
    edge_weights_sum = np.sum(adjacency)

    modularity = 0

    #STEP 2. For each region pair...
    for v in range(n_regions):
        for w in range(v+1, n_regions):

            #STEP 2a. Obtain the weighted degree of nodes: sum of node's incident edge weights
            d_v = np.sum(adjacency[v])
            d_w = np.sum(adjacency[w])

            #STEP 2b. Set delta -> If both nodes belong to the same cluster 1, else 0
            delta = 1 if regions_label_list[v] == regions_label_list[w] else 0

            #STEP 2c. Calculate modularity
            modularity += (adjacency[v,w] - (d_v * d_w) / (2 * edge_weights_sum)) * delta

    modularity = modularity / (2 * edge_weights_sum)

    return modularity

def computeSilhouetteCoefficient(regions_list, distanceMatrix, aggregation_dict):
    '''
    Obtain Silhouette score for all intermediate levels in the hierarchy  #TODO: make this better 
    '''
    #INFO: score is bounded between -1 and +1. The higher the score, the better the clustering.

    n_regions = len(regions_list)
    scores = [0 for i in range(1, n_regions-1)]   #INFO: Silhouette Coefficient can only be computed for 2 to n_samples - 1 (inclusive)
                                                  #       first and last level in the hierarchy (first is original and last is one region) are to be eliminated
    labels = [0 for i in range(n_regions)]    

    for k, regions_dict in aggregation_dict.items():

        #STEP 1. Check if k is an intermediate level in the hierarchy
        if k == 1 or k == n_regions:
            continue                         #INFO: again, eliminate computing for first and last level 

        #STEP 2. Assign labels to each group of regions (starting from 0)
        label = 0
        for sub_regions_list in regions_dict.values():  
            for region in sub_regions_list:
                ind = regions_list.index(region)
                labels[ind] = label
            
            label += 1
        
        #STEP 3. Get Silhouette Coefficient for current level in the hierarchy 
        s = metrics.silhouette_score(distanceMatrix, labels, metric='precomputed')
        scores[k-2] = s

    return scores


