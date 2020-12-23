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

    # Each variable has a matrix value
    for var, da in vars_dict.items():

        matrix_var = np.array([np.zeros(n_regions)]).T

        # Find the valid components for each variable: valid_component_weight=1, otherwise=0
        var_mean_df = da.mean(dim="space").mean(dim="TimeStep").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])

        for comp_id in valid_component_ids:
            # Compute the standardized matrix for each valid component: rescale the matrix value to range [0,1]
            # -> the values in time series for this component should be in the same scaling: matrix_MinMaxScaler()
            matrix_var_c = matrix_MinMaxScaler(da[comp_id].values) 

            # Concatenate this matrix block of one component to the final matrix for this 2d variable
            matrix_var = np.concatenate((matrix_var, matrix_var_c), axis=1)

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

    for var, da in vars_dict.items():
        
        # Find the valid components for each variable: valid_comp_weight=1, otherwise=0
        var_mean_df = da.mean(dim="space").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])
        

        # Retain only the valid components
        data = da.values[valid_component_ids]
        ds_1d[var] = min_max_scaler.fit_transform(data.T)

    return ds_1d

def preprocess2dVariables(vars_dict, component_list, handle_mode='toDissimilarity'):  #TODO: component_list is not required. take n_components as argument instead 
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

    n_components = len(component_list)   

    # Obtain th dictionary of connectivity matrices for each variable and for its valid component, each of size n_regions*n_regions (n_regions)
    ds_2d = {}

    for var, da in vars_dict.items():
        
        ds_2d_var = {}
        
        # Different region orders
        space1 = da.space.values
        space2 = da.space_2.values
        
        # Find the valid components for each variable
        var_mean_df = da.mean(dim="space").mean(dim="space_2").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])
        
        # DataArray da for this 2d-variable, component * space * space_2
        for comp_id in valid_component_ids:
            
            # Under each component, there is a (e.g. 96*96) squares matrix to represent the connection features
            var_matr = da[comp_id].values
            
            # Rearrange the columns order --> the regions order of space_2!
            da_comp_df = pd.DataFrame(data=var_matr,columns=space2)            #INFO: This is to make sure order of space and space2 is the same
            da_comp_df = da_comp_df[space1]
            
            # Standardize the matrix: keep all the values non-negative! AND keep zeros to be zeros (not change the meaning of connectivity!)
            # => scale the data to the range [0,1]
            ds_2d_var[comp_id] = matrix_MinMaxScaler(da_comp_df.to_numpy())
        
        ds_2d[var] = ds_2d_var

    # Handle the matrices according to clustering methods

    ## Possible TO-DO: maybe other transformation methods are possible, e.g. Gaussian (RBF, heat) kernel
    if handle_mode == 'toDissimilarity':
        ''' Convert similarity matrix (original symmetric connectivity matrix) into dissimilarity matrix (1-dim Distance vector)
            - higher values for connectivity means: the two regions are more likely to be grouped -> can be regarded as smaller distance!
            
            - Rescaling the connectivity matrix to range [0,1], where 1 means maximum similarity
            - dissim(x) = 1 - sim(x)
        '''

        # Obtain a dictionary containing one distance vector (1-dim matrix) for each variable and for each valid component
        for var, var_dict in ds_2d.items():
            
            # Transform the symmetric connectivity matrix to 1-dim distance vector
            for c, data in var_dict.items():
                
                # Obtain the vector form of this symmetric connectivity matrix, in the range [0,1]
                # Deactivate checks since small numerical errors can be in the dataset
                vec = hierarchy.distance.squareform(data, checks=False)    #INFO: [[0.  0.1 0.2]
                                                                           #       [0.1 0.  1. ]       -->  [0.1 0.2 1. ]   (only the elements from upper or lower triangle 
                                                                           #       [0.2 1.  0. ]]                            as the other is always redundant in a dist matrix )

                # Convert the value of connectivity (similarity) to distance (dissimilarity)
                vec = 1 - vec
                
                # Distance vector for this 2d variable and this component: 1 means maximum distance!
                ds_2d[var][c] = vec
  
    return ds_2d

    
       

def preprocessDataset(sds, handle_mode, vars='all', dims='all', var_weightings=None):   #TODO: take a pass at the function and refactor it
    '''Preprocess the Xarray dataset: Separate the dataset into 3 parts: time series data, 1d-var data, and 2d-var data
        - vars_ts: Time series variables: a feature matrix for each ts variable
        - vars_1d: a features matrix for each 1d variable
        - vars_2d: 2d variables showing connectivity between regions
            - Hierarchical: directly transformed as distance matrix (values showing dissimilarity) and combine with vars_ts and vars_1d
            - Spectral: extract this part as an affinity matrix for the (un)directed graph (indicating similarity, here using adjacency matrix)
            - handle_mode: decide which method to preprocess vars_2d

        - Return: the three parts separately 
    '''

    dataset = sds.xr_dataset

    #STEP 1. Traverse all variables in the dataset, and put them in separate categories
    #NOTE: vars_ts, vars_1d, vars_2d -> dicts of variables and their corresponding dataArrays
    vars_ts = {}
    vars_1d = {}
    vars_2d = {}

    for varname, da in dataset.data_vars.items():
        # sort the dimensions
        if sorted(da.dims) == sorted(('component','Period','TimeStep', 'space')):   #TODO: maybe space should be generalized with additional variable - dimension_description ?
            # Period is not considered -> TODO: consider the Period dimension.
            da = da.transpose('Period','component','space','TimeStep')[0]  #NOTE: eg. (component: 4, Period: 1, TimeStep: 2, space: 3) converted to (component: 4, space: 3, TimeStep: 2) (in coordinates period is still shown without *)
            vars_ts[varname] = da

        elif sorted(da.dims) == sorted(('component','space')):
            vars_1d[varname] = da

        elif sorted(da.dims) == sorted(('component','space','space_2')):
            vars_2d[varname] = da

        else:
            warnings.warn(f'Variable {varname} has dimensions {str(da.dims)} which are not considered for spatial aggregation.')

    component_list = list(dataset['component'].values)
    n_regions = len(dataset['space'].values)

    #STEP 2. Preprocess Time Series
    ds_timeseries = preprocessTimeSeries(vars_ts, n_regions, len(component_list))

    #STEP 3. Preprocess 1d Variables
    ds_1d_vars = preprocess1dVariables(vars_1d, len(component_list))
    
    #STEP 4. Varies based on handle_mode

    #STEP 4a. if handle_mode == 'toDissimilarity', call preprocess2dVariables() directly
    if handle_mode == 'toDissimilarity':
        ds_2d_vars = preprocess2dVariables(vars_2d, component_list, handle_mode='toDissimilarity')

        return ds_timeseries, ds_1d_vars, ds_2d_vars

    #TODO: try negative var_weights for some 2d vars 
    
    #STEP 4b. if handle_mode == 'toAffinity' -> convert matrix in weighted matrix based on var_weightings
    if handle_mode == 'toAffinity':
        # Weighting factors of each variable 
        if not var_weightings:
            vars_list = list(vars_ts.keys()) + list(vars_1d.keys()) + list(vars_2d.keys())
            var_weightings = dict.fromkeys(vars_list,1)     #NOTE: For now var_weightings is always {'operationFixRate': 1, '1d_capacity': 1, '2d_distance': 1} (There is no option for user to change) ??

        #STEP 4b. (i)  For each Time series varriable:  
        # convert it's corresponding matrix into weighted matrix (weights are from var_weightings)
        matrix_ts = np.array([np.zeros(n_regions)]).T

        for var, var_matrix in ds_timeseries.items():

            weight = var_weightings[var]
            
            # Concatenate the matrix of this var to the final matrix with its weighting factor
            matrix_ts = np.concatenate((matrix_ts, var_matrix * weight), axis=1)
        
        matrix_ts = np.delete(matrix_ts,0,1)     #NOTE: if var_weightings is default, this matrix should be the same as ds_timeseries.values()
                                                 #TODO: check if it is the same. If yes, reduce these lines (add to previous if statement)

        #STEP 4b. (ii)  For each 1d varriable:  convert it's corresponding matrix into weighted matrix (weights are from var_weightings)
        matrix_1d = np.array([np.zeros(n_regions)]).T   

        for var, var_matrix in ds_1d_vars.items():
            weight = var_weightings[var]
            # Concatenate the matrix of this vars to one single 1d matrix with weight factor
            matrix_1d = np.concatenate((matrix_1d, var_matrix * weight),axis=1)
        
        matrix_1d = np.delete(matrix_1d,0,1)

        #STEP 4b. (iii)  a. Preprocess 2d variables
        ds_2d_vars = preprocess2dVariables(vars_2d, component_list, handle_mode='toAffinity')

        #STEP 4b. (iii)  b. For each variable, convert the matrix corresponding to each component 
        # to weighted matrix and Add each components weighted matrices to obtain one single weighted affinity matrix.
        # Add each components weighted matrices to obtain one single weighted affinity matrix
        matrix_2d = np.zeros((n_regions,n_regions))   

        for var, var_dict in ds_2d_vars.items():
            weight = var_weightings[var]

            for component, data in var_dict.items():
                matrix_2d += data * weight

        
        return matrix_ts, matrix_1d, matrix_2d


def selfDistance(ds_ts, ds_1d, ds_2d, n_regions, a, b, var_weightings=None, part_weightings=None):  #TODO: Change a and b to something more intuitive (eg. region_n and region_m). notice that both are ints !
    ''' Custom distance function: 
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

    # Weighting factors of each variable 
    if not var_weightings:                                    
        vars_list = list(ds_ts.keys()) + list(ds_1d.keys()) + list(ds_2d.keys())
        var_weightings = dict.fromkeys(vars_list,1)

    # Weighting factors for 3 var-categories
    if not part_weightings:                             
        part_weightings = [1,1,1]

    # Distance of Time Series Part
    distance_ts = 0
    for var, var_matr in ds_ts.items():

        var_weight_factor = var_weightings[var]

        # Vectors for the two data points (regions), each feature refers to [one valid component & one timestep] for this var
        reg_a = var_matr[a]
        reg_b = var_matr[b]

        # dist_ts(a,b) = sum_var( var_weight * dist_var(a,b) )
        # dist_var(a,b) = sum_c(sum_t( [value_var_c_t(a) - value_var_c_t(b)]^2 ))
        distance_ts += sum(np.power((reg_a - reg_b),2)) * var_weight_factor    #INFO: reg_a and reg_b are vectors, subtract the vectors, square each element and add all elements. 
                                                                               #       (notice subtraction happens per time step, per component)

    # Distance of 1d Variables Part
    distance_1d = 0
    for var, var_matr in ds_1d.items():

        var_weight_factor = var_weightings[var]

        # Vectors for the two data points (regions), each feature refers to one valid component for this var
        reg_a = var_matr[a]
        reg_b = var_matr[b]

        # dist_1d(a,b) = sum_var{var_weight * sum_c( [value_var_c(a) - value_var_c(b)]^2 ) }
        distance_1d += sum(np.power((reg_a - reg_b),2)) * var_weight_factor  #INFO: same as previous but subtraction happens per component

    # Distance of 2d Variables Part
    distance_2d = 0

    # The index of corresponding value for region[a] and region[b] in the distance vectors
    index_regA_regB = a * (n_regions - a) + (b - a) -1                #INFO: since it's a condensed matrix, we have to get dist. corresponding to the two given regions
                                                                      
    for var, var_dict in ds_2d.items():

        var_weight_factor = var_weightings[var]

        for component, data in var_dict.items():
            # Find the corresponding distance value for region_a and region_b 
            value_var_c = data[index_regA_regB]

            if not np.isnan(value_var_c):                              #INFO: if the regions are not connected the value will be na
                # dist_2d(a,b) = sum_var{var_weight * sum_c( [value_var_c(a,b)]^2 ) }
                distance_2d += (value_var_c*value_var_c) * var_weight_factor

    return distance_ts * part_weightings[0] + distance_1d * part_weightings[1] + distance_2d * part_weightings[2]

def selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions, var_weightings=None):
    ''' Return a n_regions by n_regions symmetric distance matrix X #TODO: better description => for every region combination, calculates the custom distance (by calling selfDistance())
                                                                                                 Returns a n_regions by n_regions symmetric distance matrix 
    '''

    distMatrix = np.zeros((n_regions,n_regions))

    for i in range(n_regions):
        for j in range(i+1,n_regions):
            distMatrix[i,j] = selfDistance(ds_ts,ds_1d, ds_2d, n_regions, i,j, var_weightings=var_weightings)

    distMatrix += distMatrix.T - np.diag(distMatrix.diagonal())  #INFO: only upper triangle has values, reflects these values in lower triangle to make it a hollow, symmetric matrix

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

    # Square matrices for each 2d variable and each valid component
    ds_2d = preprocess2dVariables(vars_2d, component_list, handle_mode='toAffinity')

    # The neighboring information is based on the 2d vars with components related to pipeline
    connect_components = []
    for i in range(n_components):            
        if 'pipeline' in component_list[i].lower():
            connect_components.append(i)

    # If there is no components related to pipelines, then consider all existing components.
    if not connect_components:
        connect_components = list(range(n_components))  

    adjacencyMatrix = np.zeros((n_regions,n_regions))

    # Check each index pair of regions to verify, if the two regions are connected to each other
    for i in range(n_regions):
        for j in range(i+1,n_regions):
            if checkConnectivity(i,j, ds_2d, connect_components):  #NOTE: checkConnectivity returns true or false 
                adjacencyMatrix[i,j] = 1

    adjacencyMatrix += adjacencyMatrix.T - np.diag(adjacencyMatrix.diagonal())  #NOTE: adjacencyMatrix is upper triangular. so, take transpose, subtract the diagonal elements and add it back to adjacencyMatrix. Now, it is symmetrical

    # Set the diagonal values as 1
    np.fill_diagonal(adjacencyMatrix, 1)  

    return adjacencyMatrix

def checkConnectivity(i,j, ds_2d, connect_components):   #TODO: change i, j to something more meaningful 
    '''Check if region i is neighboring to region j, based on the components related to pipelines.
        - as 1 if there exists at least one non-zero value in any matrix at the position [i,j]
        - if no components related to pipelines, then the connect_components is the list of all existing components.
    '''
    
    for var, var_dict in ds_2d.items():
        for c, data in var_dict.items():
            if (c in connect_components) and (data[i,j] != 0):  
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

    # Values in the adjacency matrix as edge weights
    edge_weights_sum = np.sum(adjacency)

    modularity = 0

    for v in range(n_regions):
        for w in range(v+1, n_regions):

            # The weighted degree of nodes: sum of node's incident edge weights
            d_v = np.sum(adjacency[v])
            d_w = np.sum(adjacency[w])

            # If the two nodes belong to the same cluster
            delta = 1 if regions_label_list[v] == regions_label_list[w] else 0

            # Sum up the actual fraction of the edges minus the expected fraction of edges inside of each cluster
            modularity += (adjacency[v,w] - (d_v * d_w) / (2 * edge_weights_sum)) * delta

    modularity = modularity / (2 * edge_weights_sum)

    return modularity

def computeSilhouetteCoefficient(regions_list, distanceMatrix, aggregation_dict):
    '''
    Obtain Silhouette score for all intermediate levels in the hierarchy  #TODO: make this better 
    '''
    #INFO: score is bounded between -1 and +1. The higher the score, the better the clustering.

    n_regions = len(regions_list)
    scores = [0 for i in range(1, n_regions-1)]   #NOTE: Silhouette Coefficient can only be computed for 2 to n_samples - 1 (inclusive)
                                                  #       first and last level in the hierarchy (first is original and last is one region) are to be eliminated
    labels = [0 for i in range(n_regions)]    

    for k, regions_dict in aggregation_dict.items():

        #STEP 1. Check if k is an intermediate level in the hierarchy
        if k == 1 or k == n_regions:
            continue                         #NOTE: again, eliminate computing for first and last level 

        #STEP 2. Assign labels to each group of regions (starting from 0)
        label = 0
        for sup_region in regions_dict.values():  #TODO: change sup_region to sub_regions_list 
            for reg in sup_region:
                ind = regions_list.index(reg)
                labels[ind] = label
            
            label += 1
        
        #STEP 3. Get Silhouette Coefficient for current level in the hierarchy 
        s = metrics.silhouette_score(distanceMatrix, labels, metric='precomputed')
        scores[k-2] = s

    return scores


