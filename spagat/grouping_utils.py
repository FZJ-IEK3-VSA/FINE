import xarray as xr
import numpy as np
import pandas as pd
import geopandas

from sklearn import preprocessing as prep
from scipy.cluster import hierarchy

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
            - For each variable: the value is a flattened data feature matrix based its valid components
                - size: Row (n_regions) * Column (n_components * n_timesteps)
            - matrix block for each valid component of one particular variable: 
                - the value is a numpy array of size n_regions * TimeStep         
                - the array matrix is normalized to scale [0,1]     
    '''
    if not vars_dict: return None

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

    if not vars_dict: return None

    ds_1d = {}

    min_max_scaler = prep.MinMaxScaler()

    for var, da in vars_dict.items():
        
        # Find the valid components for each variable: valid_comp_weight=1, otherwise=0
        var_mean_df = da.mean(dim="space").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])
        

        # Only remain the valid components
        data = da.values[valid_component_ids]
        ds_1d[var] = min_max_scaler.fit_transform(data.T)

    return ds_1d

def preprocess2dVariables(vars_dict, component_list, handle_mode='toDissimilarity'):
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

    if not vars_dict: return None

    n_components = len(component_list)

    # Obtain th dictionary of connectivity matrices for each variable and for its valid component, each of size 96*96 (n_regions)
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
            da_comp_df = pd.DataFrame(data=var_matr,columns=space2)
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
 
        min_max_scaler = prep.MinMaxScaler()

        for var, var_dict in ds_2d.items():
            
            # Transform the symmetric connectivity matrix to 1-dim distance vector
            for c, data in var_dict.items():
                
                # Obtain the vector form of this symmetric connectivity matrix, in the range [0,1]
                vec = hierarchy.distance.squareform(data)

                # Convert the value of connectivity (similarity) to distance (dissimilarity)
                vec = 1 - vec
                
                # Distance vector for this 2d variable and this component: 1 means maximum distance!
                ds_2d[var][c] = vec
  
        return ds_2d

    if handle_mode == 'toAffinity':
        '''Original matrices as Adjacency matrices : 
            - adjacency matrix: 0 means identical elements; high values means very similar elements
            - adjacency matrix of a graph: symmetric, diagonals = 0
            - add all matrices of different components for each variable 
            
        '''
        return ds_2d

def preprocessDataset(sds,handle_mode, vars='all',dims='all', var_weightings=None):
    '''Preprocess the Xarray dataset: Separate the dataset into 3 parts: time series data, 1d-var data, and 2d-var data
        - vars_ts: Time series variables: a feature matrix for each ts variable
        - vars_1d: a features matrix for each 1d variable
        - vars_2d: 2d variables showing connectivity between regions
            - Hierarchical: directly transformed as distance matrix (values showing dissimilarity) and combine with vars_ts and vats_1d
            - Spectral: extract this part as an affinity matrix for the (un)directed graph (indicating similarity, here using adjacency matrix)
            - handle_mode: decide which method to preprocess vars_2d

        - Return: the three parts separately 
    '''

    dataset = sds.xr_dataset

    # Traverse all variables in the dataset, and put them in separate categories
    vars_ts = {}
    vars_1d = {}
    vars_2d = {}

    for varname, da in dataset.data_vars.items():
        if da.dims == ('component','Period','TimeStep', 'space'):
            # Do not have to consider the Period --> ToDo: consider the Period dimension.
            da = da.transpose('Period','component','space','TimeStep')[0]  
            vars_ts[varname] = da

        if da.dims == ('component','space'):
            vars_1d[varname] = da

        if da.dims == ('component','space','space_2'):
            vars_2d[varname] = da

    component_list = list(dataset['component'].values)

    n_regions = len(dataset['space'].values)

    ds_timeseries = preprocessTimeSeries(vars_ts, n_regions, len(component_list))
    ds_1d_vars = preprocess1dVariables(vars_1d, len(component_list))
    
    if handle_mode == 'toDissimilarity':
        ''' Return 3 (all standardized with minMaxScaler) dictionaries for each variable category:
            - timeseries vars: a dictionary containing one feature matrix for each variable
            - 1d vars: a dictionary containing one feature matrix for each variable
            - 2d vars: vectors for each variable and for each valid component  
                - 1-dim vector indicating dissimilarities between two regions (distance) 
                - vector length = n_regs * (n_regs - 1) / 2
        '''
        ds_2d_vars = preprocess2dVariables(vars_2d, component_list, handle_mode='toDissimilarity')

        return ds_timeseries, ds_1d_vars, ds_2d_vars

    ############### TO-DO: try negative var_weights for some 2d vars #############################
    if handle_mode == 'toAffinity':
        ''' Return 3 affinity matrices:
            - timeseries: one data feature matrix
                - concatenate matrices for different variables with weighting factors
                - return one matrix of size: n_regions * columns=(n_ts_vars * n_valid_component_per_var * n_timesteps)
            - 1d vars: one data feature matrix
                - concatenate matrices of various vars to one matrix with weighting factor for each var
                - return the matrix of size: n_regions * columns =(n_1d_vars * n_valid_component_per_var) 
            - 2d vars: one single adjacency matrix
                - original matrices as adjacency matrices
                - add them to one single matrix with weighting factors for each var
                - from adjacency matrix to affinity matrix
        '''
        # Weighting factors of each variable 
        if var_weightings:
            var_weightings = var_weightings
        else:
            vars_list = list(vars_ts.keys()) + list(vars_1d.keys()) + list(vars_2d.keys())
            var_weightings = dict.fromkeys(vars_list,1)

        ###### For Time series vars: obtain the single matrix - matrix_ts
        matrix_ts = np.array([np.zeros(n_regions)]).T

        n_timesteps = len(dataset['TimeStep'].values)

        for var, var_matrix in ds_timeseries.items():

            weight = var_weightings[var]
            
            # Concatenate the matrix of this var to the final matrix with its weighting factor
            matrix_ts = np.concatenate((matrix_ts, var_matrix * weight), axis=1)
        
        matrix_ts = np.delete(matrix_ts,0,1)

        ###### For 1d vars: obtain the single matrix - matrix_1d
        matrix_1d = np.array([np.zeros(n_regions)]).T

        for var, var_matrix in ds_1d_vars.items():

            weight = var_weightings[var]

            # Concatenate the matrix of this vars to one single 1d matrix with weight factor
            matrix_1d = np.concatenate((matrix_1d, var_matrix * weight),axis=1)
        
        matrix_1d = np.delete(matrix_1d,0,1)

        ###### For 2d vars: obtain a single square matrix of size n_regions*regions
        matrix_2d = np.zeros((n_regions,n_regions))

        ds_2d_vars = preprocess2dVariables(vars_2d, component_list, handle_mode='toAffinity')

        # After adding, the value in matrix_2d is not in the range [0,1] any more
        for var, var_dict in ds_2d_vars.items():

            weight = var_weightings[var]

            # Add the matrices of different components for one var to a single matrix
            for component, data in var_dict.items():
                matrix_2d += data * weight

        ###### Return 3 separate matrices
        return matrix_ts, matrix_1d, matrix_2d


def selfDistance(ds_ts, ds_1d, ds_2d, n_regions, a, b, var_weightings=None, part_weightings=None):
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
    if var_weightings:
        var_weightings = var_weightings
    else:
        vars_list = list(ds_ts.keys()) + list(ds_1d.keys()) + list(ds_2d.keys())
        var_weightings = dict.fromkeys(vars_list,1)

    # Weighting factors for 3 var-categories
    if part_weightings:
        part_weightings = part_weightings
    else:
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
        distance_ts += sum( np.power((reg_a - reg_b),2) ) * var_weight_factor

    # Distance of 1d Variables Part
    distance_1d = 0
    for var, var_matr in ds_1d.items():

        var_weight_factor = var_weightings[var]

        # Vectors for the two data points (regions), each feature refers to one valid component for this var
        reg_a = var_matr[a]
        reg_b = var_matr[b]

        # dist_1d(a,b) = sum_var{var_weight * sum_c( [value_var_c(a) - value_var_c(b)]^2 ) }
        distance_1d += sum(np.power((reg_a - reg_b),2)) * var_weight_factor

    # Distance of 2d Variables Part
    distance_2d = 0

    # The index of corresponding value for region[a] and region[b] in the distance vectors
    index_regA_regB = a * (n_regions - a) + (b - a) -1
    
    for var, var_dict in ds_2d.items():

        var_weight_factor = var_weightings[var]

        for component, data in var_dict.items():
            # Find the corresponding distance value for region_a and region_b
            value_var_c = data[index_regA_regB]

            # dist_2d(a,b) = sum_var{var_weight * sum_c( [value_var_c(a,b)]^2 ) }
            distance_2d += (value_var_c*value_var_c) * var_weight_factor


    return distance_ts * part_weightings[0] + distance_1d * part_weightings[1] + distance_2d * part_weightings[2]

def selfDistanceMatrix(ds_ts, ds_1d, ds_2d, n_regions, var_weightings=None):
    ''' Return a n_regions by n_regions symmetric distance matrix X 
    '''

    distMatrix = np.zeros((n_regions,n_regions))

    for i in range(n_regions):
        for j in range(i+1,n_regions):
            distMatrix[i,j] = selfDistance(ds_ts,ds_1d, ds_2d, n_regions, i,j, var_weightings=var_weightings)

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

    # Square matrices for each 2d variable and each valid component
    ds_2d = preprocess2dVariables(vars_2d, component_list, handle_mode='toAffinity')

    # The neighboring information is based on the 2d vars with components related to pipeline
    connect_components = []
    for i in range(len(component_list)):
        if 'pipeline' in component_list[i].lower():
            connect_components.append(i)

    # If there is no components related to pipelines, then consider all existing components.
    if not connect_components:
        connect_components = list(range(len(component_list)))

    adjacencyMatrix = np.zeros((n_regions,n_regions))

    # Check each index pair of regions to verify, if the two regions are connected to each other
    for i in range(n_regions):
        for j in range(i+1,n_regions):
            if checkConnectivity(i,j, ds_2d, connect_components):
                adjacencyMatrix[i,j] = 1

    adjacencyMatrix += adjacencyMatrix.T - np.diag(adjacencyMatrix.diagonal())

    # Set the diagonal values as 1
    np.fill_diagonal(adjacencyMatrix, 1)

    return adjacencyMatrix

def checkConnectivity(i,j, ds_2d, connect_components):
    '''Check if region i is neighboring to region j, based on the components related to pipelines.
        - as 1 if there exists at least one non-zero value in any matrix at the position [i,j]
        - if no components related to pipelines, then the connect_components is the list of all existing components.
    '''
    
    for var, var_dict in ds_2d.items():
        for c, data in var_dict.items():
            if (c in connect_components) and (data[i,j] != 0):
                return True
            
    return False
