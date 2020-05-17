import xarray as xr
import numpy as np

from sklearn import preprocessing as prep

def preprocessTimeSeries(vars_dict, n_regions, n_components, weightings=None):
    '''Preprocess the input dictionary of time series variables
        - Input vars_dic: dimensions of each variable value are 'component','space','TimeStep'
        - weightings: weight factors of each time series variable
        - Output ds_ts: a dictionary containing all variables
            - For each variable: the value is a dictionary containing all its valid components
            - For each variable and each valid component: 
                - the value is a numpy array of size n_regions*TimeStep         
                - the array is normalized and weighted by the variable weight factors      
    '''
    if not vars_dict: return None

    if weightings:
        weightings = weightings
    else:
        vars_list = list(vars_dict)
        weightings = dict.fromkeys(vars_list,1)

    ds_ts = {}

    for var, da in vars_dict.items():

        ds_ts_var = {}

        # Find the valid components for each variable
        var_mean_df = da.mean(dim="space").mean(dim="TimeStep").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])

        # Concatenate the normalized and weighted matrix for each valid component
        weight_factor = weightings[var]
        for comp_id in valid_component_ids:
            ds_ts_var[comp_id] = prep.scale(da[comp_id].values) * weight_factor

        ds_ts[var] = ds_ts_var
           
    return ds_ts

def preprocess1dVariables(vars_dict, n_components, weightings=None):
    ''' Preprocess 1-dimensional variables
        - return a dictionary containing all 1d variables
        - each value is of size n_regions * valid_components_of_each_variable, e.g. 96*6 for '1d_capacityFix'
    '''

    if not vars_dict: return None

    if weightings:
        weightings = weightings
    else:
        vars_list = list(vars_dict)
        weightings = dict.fromkeys(vars_list,1)
    
    ds_1d = {}

    for var, da in vars_dict.items():
        
        # Find the valid components for each variable
        var_mean_df = da.mean(dim="space").to_dataframe()
        var_mean_df['component_id'] = np.array(range(n_components))
        valid_component_ids = list(var_mean_df[var_mean_df[var].notna()]['component_id'])
        
        weight_factor = weightings[var]

        # Only remain the valid components
        data = da.values[valid_component_ids]
        ds_1d[var] = prep.scale(data.T) * weight_factor

    return ds_1d

def preprocess2dVariables(dataset,vars_list,handle_mode='extract',weightings=None):
    ''' Preprocess part_2 with one of the following mode:
        - Transform part_2 to a distance matrix by transforming the connectivity values to distance meaning
        - or extracted the matrices of all variables and add them up as one adjacency matrix for spectral clustering

        weightings: the weight factors of each 2d-variable, a dictionary
    '''

    if weightings:
            weight_factors = weightings
    else:
        weight_factors = dict.fromkeys(vars_list,1)
        
    ## TO-DO: maybe other transformation methods are possible, e.g. Gaussian (RBF, heat) kernel
    if handle_mode == 'reciprocal':
        ''' 2d variables -> Distance Matrix, 
            - when variables have a POSITIVE relation to the connectivity, 
                   i.e. higher value means stronger connection (then they are more likely to be grouped!)
            - the matrix for each variable should be symmetrical or triangular matrix! (undirected graph)
            - original diagonal values are 0's -> in distance matrix: also 0's!

            ## TO-DO: how to handle the nan values correctly?
            ## why need to handle it: need to add the matrices
            ## but can pre-clean the dataset before process it.
        '''

        ds_part2 = 1.0/dataset[vars_list[0]].values
        ds_part2[np.isinf(ds_part2)] = 100000 
        ds_part2[np.isnan(ds_part2)] = np.nanmean(ds_part2)
        np.fill_diagonal(ds_part2,0)
        ds_part2 = ds_part2 * weight_factors.get(vars_list[0])

        ### For multiple variables -> add their n_region*n_region matrix as the final distance matrix
        for var in vars_list[1:]:
            var_matrix = 1.0/dataset[var].values
            var_matrix[np.isinf(var_matrix)] = 100000
            var_matrix[np.isna(var_matrix)] = np.nanmean(var_matrix)
            np.fill_diagonal(var_matrix,0)
            ds_part2 += var_matrix * weight_factors.get(var)
  
        return ds_part2

    if handle_mode == 'extract':
        '''2d variables -> Adjacency matrix : 
            - add all matrices from various variables (## TO-DO:and various components)
            - adjacency matrix of a graph: symmetric, diagonals = 0
        '''

        part2 = dataset[vars_list[0]].values
        part2[np.isinf(part2)] = 100000
        part2[np.isnan(part2)] = 0
        part2 = part2 * weight_factors.get(vars_list[0])

        for var in vars_list[1:]:
            var_matrix = dataset[var].values
            var_matrix[np.isinf(var_matrix)] = 100000
            var_matrix[np.isna(var_matrix)] = 0
            part2 += var_matrix * weight_factors.get(var)

        return part2

def preprocessDataset(sds,n_regions,vars='all',dims='all',handle_mode='extract',weightings=None):
    '''Preprocess the Xarray dataset: Separate the dataset into 3 parts: time series data, 1d-var data, and 2d-var data
        - vars_ts: Time series variables  
            - sub distances based on timesteps and based on components 
            - e.g. dist = sum(dist_t0_c1 + dist_t0_c2 + dist_t1_c1 + ...)
        - vars_1d: 1d variables as features matrix
        - vars_2d: 2d variables showing connectivity between regions
            - Hierarchical: directly transformed as distance matrix (reciprocal of element values) and combine with vars_ts and vats_1d
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

    n_components = len(dataset['component'].values)

    ds_timeseries = preprocessTimeSeries(vars_ts, n_regions,n_components)
    ds_1d_vars = preprocess1dVariables(vars_1d,n_components)

    return ds_timeseries, ds_1d_vars

    if obtain == 'complete':
        if (vars_ts+vars_1d) and vars_2d:
            # Part 1: time series & 1d variables -> Feature Matrix
            ds_part1 = preprocessPart1(dataset,vars_ts+vars_1d)

            # Part2:
            ds_part2 = preprocessPart2(dataset,vars_2d)

            # Concatenate two parts together and save the region ids at index 0, length of part_1 at index 1
            ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[ds_part1.shape[1]]*n_regions]).T, ds_part1, ds_part2), axis=1)
            
            return ds
        elif (vars_ts+vars_1d) and not vars_2d:

            ds_part1 = preprocessPart1(dataset,vars_ts+vars_1d)
            ds_part2 = np.full([n_regions,1],np.nan)

            ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[ds_part1.shape[1]]*n_regions]).T, ds_part1, ds_part2), axis=1)

            return ds
        elif vars_2d and not (vars_ts+vars_1d) :

            ds_part2 = preprocessPart2(dataset,vars_2d)

            ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[0]*n_regions]).T, ds_part2), axis=1)

            return ds
        else:
            print('There is no variables in the given dataset!')
            ds = np.concatenate((np.array([range(n_regions)]).T,np.array([[0]*n_regions]).T, np.full([n_regions,1],np.nan)),axis=1)
            return ds
    
    if obtain == 'part_2':
        if vars_2d:
            return preprocessPart2(dataset,vars_2d,handle_mode='extract')
        else:
            return np.full([n_regions,1],np.nan)

    if obtain == 'part_1':
        if  vars_ts+vars_1d:
            return preprocessPart1(dataset,vars_ts+vars_1d)
        else:
            return np.full([n_regions,1],np.nan)


# Problem: not valid...
def selfDistance(ds_ts, a, b):
    ''' Custom distance function: 
        - parameters a, b: region ids
        - return: distance between a and b = euclidean distance of part_1 + corresponding value in part_2
        ---
        Metric space properties (PROOF):
        - Non-negativity: d(i,j) > 0
        - Identity of indiscernibles: d(i,i) = 0   ---> diagonal must be 0!
        - Symmetry: d(i,j) = d(j,i)  ---> the part_2 must be Symmetrical!
        - Triangle inequality: d(i,j) <= d(i,k) + d(k,j)  ---> NOT SATISFIED!!!
    '''
    # i= a[0]
    # n = int(a[1])

    # distance_part_1 = np.linalg.norm(a[2:2+n]-b[2:2+n]) if n != 0 else 0
    # distance_part_2 = b[n+int(i)+2] if np.isnan(b[2+n]) else 0

    #return distance_part_1 + distance_part_2

    distance_ts, l = 0, 0
    for var, var_dict in ds_ts.items():
        for component, data in var_dict.items():
            
            l += data.shape[1]
            
            reg_a_vector = data[a]
            reg_b_vector = data[b]

            distance_ts += sum(abs(reg_a_vector-reg_b_vector))

    distance_ts = distance_ts / l

    return distance_ts

def selfDistanceMatrix(ds_ts, n_regions):
    ''' Return a n_regions by n_regions symmetric distance matrix X 
    '''

    distMatrix = np.zeros((n_regions,n_regions))

    for i in range(n_regions):
        for j in range(i+1,n_regions):
            distMatrix[i,j] = selfDistance(ds_ts,i,j)

    distMatrix += distMatrix.T - np.diag(distMatrix.diagonal())

    return distMatrix
