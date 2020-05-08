import xarray as xr
import numpy as np


def preprocessPart1(dataset,vars_list,weightings=None):

    if weightings:
            weight_factors = weightings
    else:
        weight_factors = dict.fromkeys(vars_list,1)
    
    ds_part1 = dataset[vars_list[0]].values * weight_factors.get(vars_list[0])

    for var in vars_list[1:]:
        ds_part1 = np.concatenate((ds_part1, np.array([dataset[var].values * weight_factors.get(var)]).T), axis=1)
    
    return ds_part1

def preprocessPart2(dataset,vars_list,handle_mode='reciprocal',weightings=None):
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

def preprocessDataset(sds,n_regions,vars='all',dims='all',obtain='complete'):
    '''Preprocess the Xarray dataset: 
        - Part 1: Time series variables & 1d variables as features matrix => high-dimensional dataframe
        - Part 2: 2d variables showing connectivity between regions
                - directly transformed as distance matrix (reciprocal of element values) and combine with part_1
                - extract this part as an affinity matrix for the (un)directed graph (indicating similarity, here using adjacency matrix)
        - Test case NOW: 
                - only consider NOE single component, 
                - NO periods, 
                - ONE SINGLE var for each category
        - Return: according to 'obtain' - part_1, part_2 or complete 
    '''

    dataset = sds.xr_dataset

    # Traverse all variables in the dataset, and put them in separate categories
    vars_ts = []
    vars_1d = []
    vars_2d = []

    for varname, da in dataset.data_vars.items():
        if da.dims == ('Period','component','space', 'TimeStep'):
            vars_ts.append(varname)
        if da.dims == ('component','space'):
            vars_1d.append(varname)
        if da.dims == ('component','space','space_2'):
            vars_2d.append(varname)

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


# To-Do: not valid...
def selfDistance(a,b):
    ''' Custom distance function: 
        - parameters a, b: two region-data containing region-id at index 0, length of part_1 at index 1,  part_1 data and part_2 data
        - return: distance between a and b = euclidean distance of part_1 + corresponding value in part_2
        ---
        Metric space properties (PROOF):
        - Non-negativity: d(i,j) > 0
        - Identity of indiscernibles: d(i,i) = 0   ---> diagonal must be 0!
        - Symmetry: d(i,j) = d(j,i)  ---> the part_2 must be Symmetrical!
        - Triangle inequality: d(i,j) <= d(i,k) + d(k,j)  ---> NOT SATISFIED!!!
    '''
    i= a[0]
    n = int(a[1])

    distance_part_1 = np.linalg.norm(a[2:2+n]-b[2:2+n]) if n != 0 else 0
    distance_part_2 = b[n+int(i)+2] if np.isnan(b[2+n]) else 0

    return distance_part_1 + distance_part_2

