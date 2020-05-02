import xarray as xr
import numpy as np


def preprocessPart1(dataset,vars_list):
    
    ds_part1 = dataset[vars_list[0]].values

    for var in vars_list[1:]:
        ds_part1 = np.concatenate((ds_part1, np.array([dataset[var].values]).T), axis=1)
    
    return ds_part1

def preprocessPart2(dataset,vars_list,handle_mode='reciprocal',weightings=None):
    '''
    Part_2 either as distance matrix or extracted as a dictionary

    weightings: the weight factors of each 2d-variable, a dictionary
    '''

    if handle_mode == 'reciprocal':
        ''' 2d variables -> Distance Matrix, 
            - when variables have a POSITIVE relation to the connectivity, 
                   i.e. higher value means stronger connection (then they are more likely to be grouped!)
            - the matrix for each variable should be symmetrical or triangular matrix! (undirected graph)

            ## TO-DO: how to handle the nan values correctly?
        '''

        if weightings:
            weight_factors = weightings
        else:
            weight_factors = dict.fromkeys(vars_list,1)

        ds_part2 = 1.0/dataset[vars_list[0]].values
        ds_part2[np.isinf(ds_part2)] = 100000
        ds_part2[np.isnan(ds_part2)] = np.nanmean(ds_part2)
        ds_part2 = ds_part2 * weight_factors.get(vars_list[0])

        ### For multiple variables -> add their n_region*n_region matrix as the final distance matrix
        for i in range(1,len(vars_list)):
            var_matrix = 1.0/dataset[vars_list[i]].values
            var_matrix[np.isinf(var_matrix)] = 100000
            var_matrix[np.isna(var_matrix)] = np.nanmean(var_matrix)
            ds_part2 += var_matrix * weight_factors.get(vars_list[i])
        
        return ds_part2

    if handle_mode == 'extract':
        '''Extract this part as a dictionary'''

        part2 = {}
        for var in vars_list:
            part2[var] = dataset[var].values

        return part2

def preprocessDataset(sds,n_regions,vars='all',dims='all'):
    '''Preprocess the Xarray dataset: 
        - Part 1: Time series variables & 1d variables as features matrix => high-dimensional dataframe
        - Part 2: 2d variables showing connectivity between regions
                - directly transformed as distance matrix (reciprocal of element values) and combine with part_1
                - extract this part as (un)directed graph
        - Test case NOW: 
                - only consider NOE single component, 
                - NO periods, 
                - ONE SINGLE var for each category
    '''

    dataset = sds.xr_dataset

    # Traverse all variables in the dataset, and put them in separate categories
    vars_ts = []
    vars_1d = []
    vars_2d = []

    for varname, da in dataset.data_vars.items():
        if da.dims == ('space', 'TimeStep'):
            vars_ts.append(varname)
        if da.dims == ('space',):
            vars_1d.append(varname)
        if da.dims == ('space', 'space_2'):
            vars_2d.append(varname)

    if (vars_ts+vars_1d) and vars_2d:
        # Part 1: time series & 1d variables -> Feature Matrix
        ds_part1 = preprocessPart1(dataset,vars_ts+vars_1d)

        # Part2:
        ds_part2 = preprocessPart2(dataset,vars_2d)

        extracted_part2 = preprocessPart2(dataset,vars_2d,handle_mode='extract')

        # Concatenate two parts together and save the region ids at index 0, length of part_1 at index 1
        ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[ds_part1.shape[1]]*n_regions]).T, ds_part1, ds_part2), axis=1)
        
        return ds, extracted_part2
    elif (vars_ts+vars_1d) and not vars_2d:

        ds_part1 = preprocessPart1(dataset,vars_ts+vars_1d)
        ds_part2 = np.full([n_regions,1],np.nan)

        ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[ds_part1.shape[1]]*n_regions]).T, ds_part1, ds_part2), axis=1)

        return ds, {}
    elif vars_2d and not (vars_ts+vars_1d) :

        ds_part2 = preprocessPart2(dataset,vars_2d)
        extracted_part2 = preprocessPart2(dataset,vars_2d,handle_mode='extract')

        ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[0]*n_regions]).T, ds_part2), axis=1)

        return ds, extracted_part2
    else:
        print('There is no variables in the given dataset!')
        ds = np.concatenate((np.array([range(n_regions)]).T,np.array([[0]*n_regions]).T, np.full([n_regions,1],np.nan)
),axis=1)
        return ds, {}
  


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

