import xarray as xr
import numpy as np


def preprocessDataset(sds,n_regions,vars='all',dims='all'):
    '''Preprocess the Xarray dataset: 
        - Part 1: Time series variables & 1d variables as features matrix
        - Part 2: transfer 2d variables directly as distance matrix / dissimilarity matrix (Multiplicative inverse of element values)
        - Test case: only consider one single component
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

    # Part 1: time series & 1d variables -> Feature Matrix

    # TO-DO: what if vars_ts / vars_1d are empty?

    ds_part1 = dataset[vars_ts[0]].values
    for var in vars_ts[1:] + vars_1d:
        ds_part1 = np.concatenate((ds_part1, np.array([dataset[var].values]).T), axis=1)

    # Part 2: 2d variables -> Distance Matrix
    ds_part2 = 1.0/dataset[vars_2d[0]].values
    for var in vars_2d[1:]:
        ds_part2 = np.concatenate((ds_part2, 1.0/dataset[var].values), axis=1)
    ds_part2[np.isinf(ds_part2)] = 0

    # Concatenate two parts together and save the region ids at index 0, length of part_1 at index 1
    ds = np.concatenate((np.array([range(n_regions)]).T, np.array([[ds_part1.shape[1]]*n_regions]).T, ds_part1, ds_part2), axis=1)

    return ds


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
    return np.linalg.norm(a[2:2+n]-b[2:2+n]) + b[n+int(i)+2]

