import FINE 
import numpy as np
import os
import pytest
import xarray as xr
import json

from FINE.IOManagement import dictIO, xarray_io

def test_export_to_dict(minimal_test_esM):
    '''
    Export esM class to file
    '''

    esm_dict, comp_dict = dictIO.exportToDict(minimal_test_esM)

    # TODO: test against expected values

def test_import_from_dict(minimal_test_esM):

    '''
    Get a dictionary of a esM class and write it to another esM
    '''
    esm_dict, comp_dict = dictIO.exportToDict(minimal_test_esM)

    # modify log level
    esm_dict['verboseLogLevel'] = 0

    # write a new FINE model from it
    new_esM = dictIO.importFromDict(esm_dict, comp_dict)

    new_esM.optimize(timeSeriesAggregation=False, solver = 'glpk')

    # test if solve fits to the original results
    testresults = new_esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Electricity market')

    np.testing.assert_array_almost_equal(testresults.values, [np.array([1.877143e+07,  3.754286e+07,  0.0,  1.877143e+07]),],decimal=-3)


def test_dimensional_data_to_xarray_multinode(multi_node_test_esM_init):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_multinode.nc4')

    ds_extracted = xarray_io.dimensional_data_to_xarray(multi_node_test_esM_init)

    # ds_extracted.to_netcdf(nc_path)

    ds_expected = xr.open_dataset(nc_path)

    xr.testing.assert_allclose(ds_extracted.sortby('location'), ds_expected.sortby('location'))


def test_dimensional_data_to_xarray_minimal(minimal_test_esM):

    nc_path = os.path.join(os.path.dirname(__file__), '../data/ds_minimal.nc4')

    ds_extracted = xarray_io.dimensional_data_to_xarray(minimal_test_esM)

    # ds_extracted.to_netcdf(nc_path)

    ds_expected = xr.open_dataset(nc_path)

    xr.testing.assert_allclose(ds_extracted.sortby('location'), ds_expected.sortby('location'))