import FINE 
import numpy as np
import os
import pytest
import xarray as xr
import json

from FINE.IOManagement import dictIO

def test_export_to_dict(minimal_test_esM):
    '''
    Export esM class to file
    '''

    esm_dict, comp_dict = dictIO.exportToDict(minimal_test_esM)

    # TODO: test against expected values

def test_export_to_dict_2(multi_node_test_esM_init):
    # TODO: use parametrization instead of repeating the test
    '''
    Export esM class to file
    '''

    esm_dict, comp_dict = dictIO.exportToDict(multi_node_test_esM_init)

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

def test_import_from_dict_2(multi_node_test_esM_init, multi_node_test_esM_optimized):
    # TODO: use parametrization instead of repeating the test
    '''
    Get a dictionary of a esM class and write it to another esM
    '''
    esm_dict, comp_dict = dictIO.exportToDict(multi_node_test_esM_init)

    # modify log level
    esm_dict['verboseLogLevel'] = 0

    # write a new FINE model from it
    esM_imported_from_dict = dictIO.importFromDict(esm_dict, comp_dict)

    esM_imported_from_dict.cluster(numberOfTypicalPeriods=3)

    esM_imported_from_dict.optimize(timeSeriesAggregation=True, solver = 'glpk')

    # test if solve fits to the original results
    expected_results = multi_node_test_esM_optimized.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Biogas purchase')
    test_results = esM_imported_from_dict.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Biogas purchase')

    np.testing.assert_array_almost_equal(test_results.values, expected_results.values,decimal=-3)

def test_export_to_json(minimal_test_esM):
    dictIO.exportToJSON(minimal_test_esM, filename="minimal_esM.json")

def test_import_from_json(minimal_test_esM):
    esM = dictIO.importFromJSON("minimal_esM.json")