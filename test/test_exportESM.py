import FINE 
import numpy as np
import os


def test_export_to_dict(minimal_test_esM):
    '''
    Export esM class to file
    '''

    esm_dict, comp_dict = FINE.IOManagement.exportToDict(minimal_test_esM)

    return

def test_import_from_dict(minimal_test_esM):
    '''
    Get a dictionary of a esM class and write it to another esM
    '''
    esm_dict, comp_dict = FINE.IOManagement.exportToDict(minimal_test_esM)

    # modify log level
    esm_dict['verboseLogLevel'] = 0

    # write a new FINE model from it
    new_esM = FINE.IOManagement.importFromDict(esm_dict, comp_dict)
    
    # optimize the model
    new_esM.optimize(timeSeriesAggregation=False, solver = 'glpk')

    # test if solve fits to the original results
    testresults = new_esM.componentModelingDict["SourceSinkModel"].operationVariablesOptimum.xs('Electricity market')
    np.testing.assert_array_almost_equal(testresults.values, [np.array([1.877143e+07,  3.754286e+07,  0.0,  1.877143e+07]),],decimal=-3)