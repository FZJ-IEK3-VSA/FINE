import pytest
import warnings

from FINE import utils

def test_userWarnings_esm(minimal_test_esM):
    '''
    Tests if the warnings are supressed only when intended and shown otherwise in energySystemModel.py!
    '''
    
    #TESTING FOR energySystemModel.py
    with warnings.catch_warnings(record=True) as w:
        timeSeriesAggregation=False
        solver='gurobi'
        # Code
        minimal_test_esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)
        minimal_test_esM.getOptimizationSummary('SourceSinkModel', outputLevel=5)
        
        assert "Invalid input. An outputLevel parameter of 2 is assumed." in str(w[-1].message)
    #TODO: test also if DeprecationWarning and FutureWarning are ignored

    
def test_userWarnings_utils(minimal_test_esM):
    '''
    Tests if the warnings are shown in utils.
    '''
    #TESTING FOR utils.py
    with warnings.catch_warnings(record=True) as w:
        # Code
        utils.checkSinkCompCO2toEnvironment(minimal_test_esM, CO2ReductionTargets=1)
        assert 'CO2 emissions are not considered in the current esM. CO2ReductionTargets will be ignored.' in str(w[-1].message)

