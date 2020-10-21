import pandas as pd
import FINE as fn
import numpy as np

def test_shadowCostOutPut(minimal_test_esM, solver):
    '''
    Get the minimal test system, and check if the fulllload hours of electrolyzer are above 4000.
    '''
    esM = minimal_test_esM

    esM.optimize(solver=solver)

    SP = fn.getShadowPrices(esM, esM.pyM.ConstrOperation4_srcSnk,
                        dualValues=None, hasTimeSeries=True,
                        periodOccurrences=esM.periodOccurrences,
                        periodsOrder=esM.periodsOrder)

    assert np.round(SP.sum(), 4) == 0.2955

    esM.cluster(numberOfTypicalPeriods=2, numberOfTimeStepsPerPeriod=1)
    esM.optimize(timeSeriesAggregation=True, solver=solver)

    SP = fn.getShadowPrices(esM, esM.pyM.ConstrOperation4_srcSnk,
                        dualValues=None, hasTimeSeries=True,
                        periodOccurrences=esM.periodOccurrences,
                        periodsOrder=esM.periodsOrder)

    assert np.round(SP.sum(), 4) == 0.3296
    assert len(SP) == 4

