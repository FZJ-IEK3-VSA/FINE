import FINE as fn
import numpy as np
import pytest


def test_Mini_netPresentValue(minimal_test_esM):
    minimal_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    # the sum of all npv contributions in the optimization summary must equal
    # the objective value
    npv_sum_optSummary=0
    for ip in minimal_test_esM.investmentPeriodList:
        for mdl in minimal_test_esM.componentModelingDict.keys():
            optSum=minimal_test_esM.getOptimizationSummary(mdl,ip=ip)
            npv_sum_optSummary+=optSum.loc[:,"NPVcontribution",:].sum().sum()
    
    np.testing.assert_almost_equal(
        minimal_test_esM.pyM.Obj(), npv_sum_optSummary
    ) 


def test_DSM_netPresentValue(dsm_test_esM):
    dsm_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    # the sum of all npv contributions in the optimization summary must equal
    # the objective value
    npv_sum_optSummary=0
    for ip in dsm_test_esM.investmentPeriodList:
        for mdl in dsm_test_esM.componentModelingDict.keys():
            optSum=dsm_test_esM.getOptimizationSummary(mdl,ip=ip)
            npv_sum_optSummary+=optSum.loc[:,"NPVcontribution",:].sum().sum()
    
    np.testing.assert_almost_equal(
        dsm_test_esM.pyM.Obj(), npv_sum_optSummary
    ) 
