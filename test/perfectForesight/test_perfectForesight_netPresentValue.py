import fine as fn
import numpy as np
import pytest


def test_perfectForesight_netPresentValue(perfectForesight_test_esM):
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    np.testing.assert_almost_equal(
        perfectForesight_test_esM.pyM.Obj(), 11861.771783274202
    )
    # the sum of all npv contributions in the optimization summary must equal
    # the objective value
    npv_sum_optSummary = 0
    for ip in perfectForesight_test_esM.investmentPeriodNames:
        for mdl in perfectForesight_test_esM.componentModelingDict.keys():
            optSum = perfectForesight_test_esM.getOptimizationSummary(mdl, ip=ip)
            npv_sum_optSummary += optSum.loc[:, "NPVcontribution", :].sum().sum()

    np.testing.assert_almost_equal(
        perfectForesight_test_esM.pyM.Obj(), npv_sum_optSummary
    )
