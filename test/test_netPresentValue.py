import numpy as np


def test_Mini_netPresentValue(minimal_test_esM):
    """Tests that the sum of NPV contributions equals the total objective value after optimization."""
    minimal_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    npv_sum_optSummary = 0
    for ip in minimal_test_esM.investmentPeriodNames:
        for mdl in minimal_test_esM.componentModelingDict.keys():
            optSum = minimal_test_esM.getOptimizationSummary(mdl, ip=ip)
            npv_sum_optSummary += optSum.loc[:, "NPVcontribution", :].sum().sum()

    np.testing.assert_almost_equal(minimal_test_esM.pyM.Obj(), npv_sum_optSummary)
