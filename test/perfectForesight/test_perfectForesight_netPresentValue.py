import numpy as np


def test_perfectForesight_netPresentValue(perfectForesight_test_esM):
    """Test NPV consistency in the perfect foresight model.

    Ensures that the optimization objective matches the expected value
    and equals the sum of all NPV contributions from the optimization summary.
    """
    perfectForesight_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    np.testing.assert_almost_equal(
        perfectForesight_test_esM.pyM.Obj(), 11861.771783274202
    )
    npv_sum_optSummary = 0
    for ip in perfectForesight_test_esM.investmentPeriodNames:
        for mdl in perfectForesight_test_esM.componentModelingDict.keys():
            optSum = perfectForesight_test_esM.getOptimizationSummary(mdl, ip=ip)
            npv_sum_optSummary += optSum.loc[:, "NPVcontribution", :].sum().sum()

    np.testing.assert_almost_equal(
        perfectForesight_test_esM.pyM.Obj(), npv_sum_optSummary
    )
