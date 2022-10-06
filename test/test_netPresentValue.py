import FINE as fn
import numpy as np
import pytest


def test_Mini_netPresentValue(minimal_test_esM):
    minimal_test_esM.optimize(timeSeriesAggregation=False, solver="glpk")
    # the sum of all npv contributions in the optimization summary must equal
    # the objective value
    npv_sum_optSummary = 0
    for ip in minimal_test_esM.investmentPeriodNames:
        for mdl in minimal_test_esM.componentModelingDict.keys():
            optSum = minimal_test_esM.getOptimizationSummary(mdl, ip=ip)
            npv_sum_optSummary += optSum.loc[:, "NPVcontribution", :].sum().sum()

    np.testing.assert_almost_equal(minimal_test_esM.pyM.Obj(), npv_sum_optSummary)


def test_DSM_netPresentValue(dsm_test_esM):
    # add DSM
    tFwd = 3
    tBwd = 3
    esM_with = dsm_test_esM[0]
    shiftMax = 10
    esM_with.add(
        fn.DemandSideManagementBETA(
            esM=esM_with,
            name="flexible demand",
            commodity="electricity",
            hasCapacityVariable=False,
            tFwd=tFwd,
            tBwd=tBwd,
            operationRateFix=dsm_test_esM[1],
            opexShift=1,
            shiftDownMax=shiftMax,
            shiftUpMax=shiftMax,
            socOffsetDown=-1,
            socOffsetUp=-1,
        )
    )

    esM_with.optimize(timeSeriesAggregation=False, solver="glpk")
    # the sum of all npv contributions in the optimization summary must equal
    # the objective value
    npv_sum_optSummary = 0
    for ip in esM_with.investmentPeriodNames:
        for mdl in esM_with.componentModelingDict.keys():
            optSum = esM_with.getOptimizationSummary(mdl, ip=ip)
            npv_sum_optSummary += optSum.loc[:, "NPVcontribution", :].sum().sum()

    np.testing.assert_almost_equal(esM_with.pyM.Obj(), npv_sum_optSummary)
