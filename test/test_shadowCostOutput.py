import pandas as pd
import FINE as fn
import numpy as np


def test_shadowCostOutPut(minimal_test_esM):
    """
    Get the minimal test system, and check if the fulllload hours of electrolyzer are above 4000.
    """
    esM = minimal_test_esM

    esM.optimize(solver="glpk")

    SP = fn.getShadowPrices(
        esM,
        esM.pyM.commodityBalanceConstraint,
        dualValues=None,
        hasTimeSeries=True,
        periodOccurrences=esM.periodOccurrences,
        periodsOrder=esM.periodsOrder,
    )

    assert np.round(SP.loc["hydrogen", "IndustryLocation"].sum(), 4) == 0.2955

    esM.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=1,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    esM.optimize(timeSeriesAggregation=True, solver="glpk")

    SP = fn.getShadowPrices(
        esM,
        esM.pyM.commodityBalanceConstraint,
        dualValues=None,
        hasTimeSeries=True,
        periodOccurrences=esM.periodOccurrences,
        periodsOrder=esM.periodsOrder,
    )

    assert np.round(SP.loc["hydrogen", "IndustryLocation"].sum(), 4) == 0.3296
    assert len(SP.loc["hydrogen", "IndustryLocation"]) == 4
