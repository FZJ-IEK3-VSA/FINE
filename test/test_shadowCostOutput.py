import fine as fn
import numpy as np

from fine.IOManagement.utilsIO import getShadowPriceXarray
from fine.utils import ImplementedSolvers


def test_shadowCostOutPut(minimal_test_esM):
    """Get the minimal test system, and check if the fulllload hours of electrolyzer are above 4000."""
    esM = minimal_test_esM

    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

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

    esM.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )

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


def test_shadow_price_with_multiple_ip(perfectForesight_test_esM):
    esM = perfectForesight_test_esM

    esM.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)

    sp_xr = getShadowPriceXarray(esM, constraint_str="commodityBalanceConstraint")

    assert sp_xr is not None
    assert "ip" in sp_xr.dims
    assert list(sp_xr.coords["ip"].values) == esM.investmentPeriodNames
    assert "component" in sp_xr.dims
    assert "space" in sp_xr.dims
    assert "time" in sp_xr.dims
    assert sp_xr.attrs["constraint"] == "commodityBalanceConstraint"
