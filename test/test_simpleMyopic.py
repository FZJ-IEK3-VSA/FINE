import fine as fn
import pytest

# fn.optimizeSimpleMyopic has been removed; fine.expansionModules.rollingHorizon.
# rollingHorizonOptimization with numberOfInvestmentPeriodsForRollingHorizon=1
# covers the same myopic foresight use case (see issue #640) and is the
# recommended replacement. It also supports CO2 reduction pathways (via a
# per-investment-period balanceLimit) and technical-lifetime expiry, which are
# covered by test_co2_target_* and test_exceeded_lifetime_* in
# test/test_rolling_horizon.py.


def test_optimizeSimpleMyopic_raises_and_points_to_rolling_horizon():
    with pytest.raises(NotImplementedError, match="rollingHorizonOptimization"):
        fn.optimizeSimpleMyopic(esM=None, startYear=2020)
