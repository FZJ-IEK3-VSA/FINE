from FINE.expansionModules.rollingHorizon import rollingHorizonOptimization


def test_rollingHorizon(perfectForesight_test_esM):
    rollingHorizonOptimization(
        perfectForesight_test_esM,
        numberOfInvestmentPeriodsForRollingHorizon=2)
