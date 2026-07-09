from fine.expansionModules.rollingHorizon import rollingHorizonOptimization


def test_rollingHorizon(perfectForesight_test_esM):
    results = rollingHorizonOptimization(
        perfectForesight_test_esM,
        scenario_name="test",
        timeSeriesAggregation=True,
        numberOfInvestmentPeriodsForRollingHorizon=2,
        numberOfTimeStepsPerPeriod=1,
        numberOfSegments=1,
        numberOfTypicalPeriods=1,
    )

    # check that commissioning of first year is in stock of second year
    assert (
        results[2020]
        .getOptimizationSummary("SourceSinkModel", ip=2020)
        .loc["PV", "commissioning"]
        .squeeze()["ForesightLand"]
        == results[2025].getComponent("PV").stockCommissioning[2020]["ForesightLand"]
    )
