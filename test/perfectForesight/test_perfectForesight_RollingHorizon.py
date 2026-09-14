import pytest

import fine as fn
from fine.expansionModules.rollingHorizon import rollingHorizonOptimization


def test_rollingHorizon(perfectForesight_test_esM):
    results = rollingHorizonOptimization(
        perfectForesight_test_esM,
        scenario_name="test",
        timeSeriesAggregation=True,
        timeSeriesAggregationSettings={
            "n_clusters": 1,
            "period_duration": perfectForesight_test_esM.hoursPerTimeStep,
            "segments": fn.SegmentConfig(n_segments=1),
        },
        numberOfInvestmentPeriodsForRollingHorizon=2,
    )

    # check that commissioning of first year is in stock of second year
    assert results[2020].getOptimizationSummary("SourceSinkModel", ip=2020).loc[
        "PV", "commissioning"
    ].squeeze()["ForesightLand"] == pytest.approx(
        results[2025].getComponent("PV").stockCommissioning[2020]["ForesightLand"]
    )
