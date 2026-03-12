from FINE.expansionModules.rollingHorizon import rollingHorizonOptimization
from pathlib import Path


def test_rollingHorizon(perfectForesight_test_esM):
    results = rollingHorizonOptimization(
        perfectForesight_test_esM,
        resultExportPath=Path(__file__).resolve().parent,
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

    # delete created excel lists
    for year in [2020, 2025, 2030, 2035, 2040]:
        path = Path(__file__).resolve().parent
        (path / f"test_rollingHorizon_{year}.xlsx").unlink()
