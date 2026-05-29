from fine.utils import ImplementedSolvers


def test_performanceSummary(minimal_test_esM):
    logFileName = "run.log"

    minimal_test_esM.optimize(
        timeSeriesAggregation=False,
        logFileName=logFileName,
        optimizationSpecs="OptimalityTol=1e-3 method=2 cuts=0 MIPGap=5e-3",
        includePerformanceSummary=True,
        solver=ImplementedSolvers.GUROBI.value,
    )

    summary = minimal_test_esM.performanceSummary

    print(summary)

    assert summary.loc[("FineParameters", "noOfRegions")]["Value"] == len(
        minimal_test_esM.locations
    )
    assert summary.loc[("GurobiSummary", "Status")]["Value"] == "OPTIMAL"
