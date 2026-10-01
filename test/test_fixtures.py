from fine.utils import ImplementedSolvers


def test_minimal_test_esM(minimal_test_esM):
    minimal_test_esM.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=1,
        numberOfSegmentsPerPeriod=1,
    )

    minimal_test_esM.optimize(
        timeSeriesAggregation=False,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
