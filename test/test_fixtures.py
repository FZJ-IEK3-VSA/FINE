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


def test_multi_node_test_esM_init(multi_node_test_esM_init):
    multi_node_test_esM_init.aggregateTemporally(
        numberOfTypicalPeriods=5,
        segmentation=False,
        sortValues=True,
        representationMethod=None,
        rescaleClusterPeriods=True,
    )

    multi_node_test_esM_init.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
