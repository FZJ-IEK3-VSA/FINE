import numpy as np


def test_segmentation(minimal_test_esM):
    """
    Get the minimal test system, and check that for different segment and period configurations the same solution is
    found.
    """

    # First, the non-aggregated case is compared to the aggregation mode of the model, but without aggregated data.
    # For this, the mini system is first optimized without any aggregation at all.
    esM1 = minimal_test_esM
    esM1.optimize(solver="glpk")
    # Then, the four time steps of the model are represented by two 4380-hourly typical periods with two segments per
    # typical period, so effectively the data is not aggregated.
    esM2 = minimal_test_esM
    esM2.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=2,
        storeTSAinstance=False,
        segmentation=True,
        numberOfSegmentsPerPeriod=2,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )
    esM2.optimize(timeSeriesAggregation=True, solver="glpk")
    # It is now checked that both models, i.e. the one without aggregation at all and the one without aggregation, but
    # in aggregation mode, lead to the same result.
    assert esM1.pyM.Obj() == esM2.pyM.Obj()

    # For the following configurations, no storage is built and the demand is always the same. Further, the prices are
    # calculated by their centroid. Accordingly, both configurations should lead to the same objective value.
    # First, the mini system is clustered to one period with four time steps that is further segmented to three segments
    # so that the first segment is twice as long as the first and the second segment.
    esM3 = minimal_test_esM
    esM3.aggregateTemporally(
        numberOfTypicalPeriods=1,
        numberOfTimeStepsPerPeriod=4,
        storeTSAinstance=False,
        segmentation=True,
        numberOfSegmentsPerPeriod=3,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )
    esM3.optimize(timeSeriesAggregation=True, solver="glpk")
    # Then, the model is optimized again with two 4380-hourly periods that are segmented to one segment per period, i.e.
    # the model contains only two time steps in total with averaged values in each period.
    esM4 = minimal_test_esM
    esM4.aggregateTemporally(
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=2,
        storeTSAinstance=False,
        segmentation=True,
        numberOfSegmentsPerPeriod=1,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )
    esM4.optimize(timeSeriesAggregation=True, solver="glpk")
    # Here, it is checked that the results of the third and the fourth model run are identical because no storage is
    # chosen and because of the averaged data the costs should stay the same.
    # Note: The segmentation also averages the models' constraints, but in this specific example, the most restrictive
    # and thus size-determining constraints of the model are coincidentally not affected by the aggregation and the
    # optimal solutions of the third and fourth model are identical.
    assert esM3.pyM.Obj() == esM4.pyM.Obj()
