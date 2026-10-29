import numpy as np
import pandas as pd
import pytest
import fine as fn
from copy import deepcopy
from fine.utils import ImplementedSolvers


def test_aggregation_zero_columns_do_not_affect_result(single_node_test_esM):
    """Adding an all-zero time series column should not change the aggregation result.

    Two equivalent models are built: one without any zero-only time series
    (so TSAM receives all columns as-is), and one with an extra Source whose
    operationRateMax is all zeros (triggering the zero-column drop in
    aggregateTemporally). Since the extra Source cannot operate, both models
    represent the same optimisation problem and must yield the same objective
    after temporal aggregation. Covers both segmentation=False and
    segmentation=True code paths.
    """
    tsa_kwargs_base = dict(
        storeTSAinstance=False,
        numberOfTypicalPeriods=2,
        numberOfTimeStepsPerPeriod=2,
        clusterMethod="hierarchical",
        sortValues=False,
        rescaleClusterPeriods=False,
        representationMethod=None,
    )

    for segmentation, extra_kwargs in [
        (False, {}),
        (True, {"numberOfSegmentsPerPeriod": 2}),
    ]:
        # Model without zero columns — TSAM sees all columns
        esM_base = deepcopy(single_node_test_esM)
        esM_base.aggregateTemporally(
            segmentation=segmentation, **tsa_kwargs_base, **extra_kwargs
        )
        esM_base.optimize(
            timeSeriesAggregation=True,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )

        # Model with an extra Source whose operationRateMax is all zeros.
        # The zero column must be dropped before TSAM and restored afterwards.
        esM_with_zero = deepcopy(single_node_test_esM)
        esM_with_zero.add(
            fn.Source(
                esM=esM_with_zero,
                name="Zero source",
                commodity="electricity",
                hasCapacityVariable=False,
                operationRateMax=pd.Series(np.zeros(4)),
            )
        )
        esM_with_zero.aggregateTemporally(
            segmentation=segmentation, **tsa_kwargs_base, **extra_kwargs
        )
        esM_with_zero.optimize(
            timeSeriesAggregation=True,
            solver=ImplementedSolvers.STANDARD_SOLVER.value,
        )

        assert esM_base.pyM.Obj() == pytest.approx(esM_with_zero.pyM.Obj()), (
            f"segmentation={segmentation}: objective differs when zero columns are present"
        )


def test_segmentation(minimal_test_esM):
    """Get the minimal test system, and check that for different segment and period configurations the same solution is
    found.
    """
    # First, the non-aggregated case is compared to the aggregation mode of the model, but without aggregated data.
    # For this, the mini system is first optimized without any aggregation at all.
    esM1 = minimal_test_esM
    esM1.optimize(solver=ImplementedSolvers.STANDARD_SOLVER.value)
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
    esM2.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
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
    esM3.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
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
    esM4.optimize(
        timeSeriesAggregation=True,
        solver=ImplementedSolvers.STANDARD_SOLVER.value,
    )
    # Here, it is checked that the results of the third and the fourth model run are identical because no storage is
    # chosen and because of the averaged data the costs should stay the same.
    # Note: The segmentation also averages the models' constraints, but in this specific example, the most restrictive
    # and thus size-determining constraints of the model are coincidentally not affected by the aggregation and the
    # optimal solutions of the third and fourth model are identical.
    assert esM3.pyM.Obj() == esM4.pyM.Obj()
