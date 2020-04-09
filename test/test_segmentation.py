import numpy as np

def test_segmentation(minimal_test_esM):
    '''
    Get the minimal test system, and check that for different segment and period configurations the same solution is
    found.
    '''

    # First, the non-aggregated case is compared to the aggregation mode of the model, but without aggregated data.

    esM1 = minimal_test_esM
    esM1.optimize(solver='glpk')

    esM2 = minimal_test_esM
    esM2.cluster(numberOfTypicalPeriods=2, numberOfTimeStepsPerPeriod=2, storeTSAinstance=False,
                 segmentation=True, numberOfSegmentsPerPeriod=2, clusterMethod='hierarchical',
                 sortValues=False, rescaleClusterPeriods=False)
    esM2.optimize(timeSeriesAggregation=True, solver='glpk')

    assert esM1.pyM.Obj() == esM2.pyM.Obj()

    # For the following configurations, no storage is built and the demand is always the same. Further, the prices a
    # calculated by their centroid. Accordingly, both configurations should lead to the same objective value.

    esM3 = minimal_test_esM
    esM3.cluster(numberOfTypicalPeriods=1, numberOfTimeStepsPerPeriod=4, storeTSAinstance=False,
                 segmentation=True, numberOfSegmentsPerPeriod=3, clusterMethod='hierarchical',
                 sortValues=False, rescaleClusterPeriods=False)
    esM3.optimize(timeSeriesAggregation=True, solver='glpk')

    esM4 = minimal_test_esM
    esM4.cluster(numberOfTypicalPeriods=2, numberOfTimeStepsPerPeriod=2, storeTSAinstance=False,
                 segmentation=True, numberOfSegmentsPerPeriod=1, clusterMethod='hierarchical',
                 sortValues=False, rescaleClusterPeriods=False)
    esM4.optimize(timeSeriesAggregation=True, solver='glpk')

    assert esM3.pyM.Obj() == esM4.pyM.Obj()
