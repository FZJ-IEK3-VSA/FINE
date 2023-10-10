import pytest
import numpy as np
import pandas as pd

import FINE as fn


@pytest.mark.parametrize("TSA", [True, False])
def test_storageTimeseriesSetup(TSA, minimal_test_esM):
    dummy_time_series = pd.DataFrame(
        [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T

    ## add two dummy storages
    minimal_test_esM.add(
        fn.Storage(
            esM=minimal_test_esM,
            name="dummy_storage_1",
            commodity="hydrogen",
            hasCapacityVariable=True,
            chargeOpRateMax=dummy_time_series,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    minimal_test_esM.add(
        fn.Storage(
            esM=minimal_test_esM,
            name="dummy_storage_2",
            commodity="hydrogen",
            hasCapacityVariable=True,
            chargeOpRateMax=dummy_time_series,
            chargeOpRateFix=dummy_time_series,
            capacityVariableDomain="continuous",
            stateOfChargeMin=0.33,
            investPerCapacity=0.5,  # eur/kWh
            interestRate=0.08,
            economicLifetime=30,
        )
    )

    ## test a case where TSA is not used
    if TSA:
        ## function call
        minimal_test_esM.aggregateTemporally(
            numberOfTypicalPeriods=2, numberOfTimeStepsPerPeriod=1
        )
        minimal_test_esM.declareOptimizationProblem(timeSeriesAggregation=True)

        ## assertion

        ### dummy_storage_1
        original_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "chargeOpRateMax"
        )
        full_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "fullChargeOpRateMax"
        )
        processed_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "processedChargeOpRateMax"
        )
        aggregated_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "aggregatedChargeOpRateMax"
        )

        assert np.array_equal(original_max_stg1.values, dummy_time_series.values)
        assert np.array_equal(full_max_stg1.values, original_max_stg1.values)
        assert np.array_equal(processed_max_stg1.values, aggregated_max_stg1.values)

        ### dummy_storage_2
        original_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "chargeOpRateMax"
        )
        full_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "fullChargeOpRateMax"
        )
        processed_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "processedChargeOpRateMax"
        )
        aggregated_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "aggregatedChargeOpRateMax"
        )

        original_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "chargeOpRateFix"
        )
        full_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "fullChargeOpRateFix"
        )
        processed_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "processedChargeOpRateFix"
        )
        aggregated_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "aggregatedChargeOpRateFix"
        )

        assert np.array_equal(original_max_stg2.values, dummy_time_series.values)
        assert full_max_stg2 == None
        assert processed_max_stg2 == None
        assert aggregated_max_stg2 == None

        assert np.array_equal(original_fix_stg2.values, dummy_time_series.values)
        assert np.array_equal(full_fix_stg2.values, original_fix_stg2.values)
        assert np.array_equal(processed_fix_stg2.values, aggregated_fix_stg2.values)

    ## test a case where TSA is used
    else:
        ## function call
        minimal_test_esM.declareOptimizationProblem(timeSeriesAggregation=False)

        ## assertion

        ### dummy_storage_1
        original_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "chargeOpRateMax"
        )
        full_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "fullChargeOpRateMax"
        )
        processed_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "processedChargeOpRateMax"
        )
        aggregated_max_stg1 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_1", "aggregatedChargeOpRateMax"
        )

        assert np.array_equal(original_max_stg1.values, dummy_time_series.values)
        assert np.array_equal(full_max_stg1.values, original_max_stg1.values)
        assert np.array_equal(processed_max_stg1.values, full_max_stg1.values)
        assert aggregated_max_stg1 == None

        ### dummy_storage_2
        original_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "chargeOpRateMax"
        )
        full_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "fullChargeOpRateMax"
        )
        processed_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "processedChargeOpRateMax"
        )
        aggregated_max_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "aggregatedChargeOpRateMax"
        )

        original_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "chargeOpRateFix"
        )
        full_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "fullChargeOpRateFix"
        )
        processed_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "processedChargeOpRateFix"
        )
        aggregated_fix_stg2 = minimal_test_esM.getComponentAttribute(
            "dummy_storage_2", "aggregatedChargeOpRateFix"
        )

        assert np.array_equal(original_max_stg2.values, dummy_time_series.values)
        assert full_max_stg2 == None
        assert processed_max_stg2 == None
        assert aggregated_max_stg2 == None

        assert np.array_equal(original_fix_stg2.values, dummy_time_series.values)
        assert np.array_equal(full_fix_stg2.values, original_fix_stg2.values)
        assert np.array_equal(processed_fix_stg2.values, full_fix_stg2.values)
        assert aggregated_fix_stg2 == None
