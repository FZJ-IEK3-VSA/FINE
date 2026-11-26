"""Tests for ``aggregateTemporally`` and the ETHOS.TSAM 4.x result it produces."""

import warnings

import numpy as np
import pandas as pd
import pytest
import tsam
from tsam import AggregationResult, ClusterConfig, ExtremeConfig, SegmentConfig

from fine import utils


@pytest.fixture(scope="module")
def timeSeriesData():
    """Return a deterministic multi column time series covering 30 days."""
    numberOfTimeSteps = 24 * 30
    steps = np.arange(numberOfTimeSteps)
    return pd.DataFrame(
        {
            "demand": 1 + 0.3 * np.sin(steps / 12) + 0.05 * np.cos(steps / 3),
            "solar": np.clip(np.sin(steps / 12), 0, None),
            "wind": 0.5 + 0.4 * np.cos(steps / 17),
        },
        index=pd.date_range(
            "2050-01-01 00:30:00", periods=numberOfTimeSteps, freq="1h"
        ),
    )


# ==================================== Period duration ====================================#


@pytest.mark.parametrize(
    ("periodDuration", "hours"),
    [(24, 24.0), (24.0, 24.0), ("24h", 24.0), ("1d", 24.0), ("7d", 168.0)],
)
def test_period_duration_is_parsed_to_hours(periodDuration, hours):
    assert utils.parsePeriodDurationHours(periodDuration) == hours


# ==================================== The ETHOS.TSAM result ====================================#


def test_the_result_carries_what_the_model_reads(timeSeriesData):
    """The attributes aggregateTemporally and the performance summary rely on."""
    result = tsam.aggregate(
        timeSeriesData,
        n_clusters=4,
        period_duration=24,
        temporal_resolution=1,
        cluster=ClusterConfig(method="hierarchical", representation="distribution"),
        segments=SegmentConfig(n_segments=6, representation="distribution"),
    )

    assert isinstance(result, AggregationResult)
    assert result.n_clusters == 4
    assert result.n_segments == 6
    assert result.n_timesteps_per_period == 24
    assert list(result.period_index) == [0, 1, 2, 3]
    assert len(result.cluster_assignments) == 30
    assert result.clustering.cluster_config.method == "hierarchical"
    assert result.clustering.cluster_config.solver == "highs"
    assert result.clustering.segment_config.n_segments == 6
    assert result.clustering_duration >= 0


def test_the_segmented_representatives_carry_the_segment_durations(timeSeriesData):
    """The third index level holds the segment lengths, which the model splits off."""
    result = tsam.aggregate(
        timeSeriesData,
        n_clusters=4,
        period_duration=24,
        temporal_resolution=1,
        segments=SegmentConfig(n_segments=6),
    )
    typicalPeriods = result.cluster_representatives

    assert typicalPeriods.index.names[1:] == ["Segment Step", "Segment Duration"]
    data = typicalPeriods.reset_index(level=2, drop=True)
    assert data.shape == (4 * 6, 3)

    durations = pd.Series(
        typicalPeriods.index.get_level_values("Segment Duration"), index=data.index
    )
    assert (durations.groupby(level=0).sum() == 24).all()


def test_the_unsegmented_representatives_are_indexed_by_time_step(timeSeriesData):
    result = tsam.aggregate(
        timeSeriesData, n_clusters=4, period_duration=24, temporal_resolution=1
    )
    assert result.n_segments is None
    assert result.cluster_representatives.index.nlevels == 2
    assert result.cluster_representatives.shape == (4 * 24, 3)


def test_extreme_periods_reach_tsam(timeSeriesData):
    """A demand spike must only survive the aggregation when it is asked for."""
    spikedData = timeSeriesData.copy()
    spikedData.iloc[24 * 17 + 13, spikedData.columns.get_loc("demand")] = 9.0

    withoutExtremes = tsam.aggregate(spikedData, n_clusters=4, period_duration=24)
    withExtremes = tsam.aggregate(
        spikedData,
        n_clusters=4,
        period_duration=24,
        extremes=ExtremeConfig(max_value=["demand"]),
    )

    assert withoutExtremes.cluster_representatives["demand"].max() < 9.0
    assert withExtremes.cluster_representatives["demand"].max() == pytest.approx(9.0)
    assert withExtremes.n_clusters == withoutExtremes.n_clusters + 1


# ================================= Integration with the energy system model =================================#


def test_esM_uses_the_tsam_interface(minimal_test_esM):
    minimal_test_esM.aggregateTemporally(
        n_clusters=1,
        period_duration=4 * minimal_test_esM.hoursPerTimeStep,
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
        segments=SegmentConfig(n_segments=3, representation="mean"),
        preserve_column_means=False,
        storeTSAinstance=True,
    )
    tsaInstance = minimal_test_esM.tsaInstance

    assert minimal_test_esM.segmentation is True
    assert minimal_test_esM.segmentsPerPeriod == [0, 1, 2]
    assert len(minimal_test_esM.timeStepsPerPeriod) == 4
    assert isinstance(tsaInstance, AggregationResult)
    assert tsaInstance.n_clusters == 1
    assert tsaInstance.clustering.cluster_config.method == "hierarchical"
    assert tsaInstance.clustering.segment_config.n_segments == 3
    assert minimal_test_esM.tsaBuildTime >= 0


def test_esM_stores_no_instance_unless_asked(minimal_test_esM):
    minimal_test_esM.aggregateTemporally(
        n_clusters=2,
        period_duration=2 * minimal_test_esM.hoursPerTimeStep,
        segments=None,
    )
    assert minimal_test_esM.tsaInstance is None
    # The build time is measured either way, the performance summary reports it.
    assert minimal_test_esM.tsaBuildTime >= 0


def test_esM_period_duration_accepts_a_timedelta_string(minimal_test_esM):
    minimal_test_esM.aggregateTemporally(
        n_clusters=2,
        period_duration=f"{2 * minimal_test_esM.hoursPerTimeStep}h",
        segments=None,
    )
    assert len(minimal_test_esM.timeStepsPerPeriod) == 2
    assert minimal_test_esM.segmentation is False


def test_esM_segments_none_switches_segmentation_off(minimal_test_esM):
    minimal_test_esM.aggregateTemporally(
        n_clusters=2,
        period_duration=2 * minimal_test_esM.hoursPerTimeStep,
        segments=None,
    )
    assert minimal_test_esM.segmentation is False


def test_esM_clamps_the_number_of_segments(minimal_test_esM):
    """More segments than time steps per period cannot be represented."""
    with pytest.warns(UserWarning, match="exceeds the number of time steps"):
        minimal_test_esM.aggregateTemporally(
            n_clusters=2,
            period_duration=2 * minimal_test_esM.hoursPerTimeStep,
            segments=SegmentConfig(n_segments=8),
            storeTSAinstance=True,
        )
    assert minimal_test_esM.segmentsPerPeriod == [0, 1]
    assert minimal_test_esM.tsaInstance.clustering.segment_config.n_segments == 2


def test_esM_forwards_further_tsam_arguments(minimal_test_esM):
    minimal_test_esM.aggregateTemporally(
        n_clusters=2,
        period_duration=2 * minimal_test_esM.hoursPerTimeStep,
        segments=None,
        round_decimals=2,
        storeTSAinstance=True,
    )
    typicalPeriods = minimal_test_esM.tsaInstance.cluster_representatives
    assert (typicalPeriods.round(2) == typicalPeriods).all().all()


def test_esM_does_not_warn_about_deprecation(minimal_test_esM):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        minimal_test_esM.aggregateTemporally(
            n_clusters=2,
            period_duration=2 * minimal_test_esM.hoursPerTimeStep,
            segments=None,
        )


def test_esM_rejects_unknown_keywords(minimal_test_esM):
    with pytest.raises(TypeError, match="notAKeyword"):
        minimal_test_esM.aggregateTemporally(
            n_clusters=2,
            period_duration=2 * minimal_test_esM.hoursPerTimeStep,
            notAKeyword=1,
        )
